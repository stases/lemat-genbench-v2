"""Run sequential MLIP passes with independent, refillable batched workers."""

import multiprocessing
import os
import time
import traceback
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from math import floor
from multiprocessing.connection import Connection, wait

import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from lemat_genbench.batched import models as batched_models
from lemat_genbench.batched import relaxer
from lemat_genbench.batched.memory import MemoryBudget, memory_error
from lemat_genbench.models.base import (
    get_energy_above_hull_from_total_energy,
    get_formation_energy_per_atom_from_total_energy,
)
from lemat_genbench.preprocess import multi_mlip_preprocess as multi
from lemat_genbench.preprocess import reference_energies
from lemat_genbench.preprocess.base import PreprocessorResult
from lemat_genbench.utils.logging import logger

START_METHOD = "spawn"
SILENCE_WARNING = 600  # seconds without a message from a worker before a warning
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MEMORY_MODEL = {
    "orb": (150 * 2**20, 4.16 * 2**20),
    "mace": (150 * 2**20, 4.59 * 2**20),
    "uma": (700 * 2**20, 5.38 * 2**20),
}
CUDA_CONTEXT_BYTES = 512 * 2**20
MEMORY_RESERVE_MIN = 2**30
MEMORY_RESERVE_FRACTION = 0.05
ATOM_MARGIN = 1.25
MIN_ATOM_BUDGET = 150
MEMORY_FRACTION = None  # assigned only in child processes


def _device_memory(device) -> tuple[int, int] | None:
    if torch.device(device).type != "cuda":
        return None
    return torch.cuda.mem_get_info(device)


def _limit_memory(device, fraction):
    if fraction is not None:
        # The allocator cap needs a device index; a bare "cuda" is the current device.
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(fraction, index)


def _request(connection, phase, memory):
    connection.send(("request", phase, memory.max_atoms))
    return connection.recv()


def _defer_failure(phase, error, final, is_memory=False):
    return not final and (
        is_memory
        or memory_error(error)
        or (
            error is not None
            and MEMORY_FRACTION is not None
            and phase in ("single_point", "embedding")
        )
    )


def _send_result(connection, phase, index, result, final):
    error = result.get("_error", result.get("error"))
    defer = _defer_failure(phase, error, final, result.get("_memory_error", False))
    kind = "deferred" if defer else "result"
    connection.send((kind, phase, index, multi._detach_tensors(result)))


def _raise_if_cuda_error(error):
    """A CUDA runtime failure can poison the worker; out-of-memory is not fatal."""
    if "CUDA error" in str(error) and not memory_error(error):
        raise RuntimeError(str(error))


def _single_points(connection, name, calculator, model, memory, final):
    while tasks := _request(connection, "single_point", memory):
        geometries, valid_tasks, failures = [], [], []
        for index, structure in tasks:
            try:
                atoms = structure.to_ase_atoms()
                if name == "uma":
                    atoms.info = {}
                geometries.append(atoms)
                valid_tasks.append((index, structure))
            except Exception as exc:
                _raise_if_cuda_error(exc)
                memory.record_error(exc)
                failures.append((index, structure, exc))
        predictions = relaxer._evaluate(model, geometries, memory) if geometries else []
        evaluated = [
            (index, structure, prediction)
            for (index, structure), prediction in zip(valid_tasks, predictions)
        ]
        for index, structure, prediction in evaluated + failures:
            error = prediction if isinstance(prediction, Exception) else None
            _raise_if_cuda_error(error)
            if _defer_failure("single_point", error, final):
                _send_result(
                    connection,
                    "single_point",
                    index,
                    {"error": str(error), "_memory_error": memory_error(error)},
                    final,
                )
                continue
            try:
                if error is not None:
                    # Match wrapper semantics, including MACE's energy=None result.
                    prediction = calculator.calculate_energy_forces(structure)
                result = multi._detach_tensors(
                    {"energy": prediction.energy, "forces": prediction.forces}
                )
                # Retain swallowed wrapper failures for main-process OOM accounting.
                if error is not None and prediction.energy is None:
                    result["_error"] = str(error)
                    result["_memory_error"] = memory_error(error)
            except Exception as exc:
                _raise_if_cuda_error(exc)
                memory.record_error(exc)
                result = {"error": str(exc), "_memory_error": memory_error(exc)}
            _send_result(connection, "single_point", index, result, final)


def _embeddings(connection, calculator, memory, final):
    while tasks := _request(connection, "embedding", memory):
        index, structure = tasks[0]
        try:
            embeddings = calculator.extract_embeddings(structure)
            result = multi._detach_tensors(
                {
                    "node_embeddings": embeddings.node_embeddings,
                    "graph_embedding": embeddings.graph_embedding,
                }
            )
        except Exception as exc:
            _raise_if_cuda_error(exc)
            memory.record_error(exc)
            result = {
                "node_embeddings": None,
                "graph_embedding": None,
                "_error": str(exc),
                "_memory_error": memory_error(exc),
            }
        _send_result(connection, "embedding", index, result, final)


def _relaxations(connection, name, calculator, model, config, memory, final):
    def jobs():
        while tasks := _request(connection, "relaxation", memory):
            index, structure = tasks[0]
            try:
                # Unlike UMA single points, LeMat relaxations preserve atoms.info.
                atoms = structure.to_ase_atoms()
                if name == "uma":
                    # Preserve fairchem's charge/spin defaults on the relaxing atoms.
                    calculator.ase_calc._validate_charge_and_spin(atoms)
            except Exception as exc:
                _raise_if_cuda_error(exc)
                memory.record_error(exc)
                _send_result(
                    connection,
                    "relaxation",
                    index,
                    {"error": str(exc), "_memory_error": memory_error(exc)},
                    final,
                )
                continue
            yield index, atoms

    settings = config.batched
    for outcome in relaxer.relax_stream(
        jobs(),
        model,
        fmax=config.relaxation_config.get("fmax", 0.02),
        steps=config.relaxation_config.get("steps", 50),
        max_in_flight=settings["structures_per_worker"],
        memory=memory,
        geometry_tol=settings["geometry_tolerance"],
        max_requests=settings["max_requests"],
    ):
        if not outcome.memory_error:
            _raise_if_cuda_error(outcome.error)
        # Only plain data crosses the pipe; in particular no calculator or model.
        if outcome.atoms is not None:
            outcome.atoms.calc = None
        _send_result(
            connection,
            "relaxation",
            outcome.index,
            {
                "atoms": outcome.atoms,
                "energy": outcome.energy,
                "steps": outcome.steps,
                "error": outcome.error,
                "cache_hits": outcome.cache_hits,
                "_memory_error": outcome.memory_error,
            },
            final,
        )


def _worker(connection, parent_connection, name, config, assignment, final):
    global MEMORY_FRACTION
    parent_connection.close()
    try:
        n_cpus = len(os.sched_getaffinity(0))
        torch.set_num_threads(max(1, n_cpus // config.batched["workers_per_model"]))
        model_config = multi._merge_model_config(
            name, config.mlip_configs.get(name, {})
        )
        device = model_config["device"]
        if assignment is None:
            try:
                device_memory = _device_memory(device)
            except Exception:
                # Even a failed probe is a worker with no allocator share.
                _limit_memory(device, None)
                MEMORY_FRACTION = None
                raise
            connection.send(("memory", device_memory))
            assignment = connection.recv()
        fraction, atom_budget = assignment
        _limit_memory(device, fraction)
        MEMORY_FRACTION = fraction
        memory = MemoryBudget(atom_budget, adaptive=not final)

        def phase_memory(phase):
            memory.on_error = lambda error: connection.send(
                ("memory_error", phase, str(error))
            )

        calculator = multi._get_or_create_calculator(name, model_config)
        # Loaders can enable TF32; enforce the chosen precision after loading.
        torch.set_float32_matmul_precision(config.matmul_precision)
        connection.send(
            (
                "ready",
                {
                    "mlip_model": calculator.__class__.__name__,
                    "model_config": getattr(calculator, "model_type", "unknown"),
                    "hull_type": calculator.hull_type,
                },
            )
        )
        if config.extract_embeddings:
            phase_memory("embedding")
            _embeddings(connection, calculator, memory, final)
        # UMA fixes patch the shared model, so extract embeddings before applying them.
        model = batched_models.make_batched_model(
            name, calculator, uma_fixes=config.batched["uma_fixes"]
        )
        phase_memory("single_point")
        _single_points(connection, name, calculator, model, memory, final)
        if config.relax_structures:
            phase_memory("relaxation")
            _relaxations(connection, name, calculator, model, config, memory, final)
        connection.send(("done",))
    except BaseException as exc:
        try:
            if memory_error(exc):
                connection.send(("load_oom", str(exc)))
            else:
                connection.send(("fatal", traceback.format_exc()))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


@dataclass
class _WorkerState:
    process: multiprocessing.Process
    connection: Connection
    # A phase/index stays held until its result has arrived, including any job
    # prefetched by relax_stream for the atom budget but not yet evaluated.
    held: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    last_message: float = field(default_factory=time.monotonic)


def _start_worker(context, name, config, assignment=None, final=False):
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_worker,
        args=(child, parent, name, config, assignment, final),
        daemon=True,
    )
    previous = {key: os.environ.get(key) for key in _THREAD_ENV}
    try:
        os.environ.update(dict.fromkeys(_THREAD_ENV, "1"))
        process.start()
    except BaseException:
        parent.close()
        raise
    finally:
        child.close()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return _WorkerState(process, parent)


def _close_worker(worker, terminate=False):
    if terminate and worker.process.is_alive():
        worker.process.terminate()
    worker.process.join(timeout=5)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join()
    worker.connection.close()
    worker.process.close()


def _energy_properties(result, energy, composition, hull_type, config, prefix=""):
    """Hull reference tables and all derived energies stay in the main process."""
    if config.calculate_formation_energy:
        try:
            result[prefix + "formation_energy"] = multi._detach_tensors(
                get_formation_energy_per_atom_from_total_energy(energy, composition)
            )
        except Exception:
            result[prefix + "formation_energy"] = None
    if config.calculate_energy_above_hull:
        try:
            result[prefix + "e_above_hull"] = multi._detach_tensors(
                get_energy_above_hull_from_total_energy(
                    energy, composition, hull_type=hull_type
                )
            )
        except Exception:
            result[prefix + "e_above_hull"] = None


def _relaxation_properties(structure, outcome, hull_type, config):
    result = {}
    try:
        if outcome["error"] is not None:
            raise RuntimeError(outcome["error"])
        relaxed = AseAtomsAdaptor.get_structure(outcome["atoms"])
        rmse = multi._calculate_rmse(structure, relaxed)
        result["relaxed_structure"] = multi._create_clean_structure_copy(relaxed)
        result["relaxation_rmse"] = rmse
        result["relaxation_energy"] = outcome["energy"]
        result["relaxation_steps"] = outcome["steps"]
        # Reuse the endpoint energy instead of evaluating after a pymatgen round
        # trip. UMA can differ for non-default charge/spin, which its single-point
        # wrapper clears before evaluation.
        _energy_properties(
            result,
            outcome["energy"],
            relaxed.composition,
            hull_type,
            config,
            prefix="relaxed_",
        )
    except Exception as exc:
        result["relaxation_failed"] = True
        result["relaxation_error"] = outcome["error"] or str(exc)
    return result


def _take_tasks(pending, phase, structures, settings, max_atoms):
    tasks, atom_count = [], 0
    limit = settings["structures_per_worker"] if phase == "single_point" else 1
    while pending and len(tasks) < limit:
        index = pending[0]
        count = len(structures[index])
        if tasks and max_atoms is not None and atom_count + count > max_atoms:
            break
        pending.popleft()
        tasks.append((index, structures[index]))
        atom_count += count
    return tasks


def _size_pass(device_memory, structures, name, settings):
    workers = min(settings["workers_per_model"], len(structures))
    budget = settings["max_atoms_per_worker"]
    fraction = share = None
    free = total = None
    if device_memory is not None:
        free, total = device_memory
        reserve = max(MEMORY_RESERVE_MIN, MEMORY_RESERVE_FRACTION * total)
        resident, per_atom = MEMORY_MODEL[name]
        minimum = max(MIN_ATOM_BUDGET, max(map(len, structures)))
        while True:
            share = (free - reserve) / workers - CUDA_CONTEXT_BYTES
            atoms = floor((share - resident) / (ATOM_MARGIN * per_atom))
            if workers == 1 or atoms >= minimum:
                break
            workers -= 1
        # Zero available allocator memory must fail as OOM, not as an invalid cap.
        share = max(0.0, share)
        fraction = share / total
        if budget == "auto":
            budget = max(1, atoms)
    elif budget == "auto":
        budget = None
    memory_info = (
        "no CUDA device"
        if device_memory is None
        else f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free"
    )
    share_info = "none" if share is None else f"{share / 2**30:.1f} GiB"
    logger.info(
        f"[{name}] Memory: {memory_info}, {workers} workers, "
        f"share {share_info}, atom budget {budget}"
    )
    return workers, (fraction, budget)


def _run_model_pass(structures, name, config, warnings):
    start = time.monotonic()
    settings = config.batched
    n_workers = min(settings["workers_per_model"], len(structures))
    # Independent queues let workers request enabled phases in any order.
    phases = ["single_point"]
    if config.extract_embeddings:
        phases.append("embedding")
    if config.relax_structures:
        phases.append("relaxation")
    pending = {phase: deque(range(len(structures))) for phase in phases}
    deferred = {phase: {} for phase in phases}
    completed = dict.fromkeys(phases, 0)
    next_progress = dict.fromkeys(phases, 10)
    results = [{} for _ in structures]
    context = multiprocessing.get_context(START_METHOD)
    workers = []
    deaths = relaxation_errors = request_caps = 0
    oom_errors = dict.fromkeys(("single_point", "embedding", "relaxation", "load"), 0)
    deferred_count = recovered = lost = 0
    assignment = None
    final = False

    def requeue(worker):
        for phase, indices in worker.held.items():
            pending[phase].extendleft(reversed(indices))

    def replace(worker, reason):
        nonlocal deaths
        deaths += 1
        _close_worker(worker, terminate=True)
        workers.remove(worker)
        # Name what the worker held, so a structure that kills workers can be found.
        held = {
            phase: list(indices) for phase, indices in worker.held.items() if indices
        }
        if deaths > 1:
            raise RuntimeError(
                f"[{name}] Second worker death, holding structures {held}: {reason}"
            )
        requeue(worker)
        logger.warning(
            f"[{name}] Restarting worker that held structures {held}, after: {reason}"
        )
        workers.append(_start_worker(context, name, config, assignment, final))

    try:
        # Only this child probes CUDA; it waits for its share before loading.
        workers.append(_start_worker(context, name, config))
        while True:
            if not workers:
                if not final and (any(pending.values()) or any(deferred.values())):
                    final = True
                    assignment = (None, None)
                    for phase, indices in deferred.items():
                        pending[phase].extend(indices)
                    workers.append(
                        _start_worker(context, name, config, assignment, final)
                    )
                else:
                    break
            ready = wait(
                [worker.connection for worker in workers]
                + [worker.process.sentinel for worker in workers],
                timeout=60,
            )
            now = time.monotonic()
            for worker in workers:
                # No timeouts in this mode; only make a stalled worker visible.
                if now - worker.last_message > SILENCE_WARNING:
                    held = {p: list(i) for p, i in worker.held.items() if i}
                    logger.warning(
                        f"[{name}] Worker silent for {now - worker.last_message:.0f} s, "
                        f"holding structures {held}"
                    )
                    worker.last_message = now
            for worker in list(workers):
                # Drain buffered replies before acting on an exit sentinel. This
                # preserves received results even when the process then dies.
                if worker.connection in ready:
                    try:
                        message = worker.connection.recv()
                        worker.last_message = time.monotonic()
                    except (EOFError, OSError) as exc:
                        replace(
                            worker,
                            f"pipe closed ({exc}); exitcode={worker.process.exitcode}",
                        )
                        continue
                    kind = message[0]
                    if kind == "memory":
                        n_workers, assignment = _size_pass(
                            message[1], structures, name, settings
                        )
                        try:
                            worker.connection.send(assignment)
                        except (EOFError, OSError) as exc:
                            replace(worker, f"sending memory share: {exc}")
                        for _ in range(n_workers - 1):
                            workers.append(
                                _start_worker(context, name, config, assignment)
                            )
                        if config.calculate_energy_above_hull:
                            hull_type = multi._merge_model_config(
                                name, config.mlip_configs.get(name, {})
                            ).get("hull_type", "dft")
                            # Workers have their shares and can load during warm-up.
                            for retrieve in (
                                reference_energies._retrieve_df,
                                reference_energies._retrieve_matrix,
                            ):
                                try:
                                    retrieve(hull_type)
                                except Exception:
                                    pass
                    elif kind == "ready":
                        worker.metadata = message[1]
                    elif kind == "memory_error":
                        oom_errors[message[1]] += 1
                    elif kind == "request":
                        _, phase, atom_budget = message
                        tasks = _take_tasks(
                            pending[phase], phase, structures, settings, atom_budget
                        )
                        held = worker.held.setdefault(phase, {})
                        held.update(dict.fromkeys(index for index, _ in tasks))
                        try:
                            worker.connection.send(tasks)
                        except (EOFError, OSError) as exc:
                            replace(worker, f"sending {phase} work: {exc}")
                    elif kind in ("result", "deferred"):
                        _, phase, index, data = message
                        del worker.held[phase][index]
                        result = results[index]
                        metadata = worker.metadata
                        error = data.pop("_error", None) or data.get("error")
                        is_oom = data.pop("_memory_error", False) or memory_error(error)
                        if kind == "deferred":
                            deferred[phase][index] = error
                            deferred_count += 1
                            continue
                        if final and is_oom:
                            warning = (
                                f"[{name}] structure {index}: {phase} lost after "
                                f"retrying with the whole GPU: {error}"
                            )
                            warnings.append(warning)
                            logger.warning(warning)
                        if final and index in deferred[phase]:
                            fields = {
                                "single_point": ("energy",),
                                "embedding": ("node_embeddings", "graph_embedding"),
                                "relaxation": ("atoms", "energy"),
                            }[phase]
                            computed = not error and all(
                                data.get(key) is not None for key in fields
                            )
                            recovered += computed
                            lost += not computed
                        elif final and is_oom:
                            lost += 1
                        if error:
                            if is_oom or phase == "embedding":
                                logger.warning(
                                    f"[{name}] Structure {index} {phase} failed: {error}"
                                )
                        # A dropped model result needs no relaxation properties.
                        if phase == "relaxation" and "error" not in result:
                            if settings["geometry_tolerance"] is not None and data.get(
                                "cache_hits", 0
                            ):
                                logger.info(
                                    f"[{name}] Structure {index}: {data['cache_hits']} cache hits"
                                )
                            request_caps += bool(
                                data["error"]
                                and data["error"].startswith("max_requests (")
                            )
                            data = _relaxation_properties(
                                structures[index], data, metadata["hull_type"], config
                            )
                            relaxation_errors += bool(data.get("relaxation_failed"))
                        if "error" not in result:
                            if "error" in data:
                                logger.warning(
                                    f"[{name}] Structure {index} dropped for this model: "
                                    f"{data['error']}"
                                )
                                results[index] = data
                            else:
                                result.update(
                                    {
                                        key: metadata[key]
                                        for key in ("mlip_model", "model_config")
                                    }
                                )
                                result.update(data)
                                if phase == "single_point":
                                    _energy_properties(
                                        result,
                                        data["energy"],
                                        structures[index].composition,
                                        metadata["hull_type"],
                                        config,
                                    )
                        completed[phase] += 1
                        percent = 100 * completed[phase] // len(structures)
                        if percent >= next_progress[phase]:
                            logger.info(
                                f"[{name}] {phase}: {completed[phase]}/{len(structures)}"
                            )
                            next_progress[phase] = (percent // 10 + 1) * 10
                    elif kind == "fatal":
                        replace(worker, message[1])
                    elif kind == "load_oom":
                        oom_errors["load"] += 1
                        requeue(worker)
                        _close_worker(worker)
                        workers.remove(worker)
                        if final:
                            raise RuntimeError(
                                f"[{name}] Could not load model with the whole GPU: {message[1]}"
                            )
                        if assignment is None:
                            n_workers = 1
                            logger.info(
                                f"[{name}] Memory: unavailable, 1 worker, "
                                "share none, atom budget None"
                            )
                        logger.warning(
                            f"[{name}] Dropping worker after model load memory error: {message[1]}"
                        )
                    elif kind == "done":
                        _close_worker(worker)
                        workers.remove(worker)
                elif worker.process.sentinel in ready:
                    # A pipe may become readable after wait returned its snapshot.
                    if not worker.connection.poll():
                        replace(
                            worker,
                            f"process exited with code {worker.process.exitcode}",
                        )
        if any(completed[phase] != len(structures) for phase in phases):
            raise RuntimeError(f"[{name}] Workers exited with unfinished structures")
    finally:
        for worker in workers:
            _close_worker(worker, terminate=True)
    elapsed = time.monotonic() - start
    log = logger.warning if deferred_count or lost else logger.info
    log(
        f"[{name}] Batched pass: {n_workers} workers, {elapsed:.2f}s, "
        f"{len(structures) / elapsed:.2f} structures/s, "
        f"{relaxation_errors} relaxation errors, {request_caps} max_requests hits, "
        f"out-of-memory errors: {oom_errors['single_point']} single points, "
        f"{oom_errors['embedding']} embeddings, {oom_errors['relaxation']} relaxations; "
        f"retries={sum(oom_errors.values())}, deferred={deferred_count}, "
        f"recovered={recovered}, lost={lost}"
    )
    return results


def run_batched_stage(
    structures: list[Structure], config: multi.MultiMLIPStabilityPreprocessorConfig
) -> PreprocessorResult:
    """Evaluate one model at a time, then apply the shared property merge in order.

    Start at most one daemon worker per structure, up to workers_per_model.
    Workers extract embeddings before applying UMA fixes. CUDA runtime errors
    kill the worker. Capped workers defer failed single points and embeddings;
    relaxation failures are deferred only for memory errors. A final worker runs
    deferred phases without a share. Remaining memory losses become result warnings.
    The first worker death re-queues held work and starts one replacement; a
    second death propagates. Empty input starts no processes and loads no models.
    """
    start = time.monotonic()
    mlip_results = [{} for _ in structures]
    warnings = []
    if structures:
        for name in config.mlip_names:
            results = _run_model_pass(structures, name, config, warnings)
            for combined, result in zip(mlip_results, results):
                combined[name] = result
    processed, failed = [], []
    for index, (structure, results) in enumerate(zip(structures, mlip_results)):
        try:
            processed.append(multi._merge_mlip_results(deepcopy(structure), results))
        except Exception as exc:
            failed.append(index)
            warnings.append(f"Failed to process structure {index}: {exc}")
            logger.warning(
                f"Failed to process structure {index} for {config.name}", exc_info=True
            )
    return PreprocessorResult(
        processed_structures=processed,
        config=config,
        computation_time=time.monotonic() - start,
        n_input_structures=len(structures),
        failed_indices=failed,
        warnings=warnings,
    )

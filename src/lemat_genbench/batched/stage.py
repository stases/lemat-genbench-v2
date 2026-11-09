"""Run sequential MLIP passes with independent, refillable batched workers."""

import multiprocessing
import os
import time
import traceback
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait

import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from lemat_genbench.batched import models as batched_models
from lemat_genbench.batched import relaxer
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


def _request(connection, phase):
    connection.send(("request", phase))
    return connection.recv()


def _send_result(connection, phase, index, result):
    connection.send(("result", phase, index, multi._detach_tensors(result)))


def _raise_if_cuda_error(error):
    """A CUDA runtime failure can poison the worker; out-of-memory is not fatal."""
    if "CUDA error" in str(error):
        raise RuntimeError(str(error))


def _single_points(connection, name, calculator, model):
    while tasks := _request(connection, "single_point"):
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
                failures.append((index, structure, exc))
        predictions = relaxer._evaluate(model, geometries) if geometries else []
        evaluated = [
            (index, structure, prediction)
            for (index, structure), prediction in zip(valid_tasks, predictions)
        ]
        for index, structure, prediction in evaluated + failures:
            error = prediction if isinstance(prediction, Exception) else None
            _raise_if_cuda_error(error)
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
            except Exception as exc:
                _raise_if_cuda_error(exc)
                result = {"error": str(exc)}
            _send_result(connection, "single_point", index, result)


def _embeddings(connection, calculator):
    while tasks := _request(connection, "embedding"):
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
            result = {
                "node_embeddings": None,
                "graph_embedding": None,
                "_error": str(exc),
            }
        _send_result(connection, "embedding", index, result)


def _relaxations(connection, name, calculator, model, config):
    def jobs():
        while tasks := _request(connection, "relaxation"):
            index, structure = tasks[0]
            try:
                # Unlike UMA single points, LeMat relaxations preserve atoms.info.
                atoms = structure.to_ase_atoms()
                if name == "uma":
                    # Preserve fairchem's charge/spin defaults on the relaxing atoms.
                    calculator.ase_calc._validate_charge_and_spin(atoms)
            except Exception as exc:
                _raise_if_cuda_error(exc)
                _send_result(connection, "relaxation", index, {"error": str(exc)})
                continue
            yield index, atoms

    settings = config.batched
    for outcome in relaxer.relax_stream(
        jobs(),
        model,
        fmax=config.relaxation_config.get("fmax", 0.02),
        steps=config.relaxation_config.get("steps", 50),
        max_in_flight=settings["structures_per_worker"],
        max_atoms=settings["max_atoms_per_worker"],
        geometry_tol=settings["geometry_tolerance"],
        max_requests=settings["max_requests"],
    ):
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
            },
        )


def _worker(connection, parent_connection, name, config):
    parent_connection.close()
    try:
        n_cpus = len(os.sched_getaffinity(0))
        torch.set_num_threads(max(1, n_cpus // config.batched["workers_per_model"]))
        calculator = multi._get_or_create_calculator(
            name, multi._merge_model_config(name, config.mlip_configs.get(name, {}))
        )
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
            _embeddings(connection, calculator)
        # UMA fixes patch the shared model, so extract embeddings before applying them.
        model = batched_models.make_batched_model(
            name, calculator, uma_fixes=config.batched["uma_fixes"]
        )
        _single_points(connection, name, calculator, model)
        if config.relax_structures:
            _relaxations(connection, name, calculator, model, config)
        connection.send(("done",))
    except BaseException:
        try:
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


def _start_worker(context, name, config):
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_worker, args=(child, parent, name, config), daemon=True
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


def _take_tasks(pending, phase, structures, settings):
    tasks, atom_count = [], 0
    limit = settings["structures_per_worker"] if phase == "single_point" else 1
    while pending and len(tasks) < limit:
        index = pending[0]
        count = len(structures[index])
        max_atoms = settings["max_atoms_per_worker"]
        if tasks and max_atoms is not None and atom_count + count > max_atoms:
            break
        pending.popleft()
        tasks.append((index, structures[index]))
        atom_count += count
    return tasks


def _run_model_pass(structures, name, config):
    start = time.monotonic()
    settings = config.batched
    n_workers = min(settings["workers_per_model"], len(structures))
    logger.info(f"[{name}] Starting batched pass with {n_workers} workers")
    # Independent queues let workers request enabled phases in any order.
    phases = ["single_point"]
    if config.extract_embeddings:
        phases.append("embedding")
    if config.relax_structures:
        phases.append("relaxation")
    pending = {phase: deque(range(len(structures))) for phase in phases}
    completed = dict.fromkeys(phases, 0)
    next_progress = dict.fromkeys(phases, 10)
    results = [{} for _ in structures]
    context = multiprocessing.get_context(START_METHOD)
    workers = []
    deaths = relaxation_errors = request_caps = 0
    oom_errors = dict.fromkeys(("single_point", "embedding", "relaxation"), 0)

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
        for phase, indices in worker.held.items():
            pending[phase].extendleft(reversed(indices))
        logger.warning(
            f"[{name}] Restarting worker that held structures {held}, after: {reason}"
        )
        workers.append(_start_worker(context, name, config))

    try:
        for _ in range(n_workers):
            workers.append(_start_worker(context, name, config))
        if config.calculate_energy_above_hull:
            hull_type = multi._merge_model_config(
                name, config.mlip_configs.get(name, {})
            ).get("hull_type", "dft")
            # Warm main-process tables while workers load their models.
            for retrieve in (
                reference_energies._retrieve_df,
                reference_energies._retrieve_matrix,
            ):
                try:
                    retrieve(hull_type)
                except Exception:
                    pass
        while workers:
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
                    if kind == "ready":
                        worker.metadata = message[1]
                    elif kind == "request":
                        phase = message[1]
                        tasks = _take_tasks(pending[phase], phase, structures, settings)
                        held = worker.held.setdefault(phase, {})
                        held.update(dict.fromkeys(index for index, _ in tasks))
                        try:
                            worker.connection.send(tasks)
                        except (EOFError, OSError) as exc:
                            replace(worker, f"sending {phase} work: {exc}")
                    elif kind == "result":
                        _, phase, index, data = message
                        del worker.held[phase][index]
                        result = results[index]
                        metadata = worker.metadata
                        error = data.pop("_error", None) or data.get("error")
                        if error:
                            is_oom = "out of memory" in error.lower()
                            oom_errors[phase] += is_oom
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
    logger.info(
        f"[{name}] Batched pass: {n_workers} workers, {elapsed:.2f}s, "
        f"{len(structures) / elapsed:.2f} structures/s, "
        f"{relaxation_errors} relaxation errors, {request_caps} max_requests hits, "
        f"out-of-memory errors: {oom_errors['single_point']} single points, "
        f"{oom_errors['embedding']} embeddings, {oom_errors['relaxation']} relaxations"
    )
    return results


def run_batched_stage(
    structures: list[Structure], config: multi.MultiMLIPStabilityPreprocessorConfig
) -> PreprocessorResult:
    """Evaluate one model at a time, then apply the shared property merge in order.

    Start at most one daemon worker per structure, up to workers_per_model.
    Workers extract embeddings before applying UMA fixes. CUDA runtime errors
    kill the worker; out-of-memory results remain per structure and are logged.
    The first worker death re-queues held work and starts one replacement; a
    second death propagates. Empty input starts no processes and loads no models.
    """
    start = time.monotonic()
    mlip_results = [{} for _ in structures]
    if structures:
        for name in config.mlip_names:
            results = _run_model_pass(structures, name, config)
            for combined, result in zip(mlip_results, results):
                combined[name] = result
    processed, failed, warnings = [], [], []
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

"""CPU checks for memory errors that must survive bisection and pipe transport."""

import multiprocessing
import os
import time
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from test_stage import (
    GIB,
    HULL_TYPE,
    MEMORY,
    FakeBatchedModel,
    FakeCalculator,
    make_structures,
    recorded_limits,
)
from test_stage import fakes as fakes

from lemat_genbench.batched import relaxer, stage
from lemat_genbench.batched.memory import MemoryBudget


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA error: custom allocator exhausted"),
        RuntimeError("CUDA out of memory"),
        RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED"),
        RuntimeError("CUBLAS_STATUS_NOT_INITIALIZED"),
        RuntimeError("CUSOLVER_STATUS_ALLOC_FAILED"),
        RuntimeError("CUSOLVER_STATUS_INTERNAL_ERROR in cusolverDnCreate(handle)"),
        RuntimeError("CUDNN_STATUS_ALLOC_FAILED"),
    ],
)
def test_library_memory_errors_clear_cache_and_are_not_fatal(monkeypatch, error):
    cleared, observed = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))

    class FailingModel:
        def predict(self, atoms):
            raise error

    assert stage.memory_error(error)
    stage._raise_if_cuda_error(error)
    assert relaxer._evaluate(
        FailingModel(), [Atoms("Cu")], MemoryBudget(on_error=observed.append)
    ) == [error]
    assert cleared == [True]
    assert observed == [error]


def test_unrelated_cuda_errors_stay_fatal():
    error = RuntimeError("CUDA error: illegal memory access")
    assert not stage.memory_error(error)
    assert not stage.memory_error("CUSOLVER_STATUS_INTERNAL_ERROR in a solve")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        stage._raise_if_cuda_error(error)


def test_learned_budget_splits_requests_already_in_flight():
    calls, failures = [], []
    geometries = [Atoms("Cu4") for _ in range(3)]

    class LimitedModel:
        def predict(self, atoms):
            count = sum(map(len, atoms))
            calls.append(count)
            if count > 4:
                raise RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
            return [len(a) for a in atoms]

    memory = MemoryBudget(100, failures.append)
    model = LimitedModel()
    assert relaxer._evaluate(model, geometries, memory) == [4, 4, 4]
    assert memory.max_atoms == 4
    assert len(failures) == 2
    calls.clear()
    assert relaxer._evaluate(model, geometries, memory) == [4, 4, 4]
    assert calls == [4, 4, 4]
    assert len(failures) == 2


@pytest.mark.parametrize("phase", ["single_point", "embedding", "relaxation"])
def test_typed_memory_error_is_deferred_without_wrapper_fallback(phase):
    error = torch.OutOfMemoryError("CUDA error: custom allocator exhausted")
    atoms = Atoms("Cu", cell=[3, 3, 3], pbc=True)
    tasks = deque([[(7, AseAtomsAdaptor.get_structure(atoms))], []])
    messages, errors = [], []
    connection = SimpleNamespace(send=messages.append, recv=tasks.popleft)
    memory = MemoryBudget(on_error=errors.append)

    class FailingModel:
        def predict(self, atoms):
            raise error

        def extract_embeddings(self, structure):
            raise error

        def calculate_energy_forces(self, structure):
            pytest.fail("A deferred single point must not run the wrapper")

    model = FailingModel()
    if phase == "single_point":
        stage._single_points(connection, "orb", model, model, memory, False)
    elif phase == "embedding":
        stage._embeddings(connection, model, memory, False)
    else:
        config = SimpleNamespace(
            batched={
                "structures_per_worker": 2,
                "geometry_tolerance": None,
                "max_requests": 1000,
            },
            relaxation_config={"steps": 0},
        )
        stage._relaxations(connection, "orb", model, model, config, memory, False)
    deferred = [message for message in messages if message[0] == "deferred"]
    assert len(deferred) == 1
    assert deferred[0][1:3] == (phase, 7)
    assert deferred[0][3]["_memory_error"] is True
    assert len(errors) == 1
    assert not any(message[0] == "result" for message in messages)


def test_memory_helpers_use_configured_device(monkeypatch):
    probes, limits = [], []

    def probe(device):
        probes.append(device)
        return (2**30, 4 * 2**30)

    monkeypatch.setattr(torch.cuda, "mem_get_info", probe)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: limits.append((fraction, device)),
    )
    assert stage._device_memory("cpu") is None
    assert probes == []
    assert stage._device_memory("cuda:3") == (2**30, 4 * 2**30)
    stage._limit_memory("cuda:3", None)
    assert limits == []
    stage._limit_memory("cuda:3", 0.25)
    assert probes == ["cuda:3"]
    # torch requires an index here; the runner passes a bare cuda device.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    stage._limit_memory(torch.device("cuda"), 0.5)
    assert limits == [(0.25, 3), (0.5, 1)]
    assert stage.MEMORY_FRACTION is None


def _mace_config(*, embeddings=False, hull=False, workers=1):
    return stage.multi.MultiMLIPStabilityPreprocessor(
        mlip_names=["mace"],
        mlip_configs={"mace": {"device": "cpu", "hull_type": HULL_TYPE["mace"]}},
        execution_mode="batched",
        relax_structures=False,
        extract_embeddings=embeddings,
        calculate_formation_energy=False,
        calculate_energy_above_hull=hull,
        batched={
            "workers_per_model": workers,
            "structures_per_worker": 2,
            "max_atoms_per_worker": 10,
        },
    ).config


@pytest.mark.usefixtures("fakes")
def test_workers_receive_shares_during_hull_warmup(monkeypatch):
    MEMORY["device"] = (40 * GIB, 48 * GIB)
    reader, writer = multiprocessing.get_context("fork").Pipe(duplex=False)
    observed, messages = [], []
    retrieve = stage.reference_energies._retrieve_df

    def limit_memory(device, fraction):
        writer.send((os.getpid(), fraction))

    def slow_warmup(hull_type="dft", threshold=0.001):
        if not observed:
            # Block the main loop inside warm-up. Both children must receive
            # their shares without the main process handling another message.
            shares = []
            deadline = time.monotonic() + 5
            while len(shares) < 2 and reader.poll(max(0, deadline - time.monotonic())):
                shares.append(reader.recv())
            observed.append(shares)
        return retrieve(hull_type, threshold)

    monkeypatch.setattr(stage, "_limit_memory", limit_memory)
    monkeypatch.setattr(stage.reference_energies, "_retrieve_df", slow_warmup)
    monkeypatch.setattr(stage.logger, "info", messages.append)
    try:
        result = stage.run_batched_stage(
            make_structures()[:2], _mace_config(hull=True, workers=2)
        )
    finally:
        reader.close()
        writer.close()
    assert result.failed_indices == []
    (shares,) = observed
    assert len({pid for pid, _ in shares}) == 2
    assert all(fraction == pytest.approx(18.3 / 48) for _, fraction in shares)
    assert [message for message in messages if "Memory:" in message] == [
        "[mace] Memory: 40.0 of 48.0 GiB free, 2 workers, "
        "share 18.3 GiB, atom budget 10"
    ]
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("phase", ["single_point", "embedding"])
@pytest.mark.parametrize("computed", [False, True], ids=["lost", "recovered"])
@pytest.mark.parametrize("capped", [True, False], ids=["capped", "uncapped"])
@pytest.mark.usefixtures("fakes")
def test_nonmemory_failures_are_deferred_and_counted(
    monkeypatch, tmp_path, phase, computed, capped
):
    MEMORY.update(device=(40 * GIB, 48 * GIB) if capped else None, dir=tmp_path)
    calls = tmp_path / "phase_calls.txt"
    original_wrapper = FakeCalculator.calculate_energy_forces
    original_embedding = FakeCalculator.extract_embeddings

    def record_call():
        with calls.open("a") as log:
            log.write(f"{stage.MEMORY_FRACTION}\n")

    def failed_batch(self, atoms):
        raise RuntimeError("opaque batched failure")

    def wrapper(self, structure):
        record_call()
        if computed:
            return original_wrapper(self, structure)
        # MACE swallows even allocation errors without giving the caller a cause.
        return SimpleNamespace(energy=None, forces=None)

    def embedding(self, structure):
        record_call()
        if stage.MEMORY_FRACTION is not None or not computed:
            raise RuntimeError("opaque embedding failure")
        return original_embedding(self, structure)

    if phase == "single_point":
        monkeypatch.setattr(FakeBatchedModel, "predict", failed_batch)
        monkeypatch.setattr(FakeCalculator, "calculate_energy_forces", wrapper)
    else:
        monkeypatch.setattr(FakeCalculator, "extract_embeddings", embedding)
    messages = []
    monkeypatch.setattr(stage.logger, "info", messages.append)
    monkeypatch.setattr(stage.logger, "warning", messages.append)
    result = stage.run_batched_stage(
        make_structures()[:1], _mace_config(embeddings=phase == "embedding")
    )
    fractions = [fraction for _, fraction in recorded_limits()]
    if capped:
        assert len(fractions) == 2 and fractions[0] is not None
        assert fractions[1] is None
    else:
        assert fractions == [None]
    # The wrapper runs only without a share. Embeddings are attempted under
    # the cap first, then once more in the final worker.
    expected_calls = fractions if phase == "embedding" else [None]
    assert calls.read_text().splitlines() == list(map(str, expected_calls))
    (structure,) = result.processed_structures
    key = "energy_mace" if phase == "single_point" else "graph_embedding_mace"
    assert (structure.properties.get(key) is not None) is computed
    (summary,) = [message for message in messages if "Batched pass:" in message]
    assert summary.endswith(
        f"retries=0, deferred={int(capped)}, "
        f"recovered={int(capped and computed)}, lost={int(capped and not computed)}"
    )
    # Broader missing-value warnings belong to the next round.
    assert result.warnings == []
    assert multiprocessing.active_children() == []

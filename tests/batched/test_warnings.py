"""CPU checks that every missing batched value has one final diagnostic."""

import math
import multiprocessing
import re

import pytest
from test_memory import _mace_config
from test_stage import (
    GIB,
    MEMORY,
    MEMORY_ERRORS,
    FakeBatchedModel,
    FakeCalculator,
    make_structures,
)
from test_stage import fakes as fakes

from lemat_genbench.batched import stage

pytestmark = pytest.mark.usefixtures("fakes")


@pytest.fixture
def logged(monkeypatch):
    messages = []
    monkeypatch.setattr(stage.logger, "warning", messages.append)
    monkeypatch.setattr(stage.logger, "info", messages.append)
    return messages


def assert_warnings(result, logged, expected):
    assert sorted(result.warnings) == sorted(expected)
    assert sorted(
        message for message in logged if message.startswith("[mace] structure ")
    ) == sorted(expected)
    assert len(set(result.warnings)) == len(result.warnings)


def counts(logged):
    (summary,) = [message for message in logged if "Batched pass:" in message]
    return {key: int(value) for key, value in re.findall(r"\b([a-z_]+)=(\d+)", summary)}


def relax_config(*, steps=0, embeddings=False):
    config = _mace_config(embeddings=embeddings)
    config.relax_structures = True
    config.relaxation_config = {"steps": steps, "fmax": 1e-12}
    return config


@pytest.mark.parametrize("retrieve", ["_retrieve_df", "_retrieve_matrix"])
def test_hull_warmup_failure_is_fatal(monkeypatch, retrieve):
    def fail(hull_type):
        raise OSError("hull table unavailable")

    monkeypatch.setattr(stage.reference_energies, retrieve, fail)
    with pytest.raises(
        RuntimeError, match=r"\[mace\].*mace_mp.*hull table unavailable"
    ):
        stage.run_batched_stage(make_structures()[:1], _mace_config(hull=True))
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("relax", [False, True])
def test_derived_energy_failures_are_reported_once(monkeypatch, logged, relax):
    def formation(*args, **kwargs):
        raise ValueError("formation reference missing")

    def hull(*args, **kwargs):
        raise ValueError("terminal entry missing")

    monkeypatch.setattr(
        stage, "get_formation_energy_per_atom_from_total_energy", formation
    )
    monkeypatch.setattr(stage, "get_energy_above_hull_from_total_energy", hull)
    config = relax_config() if relax else _mace_config()
    config.calculate_formation_energy = config.calculate_energy_above_hull = True
    result = stage.run_batched_stage(make_structures()[:1], config)
    expected = [
        f"[mace] structure 0: {prefix}{quantity} failed: {error}"
        for prefix in (["", "relaxed_"] if relax else [""])
        for quantity, error in (
            ("formation_energy", "formation reference missing"),
            ("e_above_hull", "terminal entry missing"),
        )
    ]
    assert_warnings(result, logged, expected)
    assert counts(logged)["lost"] == len(expected)
    assert counts(logged)["memory_lost"] == 0
    props = result.processed_structures[0].properties
    assert props["energy_mace"] is not None
    if relax:
        assert props["relaxed_structure_mace"] is not None
        assert "relaxation_failed_mace" not in props
    assert all(
        f"{prefix}{quantity}_mace" not in props
        for prefix in ("", "relaxed_")
        for quantity in ("formation_energy", "e_above_hull")
    )


@pytest.mark.parametrize("failure", ["line search", "max_requests", "non-finite"])
def test_relaxation_failure_has_one_warning(monkeypatch, logged, failure):
    original = FakeBatchedModel.predict

    def predict(self, atoms):
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls > 1 and failure == "line search":
            raise RuntimeError("line search failed")
        predictions = original(self, atoms)
        if self.calls > 1 and failure == "non-finite":
            for prediction in predictions:
                prediction.energy = float("nan")
        return predictions

    monkeypatch.setattr(FakeBatchedModel, "predict", predict)
    config = relax_config(steps=3)
    if failure == "max_requests":
        config.batched["max_requests"] = 1
    result = stage.run_batched_stage(make_structures()[1:2], config)
    error = {
        "line search": "line search failed",
        "max_requests": "max_requests (1) exceeded",
        "non-finite": "Model prediction was not finite",
    }[failure]
    assert_warnings(result, logged, [f"[mace] structure 0: relaxation failed: {error}"])
    props = result.processed_structures[0].properties
    assert props["relaxation_failed_mace"] is True
    assert props["relaxation_error_mace"] == error
    assert counts(logged)["lost"] == 1
    assert counts(logged)["unconverged"] == 0


def test_relaxation_postprocessing_failure_is_reported(monkeypatch, logged):
    def fail(*args):
        raise ValueError("alignment failed")

    monkeypatch.setattr(stage.multi, "_calculate_rmse", fail)
    result = stage.run_batched_stage(make_structures()[:1], relax_config())
    assert_warnings(
        result, logged, ["[mace] structure 0: relaxation failed: alignment failed"]
    )
    assert counts(logged)["lost"] == 1


@pytest.mark.parametrize("energy", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_single_point_is_preserved_and_reported(monkeypatch, logged, energy):
    original = FakeBatchedModel.predict

    def predict(self, atoms):
        predictions = original(self, atoms)
        for prediction in predictions:
            prediction.energy = energy
        return predictions

    monkeypatch.setattr(FakeBatchedModel, "predict", predict)
    result = stage.run_batched_stage(make_structures()[:1], _mace_config())
    got = result.processed_structures[0].properties["energy_mace"]
    assert math.isnan(got) if math.isnan(energy) else got == energy
    assert_warnings(
        result,
        logged,
        [f"[mace] structure 0: single_point energy is not finite: {energy}"],
    )
    assert counts(logged)["lost"] == counts(logged)["memory_lost"] == 0


def test_unconverged_endpoints_are_valid_and_counted(logged):
    result = stage.run_batched_stage(make_structures()[:2], relax_config())
    assert_warnings(result, logged, [])
    assert counts(logged)["unconverged"] == 2
    assert counts(logged)["lost"] == 0
    for structure in result.processed_structures:
        assert structure.properties["relaxation_steps_mace"] == 0
        assert structure.properties["relaxed_structure_mace"] is not None
        assert "relaxation_failed_mace" not in structure.properties


@pytest.mark.parametrize("permanent", [False, True], ids=["recovered", "lost"])
def test_memory_retries_only_warn_for_final_losses(logged, permanent):
    MEMORY.update(
        device=(40 * GIB, 48 * GIB), faults={4: "always" if permanent else "oom"}
    )
    result = stage.run_batched_stage(
        make_structures()[1:2], relax_config(embeddings=True)
    )
    expected = (
        []
        if not permanent
        else [
            f"[mace] structure 0: {phase} lost after retrying with the whole GPU: {MEMORY_ERRORS['oom']()}"
            for phase in ("single_point", "embedding", "relaxation")
        ]
    )
    assert_warnings(result, logged, expected)
    summary = counts(logged)
    assert summary["deferred"] == 3
    assert summary["recovered"] == (0 if permanent else 3)
    assert summary["lost"] == summary["memory_lost"] == (3 if permanent else 0)


def test_whole_model_drop_reports_discarded_phases(monkeypatch, logged):
    original = FakeBatchedModel.predict

    def predict(self, atoms):
        if not getattr(self, "called", False):
            self.called = True
            raise ValueError("single point failed")
        return original(self, atoms)

    def wrapper(self, structure):
        raise ValueError("single point failed")

    monkeypatch.setattr(FakeBatchedModel, "predict", predict)
    monkeypatch.setattr(FakeCalculator, "calculate_energy_forces", wrapper)
    result = stage.run_batched_stage(
        make_structures()[:1], relax_config(embeddings=True)
    )
    assert_warnings(
        result,
        logged,
        [
            f"[mace] structure 0: {phase} failed: single point failed"
            for phase in ("single_point", "embedding", "relaxation")
        ],
    )
    assert (
        result.processed_structures[0].properties["mlip_error_mace"]
        == "single point failed"
    )
    assert counts(logged)["lost"] == 3


def test_shared_merge_keeps_model_error_reason():
    structure = stage.multi._merge_mlip_results(
        make_structures()[0],
        {"mace": {"error": "model failed"}, "orb": {"energy": 1.0}},
    )
    assert structure.properties["mlip_error_mace"] == "model failed"
    assert "energy_mace" not in structure.properties
    assert (
        structure.properties["energy_orb"] == structure.properties["energy_mean"] == 1.0
    )

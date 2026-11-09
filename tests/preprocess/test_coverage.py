"""Small, offline checks for preprocessing coverage and runner failure handling."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from lemat_genbench.preprocess.base import PreprocessorConfig, PreprocessorResult
from lemat_genbench.preprocess.coverage import (
    coverage_warnings,
    mlip_coverage,
    preprocessor_coverage,
)


def structure(**properties):
    return Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]], properties=properties)


def result(structures, n_input=None, failed_indices=(), warnings=(), config=None):
    return PreprocessorResult(
        processed_structures=structures,
        config=config if config is not None else PreprocessorConfig(name="test"),
        computation_time=0.0,
        n_input_structures=len(structures) if n_input is None else n_input,
        failed_indices=list(failed_indices),
        warnings=list(warnings),
    )


def complete_structure(models):
    properties = {"e_above_hull_n_mlips": len(models)}
    for model in models:
        properties.update(
            {
                f"energy_{model}": 0.0,
                f"formation_energy_{model}": -1.0,
                f"e_above_hull_{model}": 0.0,
                f"relaxation_rmse_{model}": 0.1,
                f"relaxed_e_above_hull_{model}": 0.0,
                f"relaxation_failed_{model}": False,
                f"graph_embedding_{model}": [1.0, 2.0],
            }
        )
    return structure(**properties)


@pytest.fixture
def gappy_structures():
    structures = [complete_structure(["mace", "uma"]) for _ in range(4)]
    del structures[0].properties["energy_mace"]
    structures[0].properties["graph_embedding_mace"] = None
    del structures[0].properties["e_above_hull_n_mlips"]
    for index, reason in enumerate(
        ["CUDA out of memory\ntraceback", "solver failure", "CUDA out of memory\nmore"]
    ):
        structures[index].properties.update(
            relaxation_failed_uma=True,
            relaxation_error_uma=reason,
            relaxation_rmse_uma=None,
            relaxed_e_above_hull_uma=None,
        )
    structures[1].properties.update(
        mlip_error_mace="timeout\ntraceback",
        formation_energy_mace=None,
        e_above_hull_mace=None,
        e_above_hull_n_mlips=1,
    )
    structures[2].properties.update(energy_mace=np.nan)
    structures[3].properties.update(
        graph_embedding_uma=np.array([np.inf, -np.inf, np.nan])
    )
    return structures


def test_mlip_coverage_counts(gappy_structures):
    coverage = mlip_coverage(gappy_structures, ["mace", "uma"], True, True)
    assert coverage == {
        "n_structures": 4,
        "models": {
            "mace": {
                "energy": 3,
                "formation_energy": 3,
                "e_above_hull": 3,
                "relaxation_rmse": 4,
                "relaxed_e_above_hull": 4,
                "graph_embedding": 3,
                "relaxation_failed": 0,
                "relaxation_errors": {},
                "model_errors": {"timeout": 1},
                "non_finite": {"energy": 1},
            },
            "uma": {
                "energy": 4,
                "formation_energy": 4,
                "e_above_hull": 4,
                "relaxation_rmse": 1,
                "relaxed_e_above_hull": 1,
                "graph_embedding": 4,
                "relaxation_failed": 3,
                "relaxation_errors": {"CUDA out of memory": 2, "solver failure": 1},
                "model_errors": {},
                "non_finite": {"graph_embedding": 1},
            },
        },
        "e_above_hull_n_mlips": {"0": 1, "1": 1, "2": 2},
    }
    # Reading coverage must not replace or sanitize a metric's inputs.
    assert np.isnan(gappy_structures[2].properties["energy_mace"])
    assert np.isinf(gappy_structures[3].properties["graph_embedding_uma"][0])
    json.dumps(coverage, allow_nan=False)


@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("embeddings", [False, True])
def test_requested_quantities_only(gappy_structures, relaxed, embeddings):
    coverage = mlip_coverage(gappy_structures, ["mace", "uma"], relaxed, embeddings)
    assert ("e_above_hull_n_mlips" in coverage) == relaxed
    for counts in coverage["models"].values():
        expected = {"energy", "model_errors", "non_finite"}
        if relaxed:
            expected.update(
                {
                    "formation_energy",
                    "e_above_hull",
                    "relaxation_rmse",
                    "relaxed_e_above_hull",
                    "relaxation_failed",
                    "relaxation_errors",
                }
            )
        if embeddings:
            expected.add("graph_embedding")
        assert set(counts) == expected
        assert ("graph_embedding" in counts["non_finite"]) <= embeddings


def test_error_reasons_are_short_first_lines():
    structures = [
        structure(
            mlip_error_mace="x" * 140 + "\ntraceback",
            relaxation_failed_mace=True,
            relaxation_error_mace="y" * 140 + "\r\ntraceback",
        ),
        structure(mlip_error_mace="x" * 120 + "different suffix"),
    ]
    counts = mlip_coverage(structures, ["mace"], True, False)["models"]["mace"]
    assert counts["model_errors"] == {"x" * 120: 2}
    assert counts["relaxation_errors"] == {"y" * 120: 1}


def test_preprocessor_coverage():
    processed = result([structure()], 3, [0, 2], ["failure one", "failure two"])
    assert preprocessor_coverage(processed) == {
        "n_input": 3,
        "n_output": 1,
        "failed_indices": [0, 2],
        "warnings": ["failure one", "failure two"],
    }


def test_coverage_warnings_one_per_gap(gappy_structures):
    coverage = {
        "validity": preprocessor_coverage(result(gappy_structures, 6, [1, 3])),
        "preprocessors": {
            "distribution": preprocessor_coverage(
                result(gappy_structures, 7, [0, 2, 4])
            ),
            "multi_mlip": preprocessor_coverage(
                result(gappy_structures, warnings=["same warning"] * 41)
            ),
        },
        "multi_mlip": mlip_coverage(gappy_structures, ["mace", "uma"], True, True),
    }
    messages = coverage_warnings(coverage)
    assert len(messages) == len(set(messages)) == 14
    assert set(messages) == {
        "[validity] 2 of 6 structures raised during the validity check",
        "[distribution] 3 of 7 structures dropped",
        "[multi_mlip] 41 preprocessor warnings",
        "[multi_mlip] mace: energy missing for 1 of 4 structures",
        "[multi_mlip] mace: formation_energy missing for 1 of 4 structures",
        "[multi_mlip] mace: e_above_hull missing for 1 of 4 structures",
        "[multi_mlip] mace: graph_embedding missing for 1 of 4 structures",
        "[multi_mlip] mace: model errors for 1 of 4 structures (timeout: 1)",
        "[multi_mlip] mace: energy non-finite for 1 of 4 structures",
        "[multi_mlip] uma: relaxation_rmse missing for 3 of 4 structures",
        "[multi_mlip] uma: relaxed_e_above_hull missing for 3 of 4 structures",
        "[multi_mlip] uma: relaxation failed for 3 of 4 structures "
        "(CUDA out of memory: 2, solver failure: 1)",
        "[multi_mlip] uma: graph_embedding non-finite for 1 of 4 structures",
        "[multi_mlip] e_above_hull from fewer than 2 models for 2 of 4 structures "
        "(0: 1, 1: 1)",
    }


@pytest.mark.parametrize("n_structures", [0, 2])
def test_clean_coverage_has_no_warnings(n_structures):
    structures = [complete_structure(["orb", "mace", "uma"])] * n_structures
    coverage = {
        "validity": preprocessor_coverage(result(structures)),
        "preprocessors": {"multi_mlip": preprocessor_coverage(result(structures))},
        "multi_mlip": mlip_coverage(structures, ["orb", "mace", "uma"], True, True),
    }
    assert coverage_warnings(coverage) == []
    assert coverage["multi_mlip"]["e_above_hull_n_mlips"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": n_structures,
    }


@pytest.fixture
def runner(monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_benchmarks.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("coverage_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "logger", Mock())
    monkeypatch.setattr(module, "cleanup_after_preprocessor", Mock())
    monkeypatch.setattr(module, "clear_memory", Mock())
    monkeypatch.setattr(module, "clear_mlip_models", Mock())
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    return module


@pytest.fixture
def small_coverage():
    structures = [
        structure(energy_mace=np.nan, graph_embedding_mace=[1.0]),
        structure(mlip_error_mace="timeout"),
    ]
    return {
        "validity": preprocessor_coverage(result(structures)),
        "preprocessors": {"multi_mlip": preprocessor_coverage(result(structures))},
        "multi_mlip": mlip_coverage(structures, ["mace"], False, True),
    }


def test_save_results_preserves_scores_and_writes_coverage(
    runner, monkeypatch, tmp_path, small_coverage
):
    monkeypatch.setattr(
        runner, "__file__", str(tmp_path / "scripts" / "run_benchmarks.py")
    )
    validity = {"score": 0.75, "total_structures": 4}
    remaining = {"stability": {"score": 1.25, "per_model": [0.5, 2.0]}}
    expected = deepcopy({"validity": validity, **remaining})
    path = runner.save_results(
        validity, remaining, {}, "test", "test", 4, small_coverage
    )
    assert path.parent == tmp_path / "results_final"
    saved = json.loads(path.read_text())
    assert saved["coverage"] == small_coverage
    assert saved["results"] == expected
    assert {"validity": validity, **remaining} == expected


@pytest.mark.parametrize(
    "name,constructor,flag",
    [
        ("fingerprint", "FingerprintPreprocessor", "fingerprint"),
        ("distribution", "DistributionPreprocessor", "distribution"),
        ("multi_mlip", "MultiMLIPStabilityPreprocessor", "stability"),
    ],
)
def test_runner_rejects_empty_preprocessor_output(
    runner, monkeypatch, name, constructor, flag
):
    processed = result([], 1, [0], ["worker crashed", "later warning"])
    preprocessor = Mock(return_value=processed)
    monkeypatch.setattr(runner, constructor, Mock(return_value=preprocessor))
    config = dict.fromkeys(
        ["fingerprint", "distribution", "stability", "embeddings"], False
    )
    config[flag] = True
    with pytest.raises(RuntimeError, match=f"{name}.*worker crashed"):
        runner.run_remaining_preprocessors([structure()], config, {}, "test")
    preprocessor.assert_called_once()


def test_runner_keeps_partial_output_order_and_results(runner, monkeypatch):
    structures = [structure(), structure(), structure()]
    processed = result([structures[0], structures[2]], 3, [1], ["failed"])
    monkeypatch.setattr(
        runner,
        "DistributionPreprocessor",
        Mock(return_value=Mock(return_value=processed)),
    )
    output, results = runner.run_remaining_preprocessors(
        structures,
        {
            "fingerprint": False,
            "distribution": True,
            "stability": False,
            "embeddings": False,
        },
        {},
        "test",
    )
    assert output is processed.processed_structures
    assert output[0] is structures[0] and output[1] is structures[2]
    assert results == {"distribution": processed}


@pytest.mark.parametrize("no_valid_structures", [False, True])
def test_main_writes_and_logs_coverage(
    runner, monkeypatch, tmp_path, no_valid_structures
):
    structures = [structure(overall_valid=not no_valid_structures)]
    validity_result = result(structures)
    validity_scores = {"score": 0.0 if no_valid_structures else 1.0}
    validity = SimpleNamespace(
        run=Mock(return_value=validity_result),
        generate_benchmark_result=Mock(return_value=validity_scores),
    )
    monkeypatch.setattr(runner, "ValidityPreprocessor", Mock(return_value=validity))
    monkeypatch.setattr(runner, "load_structures_from_wycoff_csv", lambda _: structures)
    monkeypatch.setattr(runner, "load_benchmark_config", lambda _: {})
    monkeypatch.setattr(
        runner, "__file__", str(tmp_path / "scripts" / "run_benchmarks.py")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmarks", "--csv", "unused", "--config", "test", "--name", "test"],
    )
    mlip_result = result(
        structures,
        config=SimpleNamespace(
            mlip_names=["mace"], relax_structures=False, extract_embeddings=True
        ),
    )
    remaining_preprocessors = Mock(
        return_value=(structures, {"multi_mlip": mlip_result})
    )
    monkeypatch.setattr(runner, "run_remaining_preprocessors", remaining_preprocessors)
    remaining_benchmarks = Mock(return_value={"distribution": {"score": 0.5}})
    monkeypatch.setattr(runner, "run_remaining_benchmarks", remaining_benchmarks)
    runner.main()
    saved = json.loads(next((tmp_path / "results_final").glob("*.json")).read_text())
    expected = {"validity": preprocessor_coverage(validity_result)}
    if no_valid_structures:
        remaining_preprocessors.assert_not_called()
        remaining_benchmarks.assert_not_called()
        assert saved["results"] == {"validity": validity_scores}
        runner.logger.info.assert_any_call(
            "Preprocessing coverage: no missing values or failures"
        )
    else:
        expected.update(
            preprocessors={"multi_mlip": preprocessor_coverage(mlip_result)},
            multi_mlip=mlip_coverage(structures, ["mace"], False, True),
        )
        assert saved["results"] == {
            "validity": validity_scores,
            "distribution": {"score": 0.5},
        }
        assert [
            call.args[0] for call in runner.logger.warning.call_args_list
        ] == coverage_warnings(expected)
        remaining_benchmarks.assert_called_once_with(
            structures, ANY, {"fingerprint_method": "structure-matcher"}, False
        )
    assert saved["coverage"] == expected

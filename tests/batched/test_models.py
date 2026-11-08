"""Checkpoint-free CPU checks and opt-in-by-CUDA ASE parity checks."""

from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.stress import full_3x3_to_voigt_6_stress

from lemat_genbench.batched.models import (
    _apply_uma_fixes,
    _generate_graph_batched,
    _mole_forward_split,
    _split_predictions,
    make_batched_model,
)
from lemat_genbench.batched.types import Prediction


def _mole():
    from fairchem.core.models.uma.nn.mole import MOLE, MOLEGlobals

    with torch.random.fork_rng():
        torch.manual_seed(824)
        globals_ = MOLEGlobals(
            expert_mixing_coefficients=torch.randn(4, 3),
            mole_sizes=torch.tensor([5, 1, 17, 9]),
        )
        return MOLE(3, 7, 11, globals_, bias=True)


@pytest.mark.parametrize("start,length", [(0, 32), (3, 25), (5, 18), (23, 9)])
@pytest.mark.parametrize("extra_dims", [(), (3,)])
def test_mole_split_bitwise_forward_and_backward(start, length, extra_dims):
    stock = _mole()
    stock.global_mole_tensors.ac_start_idx = start
    stock.global_mole_tensors.expert_mixing_coefficients.requires_grad_(True)
    fixed = deepcopy(stock)
    fixed.forward = MethodType(_mole_forward_split, fixed)
    rng = torch.Generator().manual_seed(121)
    x = torch.randn(length, *extra_dims, 7, generator=rng, requires_grad=True)
    fixed_x = x.detach().clone().requires_grad_(True)
    stock_out, fixed_out = stock(x), fixed(fixed_x)
    assert torch.equal(stock_out, fixed_out)
    grad = torch.randn(stock_out.shape, generator=rng)
    stock_out.backward(grad)
    fixed_out.backward(grad)
    assert torch.equal(x.grad, fixed_x.grad)
    assert torch.equal(stock.weights.grad, fixed.weights.grad)
    assert torch.equal(stock.bias.grad, fixed.bias.grad)
    assert torch.equal(
        stock.global_mole_tensors.expert_mixing_coefficients.grad,
        fixed.global_mole_tensors.expert_mixing_coefficients.grad,
    )


def _backbone():
    """Only graph configuration is needed; no network or checkpoint is loaded."""
    from fairchem.core.models.uma.escn_md import eSCNMDBackbone

    backbone = eSCNMDBackbone.__new__(eSCNMDBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.otf_graph = True
    backbone.always_use_pbc = False
    backbone.radius_pbc_version = 2
    backbone.cutoff = 6.0
    backbone.max_neighbors = 300
    backbone.enforce_max_neighbors_strictly = False
    backbone.direct_forces = False
    backbone.add_module("mole", _mole())
    backbone.global_mole_tensors = backbone.mole.global_mole_tensors
    return backbone


def _graph_batch():
    from fairchem.core.datasets import data_list_collater
    from fairchem.core.datasets.atomic_data import AtomicData

    structures = [
        Atoms("He", positions=[[0.3, 0.2, 0.1]], cell=[2.1, 2.4, 2.7], pbc=True),
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("Mg", "hcp", a=3.21, c=5.21),
        bulk("Cu", "fcc", a=3.61).repeat((2, 1, 1)),
    ]
    structures[-1].rattle(stdev=0.03, seed=11)
    return data_list_collater(
        [
            AtomicData.from_ase(
                atoms,
                r_edges=False,
                radius=6.0,
                max_neigh=300,
                task_name="omat",
                r_data_keys=["spin", "charge"],
            )
            for atoms in structures
        ],
        otf_graph=True,
    )


def _sorted_edges(graph):
    # fairchem uses edge_index[0] for source and edge_index[1] for target.
    keys = torch.cat(
        (graph["edge_index"].flip(0).T, graph["cell_offsets"].long()), dim=1
    ).tolist()
    order = sorted(range(len(keys)), key=keys.__getitem__)
    return [keys[i] for i in order], order


def _assert_graph_equal(actual, expected):
    actual_keys, actual_order = _sorted_edges(actual)
    expected_keys, expected_order = _sorted_edges(expected)
    assert actual_keys == expected_keys
    assert torch.equal(actual["neighbors"], expected["neighbors"])
    for key in ("edge_distance", "edge_distance_vec", "offset_distances"):
        assert torch.equal(actual[key][actual_order], expected[key][expected_order])


@pytest.mark.parametrize(
    "max_neighbors,strict", [(300, False), (12, False), (12, True)]
)
@pytest.mark.parametrize("always_use_pbc", [False, True])
def test_batched_graph_matches_stock(max_neighbors, strict, always_use_pbc):
    from fairchem.core.graph.compute import generate_graph

    batch = _graph_batch()
    backbone = _backbone()
    backbone.max_neighbors = max_neighbors
    backbone.enforce_max_neighbors_strictly = strict
    backbone.always_use_pbc = always_use_pbc
    expected = generate_graph(batch, 6.0, max_neighbors, strict, 2, batch.pbc)
    stock_backbone = backbone._generate_graph(batch.clone())
    _apply_uma_fixes(backbone)
    actual = backbone._generate_graph(batch.clone())
    _assert_graph_equal(actual, expected)
    _assert_graph_equal(actual, stock_backbone)
    assert set(actual) == set(stock_backbone)
    assert actual["node_offset"] == 0
    for key in ("edge_distance", "edge_distance_vec", "edge_index"):
        assert torch.equal(actual[f"{key}_full"], actual[key])
    assert batch.batch[actual["edge_index"][0]].equal(
        batch.batch[actual["edge_index"][1]]
    )
    edge_systems = batch.batch[actual["edge_index"][1]]
    assert torch.all(edge_systems[1:] >= edge_systems[:-1])
    if max_neighbors == 300:
        assert actual["cell_offsets"].abs().max() >= 2


@pytest.mark.parametrize(
    "case", ["version1", "precomputed", "mixed_pbc", "no_pbc", "strict"]
)
def test_graph_fallback_uses_stock(case):
    backbone = _backbone()
    batch = _graph_batch()
    if case == "version1":
        backbone.radius_pbc_version = 1
    elif case == "strict":
        backbone.enforce_max_neighbors_strictly = True
        backbone.max_neighbors = 12
    elif case == "precomputed":
        graph = backbone._generate_graph(batch)
        batch.edge_index = graph["edge_index"]
        batch.cell_offsets = graph["cell_offsets"]
        batch.nedges = graph["neighbors"]
        backbone.otf_graph = False
    elif case == "mixed_pbc":
        batch.pbc[0, 0] = False
    else:
        batch.pbc[:] = False
    stock = backbone._generate_graph
    _apply_uma_fixes(backbone)
    calls = []

    def record_stock(self, data):
        calls.append(data)
        return stock(data)

    backbone._lemat_stock_generate_graph = MethodType(record_stock, backbone)
    if case == "mixed_pbc":
        with pytest.raises(AssertionError, match="pbc"):
            backbone._generate_graph(batch)
        assert len(calls) == 1
    else:
        expected = stock(batch.clone())
        actual = backbone._generate_graph(batch.clone())
        # All-false PBC passes the stock check and can use the batched graph.
        assert len(calls) == (0 if case == "no_pbc" else 1)
        assert set(actual) == set(expected)
        if case == "precomputed":
            for key in actual:
                if torch.is_tensor(actual[key]):
                    assert torch.equal(actual[key], expected[key])
                else:
                    assert actual[key] == expected[key]
        else:
            _assert_graph_equal(actual, expected)


def test_uma_fixes_are_instance_local_idempotent_and_optional():
    from fairchem.core.models.uma.escn_md import eSCNMDBackbone
    from fairchem.core.models.uma.nn.mole import MOLE

    stock_forward = MOLE.forward
    stock_graph = eSCNMDBackbone._generate_graph
    backbone, untouched = _backbone(), _backbone()
    calculator = SimpleNamespace(
        ase_calc=SimpleNamespace(predictor=SimpleNamespace(model=backbone))
    )
    make_batched_model("uma", calculator, uma_fixes=False)
    assert backbone.mole.forward.__func__ is stock_forward
    assert backbone._generate_graph.__func__ is stock_graph
    make_batched_model("uma", calculator)
    forward, graph = backbone.mole.forward, backbone._generate_graph
    saved_stock = backbone._lemat_stock_generate_graph
    make_batched_model("uma", calculator)
    make_batched_model("uma", calculator, uma_fixes=False)
    assert backbone.mole.forward is forward
    assert backbone._generate_graph is graph
    assert backbone._lemat_stock_generate_graph is saved_stock
    assert forward.__func__ is _mole_forward_split
    assert graph.__func__ is _generate_graph_batched
    assert untouched.mole.forward.__func__ is MOLE.forward is stock_forward
    assert untouched._generate_graph.__func__ is stock_graph
    assert eSCNMDBackbone._generate_graph is stock_graph


@pytest.mark.parametrize("target,attribute", [("backbone", "cutoff"), ("mole", "bias")])
def test_uma_fixes_validate_before_changing_anything(target, attribute):
    from fairchem.core.models.uma.nn.mole import MOLE

    backbone = _backbone()
    delattr(backbone if target == "backbone" else backbone.mole, attribute)
    with pytest.raises(RuntimeError, match="Unsupported fairchem.*missing"):
        _apply_uma_fixes(backbone)
    assert backbone.mole.forward.__func__ is MOLE.forward
    assert not hasattr(backbone, "_lemat_stock_generate_graph")


@pytest.mark.parametrize(
    "name", ["MOLE.forward", "generate_graph", "eSCNMDBackbone._generate_graph"]
)
def test_uma_fixes_reject_changed_source(monkeypatch, name):
    from lemat_genbench.batched.models import _UMA_SOURCE_HASHES

    backbone = _backbone()
    original_forward = backbone.mole.forward.__func__
    original_graph = backbone._generate_graph.__func__
    monkeypatch.setitem(_UMA_SOURCE_HASHES, name, "0" * 64)
    calculator = SimpleNamespace(
        ase_calc=SimpleNamespace(predictor=SimpleNamespace(model=backbone))
    )
    # Stock fairchem remains selectable even if the replacement sources differ.
    make_batched_model("uma", calculator, uma_fixes=False)
    with pytest.raises(RuntimeError) as caught:
        make_batched_model("uma", calculator)
    message = str(caught.value)
    assert name in message
    assert "UMA fixes were validated for fairchem 2.3.0" in message
    assert "uma_fixes=False" in message
    assert "batched.uma_fixes: false" in message
    assert backbone.mole.forward.__func__ is original_forward
    assert backbone._generate_graph.__func__ is original_graph
    assert not hasattr(backbone, "_lemat_stock_generate_graph")


@pytest.mark.parametrize("use_dgl", [False, True])
def test_uma_fixes_require_stock_mole_modules(use_dgl):
    from fairchem.core.models.uma.nn.mole import MOLEDGL

    backbone = _backbone()
    original_graph = backbone._generate_graph.__func__
    if use_dgl:
        # Construction needs no fairchem_cpp; only its forward would need it.
        backbone.mole = MOLEDGL(3, 7, 11, backbone.global_mole_tensors, bias=True)
    else:
        del backbone.mole
    with pytest.raises(RuntimeError, match="no stock MOLE modules"):
        _apply_uma_fixes(backbone)
    assert backbone._generate_graph.__func__ is original_graph
    assert not hasattr(backbone, "_lemat_stock_generate_graph")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("voigt", [False, True])
def test_split_predictions_order_units_voigt_and_dtype(dtype, voigt):
    energy = torch.tensor([1.1, -2.2, 3.3], dtype=dtype, requires_grad=True)
    forces = torch.arange(18, dtype=dtype).reshape(6, 3).requires_grad_(True)
    # Deliberately asymmetric: ASE averages yz/zy, xz/zx and xy/yx.
    stress = torch.arange(27, dtype=dtype).reshape(3, 3, 3).requires_grad_(True)
    expected_stress = np.array([[0, 4, 8, 6, 4, 2]]) + 9 * np.arange(3)[:, None]
    if voigt:
        stress = torch.tensor(expected_stress, dtype=dtype, requires_grad=True)
    converted_stress = stress.detach().numpy() * 2.3 / 1.7**3
    ase_stress = (
        converted_stress
        if voigt
        else np.stack(
            [full_3x3_to_voigt_6_stress(matrix) for matrix in converted_stress]
        )
    )
    result = _split_predictions(
        [2, 1, 3], energy, forces, stress, energy_units_to_eV=2.3, length_units_to_A=1.7
    )
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    expected_stress = expected_stress.astype(numpy_dtype) * 2.3 / 1.7**3
    expected_forces = forces.detach().numpy() * 2.3 / 1.7
    start = 0
    for i, count in enumerate([2, 1, 3]):
        assert type(result[i].energy) is float
        assert result[i].energy == float(energy[i].detach()) * 2.3
        assert result[i].forces.shape == (count, 3)
        assert result[i].stress.shape == (6,)
        assert result[i].forces.dtype == numpy_dtype
        assert result[i].stress.dtype == ase_stress.dtype
        np.testing.assert_array_equal(
            result[i].forces, expected_forces[start : start + count]
        )
        # Units are applied before ASE symmetrises the matrix.
        np.testing.assert_array_equal(result[i].stress, ase_stress[i])
        np.testing.assert_allclose(result[i].stress, expected_stress[i], rtol=2e-7)
        start += count


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown batched MLIP"):
        make_batched_model("not-an-mlip", object())


def _fake_uma_calculator(backbone=None):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    class Predictor:
        datasets = ["omat"]
        inference_mode = SimpleNamespace(external_graph_gen=False)
        dataset_to_tasks = {
            "omat": [
                SimpleNamespace(property=p) for p in ("energy", "forces", "stress")
            ]
        }

        def __init__(self):
            self.model = torch.nn.Module()
            self.model.module = torch.nn.Module()
            self.model.module.backbone = _backbone() if backbone is None else backbone
            self.calls = []

        def seed(self, seed):
            pass

        def predict(self, batch):
            assert torch.is_grad_enabled()
            self.calls.append(batch)
            n = len(batch.natoms)
            if backbone is None:
                graph = self.model.module.backbone._generate_graph(batch)
                self.model.module.backbone.global_mole_tensors.mole_sizes = graph[
                    "neighbors"
                ].cpu()
            else:
                self.model.module.backbone(batch)
            return {
                "energy": batch.natoms.float(),
                "forces": batch.pos.clone(),
                "stress": torch.arange(n * 9, dtype=torch.float32).reshape(n, 3, 3),
            }

    predictor = Predictor()
    return SimpleNamespace(
        ase_calc=FAIRChemCalculator(predict_unit=predictor, task_name="omat")
    )


def test_uma_single_forward_metadata_and_input_immutability():
    calculator = _fake_uma_calculator()
    adapter = make_batched_model("uma", calculator, uma_fixes=False)
    atoms = [bulk("Cu", "fcc", a=3.61), bulk("NaCl", "rocksalt", a=5.64)]
    atoms[0].info.update(charge=2, spin=3, user_metadata="preserve")
    before = deepcopy(atoms)
    with torch.no_grad():
        result = adapter.predict(atoms)
    batches = calculator.ase_calc.predictor.calls
    assert len(batches) == 1
    assert batches[0].charge.tolist() == [2, 0]
    assert batches[0].spin.tolist() == [3, 0]
    assert [item.energy for item in result] == [1.0, 2.0]
    assert [item.forces.shape for item in result] == [(1, 3), (2, 3)]
    for actual, original in zip(atoms, before):
        assert actual == original
        assert actual.info == original.info
    atoms[0].info["charge"] = "invalid"
    with pytest.raises(TypeError, match="charge"):
        adapter.predict(atoms)
    assert len(batches) == 1


@pytest.mark.parametrize("uma_fixes", [False, True])
def test_uma_rejects_edgeless_systems_in_mixed_batch(uma_fixes):
    from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

    # Exercise a real forward and its edge counts, with random weights and
    # synthetic energy/force/stress heads. No model checkpoint is needed.
    with torch.random.fork_rng():
        torch.manual_seed(41)
        backbone = eSCNMDMoeBackbone(
            num_experts=2,
            sphere_channels=4,
            hidden_channels=4,
            edge_channels=4,
            num_distance_basis=8,
            lmax=1,
            mmax=1,
            num_layers=1,
            dataset_list=["omat"],
            otf_graph=True,
            radius_pbc_version=2,
            cutoff=6.0,
            max_neighbors=300,
            direct_forces=False,
            regress_forces=False,
            regress_stress=False,
        ).eval()
    calculator = _fake_uma_calculator(backbone)
    isolated = Atoms("Cu", cell=[7, 7, 7], pbc=True)
    crystal = bulk("Cu", "fcc", a=3.61)
    with pytest.raises(
        ValueError, match="^No edges found in input system"
    ) as ase_error:
        _ase_predictions(calculator.ase_calc, [isolated])
    adapter = make_batched_model("uma", calculator, uma_fixes=uma_fixes)
    with pytest.raises(
        ValueError, match="^No edges found in input system"
    ) as mixed_error:
        adapter.predict([isolated, crystal, isolated.copy()])
    assert "batch positions [0, 2]" in str(mixed_error.value)
    assert backbone.global_mole_tensors.mole_sizes.tolist() == [0, 78, 0]
    with pytest.raises(
        ValueError, match="^No edges found in input system"
    ) as single_error:
        adapter.predict([isolated])
    assert str(single_error.value) == str(ase_error.value)
    # Edge counts are refreshed on subsequent forwards, not retained failures.
    assert len(adapter.predict([crystal])) == 1


@pytest.mark.parametrize("uma_fixes", [False, True])
def test_uma_rejects_changed_edge_count_shape(uma_fixes):
    calculator = _fake_uma_calculator()
    predictor = calculator.ase_calc.predictor
    predict = predictor.predict

    def wrong_edge_counts(batch):
        result = predict(batch)
        predictor.model.module.backbone.global_mole_tensors.mole_sizes = torch.ones(2)
        return result

    predictor.predict = wrong_edge_counts
    adapter = make_batched_model("uma", calculator, uma_fixes=uma_fixes)
    with pytest.raises(RuntimeError, match="expected UMA mole_sizes.*1 per-system"):
        adapter.predict([bulk("Cu", "fcc", a=3.61)])


@pytest.mark.parametrize("uma_fixes", [False, True])
def test_uma_rejects_stale_edge_counts(uma_fixes):
    """A forward pass that no longer writes mole_sizes must not reuse an old value."""
    calculator = _fake_uma_calculator()
    predictor = calculator.ase_calc.predictor
    adapter = make_batched_model("uma", calculator, uma_fixes=uma_fixes)
    crystal = bulk("Cu", "fcc", a=3.61)
    assert len(adapter.predict([crystal])) == 1
    predict = predictor.predict

    def forward_without_edge_counts(batch):
        tensors = predictor.model.module.backbone.global_mole_tensors
        before = tensors.mole_sizes
        result = predict(batch)
        tensors.mole_sizes = before
        return result

    predictor.predict = forward_without_edge_counts
    with pytest.raises(RuntimeError, match="expected UMA mole_sizes"):
        adapter.predict([crystal])


@pytest.mark.parametrize("name", ["orb", "mace", "uma"])
def test_empty_and_nonperiodic_inputs(name):
    # Constructors need settings, but validation must run before featurisation.
    calculator = SimpleNamespace(
        ase_calc=SimpleNamespace(model=None, predictor=None, model_type="MACE")
    )
    adapter = make_batched_model(name, calculator, uma_fixes=False)
    assert adapter.predict([]) == []
    for pbc in (False, [True, True, False]):
        with pytest.raises(ValueError, match="fully periodic"):
            adapter.predict([Atoms("He", cell=[3, 3, 3], pbc=pbc)])


def test_uma_oom_propagates():
    calculator = _fake_uma_calculator()
    error = torch.OutOfMemoryError("test OOM")

    def fail(batch):
        raise error

    calculator.ase_calc.predictor.predict = fail
    adapter = make_batched_model("uma", calculator, uma_fixes=False)
    with pytest.raises(torch.OutOfMemoryError) as caught:
        adapter.predict([bulk("Cu", "fcc", a=3.61)])
    assert caught.value is error


def _parity_structures():
    structures = [
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("GaAs", "zincblende", a=5.65, cubic=True),
        bulk("ZnO", "wurtzite", a=3.25, c=5.21),
        bulk("Fe", "bcc", a=2.87),
        bulk("Mg", "hcp", a=3.21, c=5.21),
        Atoms(
            "SrTiO3",
            scaled_positions=[
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
            ],
            cell=[3.905] * 3,
            pbc=True,
        ),
        bulk("Cu", "fcc", a=3.61),
        bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2)),
    ]
    for i in (0, 1, 2, 5):
        distorted = structures[i].copy()
        strain = np.array([[1.02, 0.015, 0], [0, 0.98, 0.01], [0.005, 0, 1.01]])
        distorted.set_cell(distorted.cell.array @ strain, scale_atoms=True)
        distorted.rattle(stdev=0.04, seed=100 + i)
        structures.append(distorted)
    return structures


def _ase_predictions(calculator, structures):
    from ase.calculators.calculator import all_changes

    predictions = []
    for atoms in structures:
        calculator.calculate(
            atoms.copy(),
            properties=["energy", "forces", "stress"],
            system_changes=all_changes,
        )
        result = calculator.results
        predictions.append(
            Prediction(
                float(np.asarray(result["energy"]).item()),
                result["forces"].copy(),
                result["stress"].copy(),
            )
        )
    return predictions


def _assert_predictions_close(actual, expected, structures):
    assert len(actual) == len(expected) == len(structures)
    for i, (got, want, atoms) in enumerate(zip(actual, expected, structures)):
        assert type(got.energy) is float
        assert got.forces.shape == want.forces.shape == (len(atoms), 3)
        assert got.stress.shape == want.stress.shape == (6,)
        assert got.forces.dtype == want.forces.dtype
        assert got.stress.dtype == want.stress.dtype
        assert abs(got.energy - want.energy) / len(atoms) <= 2e-5, f"energy {i}"
        np.testing.assert_allclose(got.forces, want.forces, rtol=0, atol=5e-4)
        np.testing.assert_allclose(got.stress, want.stress, rtol=0, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ASE parity needs CUDA checkpoints"
)
@pytest.mark.parametrize("name", ["orb", "mace", "uma"])
def test_gpu_ase_parity(name):
    from lemat_genbench.models.registry import get_calculator

    config = {
        "orb": {"model_type": "orb_v3_conservative_inf_omat"},
        "mace": {"model_type": "mp"},
        "uma": {"task": "omat"},
    }
    calculator = get_calculator(name, device="cuda", **config[name])
    # ORB's loader changes process-wide flags, so disable TF32 after loading.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    structures = _parity_structures()
    before = deepcopy(structures)
    ase_stock = _ase_predictions(calculator.ase_calc, structures)
    stock = make_batched_model(name, calculator, uma_fixes=False)

    def batches(adapter, size):
        return [
            prediction
            for start in range(0, len(structures), size)
            for prediction in adapter.predict(structures[start : start + size])
        ]

    stock_results = {}
    for size in (1, 5, len(structures)):
        stock_results[size] = batches(stock, size)
        _assert_predictions_close(stock_results[size], ase_stock, structures)
    if name == "uma":
        # Collect every stock result before changing this shared model instance.
        fixed = make_batched_model(name, calculator, uma_fixes=True)
        for size, stock_result in stock_results.items():
            result = batches(fixed, size)
            _assert_predictions_close(result, ase_stock, structures)
            _assert_predictions_close(result, stock_result, structures)
        _assert_predictions_close(
            _ase_predictions(calculator.ase_calc, structures), ase_stock, structures
        )
    for actual, original in zip(structures, before):
        assert actual == original
        assert actual.info == original.info

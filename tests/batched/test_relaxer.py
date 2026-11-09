"""CPU tests for the batched relaxer, ``lemat_genbench.batched.relaxer.relax_stream``.

The reference is LeMat's own protocol (``BaseMLIPCalculator.relax_structure``): ASE
``BFGSLineSearch(FrechetCellFilter(atoms))`` run to fmax for at most ``steps`` steps, with the endpoint energy read
afterwards. The reference calculator below evaluates every request through the same ``EMTModel.predict`` that the
batched relaxer calls, and that function is stateless, so the batched relaxer must reproduce the reference bit for
bit: every endpoint, energy, step count and number of model calls.
"""

import threading
from collections import Counter

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.optimize import BFGSLineSearch

from lemat_genbench.batched.relaxer import relax_stream
from lemat_genbench.batched.types import Prediction

pytestmark = pytest.mark.filterwarnings("ignore:logm result may be inaccurate")

METALS = ["Al", "Cu", "Ni", "Pd", "Ag", "Pt", "Au"]
TIMEOUT = 300  # seconds; a relax_stream call that takes longer is treated as a deadlock

# Structures with unique atomic-number sequences, used to inject faults into one structure only.
RAISE_KEY = (29, 79, 29, 29)  # Cu3Au, Au on the second site
NAN_KEY = (28, 28, 46, 28)  # Ni3Pd, Pd on the third site
BAD_SHAPE_KEY = (13, 13, 13, 78)  # Al3Pt, Pt on the fourth site


def _key(atoms):
    return tuple(int(z) for z in atoms.numbers)


def _perturb(atoms, rng):
    eps = rng.normal(0.0, 0.02, (3, 3))
    atoms.set_cell(
        atoms.cell.array @ (np.eye(3) + 0.5 * (eps + eps.T)), scale_atoms=True
    )
    atoms.positions += rng.normal(0.0, 0.05, atoms.positions.shape)
    return atoms


def make_structures(n, seed=0):
    """Rattled, strained fcc metals and binary alloys with 4 or 8 atoms."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        atoms = bulk(METALS[i % len(METALS)], "fcc", cubic=True)
        if i % 3 == 1:
            atoms = atoms.repeat((1, 1, 2))
        if i % 3 == 2:
            atoms[0].symbol = METALS[(i + 3) % len(METALS)]
        out.append(_perturb(atoms, rng))
    return out


def make_fault_structure(key, seed):
    atoms = Atoms(numbers=list(key), cell=bulk("Cu", "fcc", cubic=True).cell, pbc=True)
    atoms.set_scaled_positions([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    vol_per_atom = {13: 16.6, 28: 10.9, 29: 11.8}[key[0]]
    atoms.set_cell(
        atoms.cell.array * (4 * vol_per_atom / atoms.get_volume()) ** (1 / 3),
        scale_atoms=True,
    )
    return _perturb(atoms, np.random.default_rng(seed))


def emt_predict(atoms):
    """ASE EMT evaluated from scratch on a clean copy of the geometry."""
    clean = Atoms(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell.array,
        pbc=atoms.pbc,
    )
    calc = EMT()
    calc.calculate(clean, ["energy", "forces", "stress"], all_changes)
    return Prediction(
        energy=float(calc.results["energy"]),
        forces=np.array(calc.results["forces"]),
        stress=np.array(calc.results["stress"]),
    )


class EMTModel:
    """A ``BatchedModel`` over ``emt_predict`` with optional faults.

    raise_after: {key: n} raises ValueError for the whole call once that structure has had n predictions.
    nan_after:   {key: n} returns NaN energy and forces for that structure from its (n+1)-th prediction on.
    bad_shape:   set of keys whose forces lack the last atom (ASE then raises inside the optimizer).
    oom_atoms:   raises torch.OutOfMemoryError when a call holds more atoms than this.
    A call that raises changes no counter, so retries of the same requests see the same state.
    """

    def __init__(self, raise_after=None, nan_after=None, bad_shape=(), oom_atoms=None):
        self.raise_after = raise_after or {}
        self.nan_after = nan_after or {}
        self.bad_shape = set(bad_shape)
        self.oom_atoms = oom_atoms
        self.seen = Counter()  # successful predictions per structure key
        self.calls = []  # (n_structures, n_atoms) of every call, including calls that raised
        self._lock = threading.Lock()

    def predict(self, atoms):
        atoms = list(atoms)
        with self._lock:
            n_atoms = sum(len(a) for a in atoms)
            self.calls.append((len(atoms), n_atoms))
            if self.oom_atoms is not None and n_atoms > self.oom_atoms:
                raise torch.OutOfMemoryError(f"fake OOM: {n_atoms} atoms in one call")
            keys = [_key(a) for a in atoms]
            for k in keys:
                if k in self.raise_after and self.seen[k] >= self.raise_after[k]:
                    raise ValueError("fake model failure")
            out = []
            for a, k in zip(atoms, keys):
                p = emt_predict(a)
                if k in self.bad_shape:
                    p = Prediction(p.energy, p.forces[:-1], p.stress)
                if k in self.nan_after and self.seen[k] >= self.nan_after[k]:
                    p = Prediction(
                        float("nan"), np.full_like(p.forces, np.nan), p.stress
                    )
                self.seen[k] += 1
                out.append(p)
            return out


class ModelCalculator(Calculator):
    """Serial reference calculator: one ``predict`` call per request, as LeMat's ASE calculators do."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        (p,) = self.model.predict([self.atoms.copy()])
        self.n_calls += 1
        self.results = {
            "energy": p.energy,
            "free_energy": p.energy,
            "forces": p.forces,
            "stress": p.stress,
        }


def reference_relax(atoms, model, fmax=0.02, steps=50):
    """LeMat's relax_structure (models/base.py), with errors caught as in multi_mlip_preprocess."""
    atoms = atoms.copy()
    calc = ModelCalculator(model)
    atoms.calc = calc
    try:
        dyn = BFGSLineSearch(FrechetCellFilter(atoms), logfile=None)
        converged = dyn.run(fmax=fmax, steps=steps)
        energy = atoms.get_potential_energy()
    except Exception as e:
        return dict(error=str(e), calls=calc.n_calls)
    return dict(
        error=None,
        positions=atoms.positions.copy(),
        cell=atoms.cell.array.copy(),
        energy=energy,
        steps=dyn.get_number_of_steps(),
        converged=bool(converged),
        calls=calc.n_calls,
    )


def run_relaxer(jobs, model, **kwargs):
    """Run relax_stream to completion in a daemon thread, failing the test on a deadlock."""
    box = {}

    def target():
        try:
            box["out"] = list(relax_stream(jobs, model, **kwargs))
        except BaseException as e:  # noqa: BLE001 - re-raised in the test thread
            box["err"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    if thread.is_alive():
        pytest.fail(f"relax_stream did not finish within {TIMEOUT} s (deadlock?)")
    if "err" in box:
        raise box["err"]
    return box["out"]


def by_index(outcomes):
    indices = [o.index for o in outcomes]
    assert len(indices) == len(set(indices)), "an index was reported more than once"
    return {o.index: o for o in outcomes}


def assert_matches_reference(outcome, ref):
    assert outcome.error is None, outcome.error
    assert ref["error"] is None
    assert outcome.atoms is not None
    assert np.array_equal(outcome.atoms.positions, ref["positions"])
    assert np.array_equal(outcome.atoms.cell.array, ref["cell"])
    assert outcome.energy == ref["energy"]
    assert outcome.steps == ref["steps"]
    assert outcome.converged == ref["converged"]
    assert outcome.model_calls == ref["calls"]
    assert outcome.cache_hits == 0


@pytest.fixture(scope="module")
def structures():
    return make_structures(16)


@pytest.fixture(scope="module")
def references(structures):
    refs = [reference_relax(a, EMTModel()) for a in structures]
    assert all(r["error"] is None and r["converged"] for r in refs)
    # line searches with several calls per step
    assert sum(r["calls"] for r in refs) > sum(r["steps"] for r in refs) + len(refs)
    return refs


@pytest.mark.parametrize("max_in_flight", [1, 4, 64])
@pytest.mark.parametrize("order", ["forward", "reversed"])
def test_bit_identical_to_serial_protocol(structures, references, max_in_flight, order):
    # arbitrary, non-contiguous job indices
    indices = [10 * i + 3 for i in range(len(structures))]
    jobs = list(zip(indices, structures))
    if order == "reversed":
        jobs = jobs[::-1]
    model = EMTModel()
    outcomes = by_index(
        run_relaxer(jobs, model, fmax=0.02, steps=50, max_in_flight=max_in_flight)
    )
    assert sorted(outcomes) == sorted(indices)
    for idx, ref in zip(indices, references):
        assert_matches_reference(outcomes[idx], ref)
        assert _key(outcomes[idx].atoms) == _key(structures[indices.index(idx)])
    assert max(n for n, _ in model.calls) <= max_in_flight
    if max_in_flight > 1:
        # requests from different structures share calls; refill keeps the calls full
        assert len(model.calls) < sum(r["calls"] for r in references)
        assert max(n for n, _ in model.calls) == min(max_in_flight, len(structures))


def test_defaults_are_lemat_protocol(structures, references):
    outcomes = by_index(run_relaxer(list(enumerate(structures)), EMTModel()))
    for i, ref in enumerate(references):
        assert_matches_reference(outcomes[i], ref)


@pytest.mark.parametrize("steps", [0, 1, 8])
def test_step_cap(structures, steps):
    """Capped relaxations keep their unconverged endpoints, as in LeMat."""
    refs = [reference_relax(a, EMTModel(), steps=steps) for a in structures]
    if steps == 8:
        assert any(r["converged"] for r in refs) and not all(
            r["converged"] for r in refs
        )
    outcomes = by_index(
        run_relaxer(
            list(enumerate(structures)), EMTModel(), steps=steps, max_in_flight=4
        )
    )
    for i, ref in enumerate(refs):
        assert_matches_reference(outcomes[i], ref)
        assert outcomes[i].steps <= steps


def test_inputs_not_modified_and_outputs_independent(structures):
    before = [(a.positions.copy(), a.cell.array.copy()) for a in structures]
    outcomes = run_relaxer(list(enumerate(structures)), EMTModel(), max_in_flight=8)
    for a, (pos, cell) in zip(structures, before):
        assert np.array_equal(a.positions, pos)
        assert np.array_equal(a.cell.array, cell)
    for o in outcomes:
        assert o.atoms is not structures[o.index]


def test_empty_input():
    assert run_relaxer([], EMTModel()) == []


def test_jobs_are_pulled_lazily(structures):
    pulled = []

    def jobs():
        for i, a in enumerate(structures):
            pulled.append(i)
            yield i, a

    first_seen = []
    out = []

    def target():
        for o in relax_stream(jobs(), EMTModel(), max_in_flight=4):
            if not first_seen:
                first_seen.append(len(pulled))
            out.append(o)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "relax_stream did not finish (deadlock?)"
    assert len(out) == len(structures)
    assert first_seen[0] <= 8, (
        f"{first_seen[0]} jobs were pulled before the first outcome"
    )


def test_failures_are_isolated(structures, references):
    """A model error, a NaN prediction and an error inside ASE each end only their own relaxation."""
    faults = [
        make_fault_structure(RAISE_KEY, 1),
        make_fault_structure(NAN_KEY, 2),
        make_fault_structure(BAD_SHAPE_KEY, 3),
    ]

    def fault_model():
        return EMTModel(
            raise_after={RAISE_KEY: 3},
            nan_after={NAN_KEY: 3},
            bad_shape={BAD_SHAPE_KEY},
        )

    raise_ref = reference_relax(faults[0], fault_model())
    shape_ref = reference_relax(faults[2], fault_model())
    assert raise_ref["error"] == "fake model failure" and raise_ref["calls"] == 3
    assert shape_ref["error"] is not None and shape_ref["calls"] == 1
    # the NaN arrives mid-relaxation
    assert reference_relax(faults[1], EMTModel())["calls"] > 4

    n = len(structures)
    jobs = list(enumerate(structures[:5])) + [
        (n, faults[0]),
        (n + 1, faults[1]),
        (n + 2, faults[2]),
    ]
    jobs += [(i, a) for i, a in enumerate(structures) if i >= 5]
    outcomes = by_index(run_relaxer(jobs, fault_model(), max_in_flight=6))
    assert len(outcomes) == n + 3

    for i, ref in enumerate(references):
        assert_matches_reference(outcomes[i], ref)

    for idx, ref in [(n, raise_ref), (n + 2, shape_ref)]:
        o = outcomes[idx]
        assert o.error == ref["error"]
        assert o.model_calls == ref["calls"]
    nan = outcomes[n + 1]
    assert nan.error is not None and nan.model_calls == 4
    for idx in (n, n + 1, n + 2):
        o = outcomes[idx]
        assert o.atoms is None and o.energy is None and not o.converged


def test_out_of_memory_splits_the_call(structures, references):
    # 72 atoms, more than the fake memory holds
    big = bulk("Cu", "fcc", cubic=True).repeat((3, 3, 2))
    big = _perturb(big, np.random.default_rng(7))
    n = len(structures)
    jobs = list(enumerate(structures)) + [(n, big)]
    model = EMTModel(oom_atoms=40)
    outcomes = by_index(run_relaxer(jobs, model, max_in_flight=16))
    for i, ref in enumerate(references):
        assert_matches_reference(outcomes[i], ref)
    assert "fake OOM" in outcomes[n].error
    assert outcomes[n].atoms is None and outcomes[n].model_calls == 0
    assert any(n_atoms > 40 for _, n_atoms in model.calls)  # the split was needed


def test_atom_budget_limits_calls(structures, references):
    big = bulk("Cu", "fcc", cubic=True).repeat((2, 2, 2))  # 32 atoms
    big = _perturb(big, np.random.default_rng(8))
    big_ref = reference_relax(big, EMTModel())
    n = len(structures)
    jobs = (
        list(enumerate(structures[:8]))
        + [(n, big)]
        + [(i, a) for i, a in enumerate(structures) if i >= 8]
    )
    model = EMTModel()
    outcomes = by_index(run_relaxer(jobs, model, max_in_flight=16, max_atoms=20))
    for i, ref in enumerate(references):
        assert_matches_reference(outcomes[i], ref)
    assert_matches_reference(outcomes[n], big_ref)
    # a structure over the budget runs alone
    for n_structures, n_atoms in model.calls:
        assert n_atoms <= 20 or n_structures == 1


def test_request_cap_is_a_relaxation_error(structures, references):
    cap = int(np.median([r["calls"] for r in references]))
    assert any(r["calls"] <= cap for r in references) and any(
        r["calls"] > cap for r in references
    )
    outcomes = by_index(
        run_relaxer(
            list(enumerate(structures)), EMTModel(), max_in_flight=8, max_requests=cap
        )
    )
    for i, ref in enumerate(references):
        o = outcomes[i]
        if ref["calls"] <= cap:
            assert_matches_reference(o, ref)
        else:
            assert o.error is not None and "max_requests" in o.error
            assert o.atoms is None and o.model_calls == cap


def test_geometry_tolerance_cache(structures, references):
    """With a 1e-10 Å tolerance, near-identical requests reuse the last result and the endpoints barely move."""
    model = EMTModel()
    outcomes = by_index(
        run_relaxer(
            list(enumerate(structures)), model, max_in_flight=8, geometry_tol=1e-10
        )
    )
    assert sum(o.cache_hits for o in outcomes.values()) > 0
    assert sum(o.model_calls for o in outcomes.values()) < sum(
        r["calls"] for r in references
    )
    for i, ref in enumerate(references):
        o = outcomes[i]
        assert o.error is None
        assert o.converged == ref["converged"]
        assert np.max(np.abs(o.atoms.positions - ref["positions"])) < 1e-8
        assert np.max(np.abs(o.atoms.cell.array - ref["cell"])) < 1e-8
        assert abs(o.energy - ref["energy"]) < 1e-8

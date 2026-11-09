"""End-to-end CPU test: the batched Multi-MLIP stage gives the per-structure stage's properties.

Fake ORB, MACE and UMA models built on ASE's EMT run through both paths: LeMat's calculator
interface for ``per_structure`` and the ``BatchedModel`` interface for ``batched``. The fakes copy
the habits of the real wrappers that matter here: MACE's ``calculate_energy_forces`` returns None
instead of raising (models/mace/calculator.py), UMA's clears ``atoms.info`` before a single point
(models/uma/calculator.py) and fairchem's calculator writes charge and spin defaults into the
atoms it evaluates, and loading a model switches TF32 on, as ORB's loader does. The batched
workers are forked in this test so that the patches reach them.

GPU memory is emulated: ``stage._device_memory`` reports a configurable free and total memory and
``stage._limit_memory`` only records the share a worker gets. A worker held to a share raises the
memory errors seen on real cards (PyTorch out-of-memory, a cuSOLVER handle failure, a cuBLAS
allocation failure) for marked structures and for calls above an atom limit; a worker without a
share does not, unless the fault is marked as permanent.
"""

import multiprocessing
import os
import signal
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from lemat_genbench.batched import models as batched_models
from lemat_genbench.batched import stage
from lemat_genbench.batched.types import Prediction
from lemat_genbench.models import registry
from lemat_genbench.models.base import (
    BaseMLIPCalculator,
    CalculationResult,
    EmbeddingResult,
    get_energy_above_hull_from_total_energy,
    get_formation_energy_per_atom_from_total_energy,
)
from lemat_genbench.preprocess import multi_mlip_preprocess, reference_energies
from lemat_genbench.preprocess.multi_mlip_preprocess import (
    MultiMLIPStabilityPreprocessor,
)
from lemat_genbench.preprocess.reference_energies import one_hot_encode_composition

pytestmark = pytest.mark.filterwarnings("ignore:logm result may be inaccurate")

MLIPS = ["orb", "mace", "uma"]
SCALE = {"orb": 1.0, "mace": 0.95, "uma": 1.05}
HULL_TYPE = {"orb": "orb_conserv_inf", "mace": "mace_mp", "uma": "uma"}
HULL_OFFSET = {"orb_conserv_inf": 0.0, "mace_mp": 0.03, "uma": -0.02}
# The model raises for every geometry of a structure that contains this element.
SINGLE_POINT_FAULT = {"mace": "Au", "uma": "Pt"}
EMBEDDING_FAULT = {"orb": "Ag"}
HULL_GAP = {"uma": "Pd"}  # missing from this model's hull table
ELEMENTS = ["Al", "Ni", "Cu", "Pd", "Ag", "Pt", "Au"]
STEPS = 6
# Relaxed formation energy and E_hull: batched reuses the endpoint evaluation, per_structure
# evaluates the endpoint again after a pymatgen round trip.
TOL = 1e-9
TOLERANT_PREFIXES = ("relaxed_formation_energy", "relaxed_e_above_hull")
BATCHED = {
    "workers_per_model": 2,
    "structures_per_worker": 3,
    "max_atoms_per_worker": 10,
    "uma_fixes": False,
}
RELAX_FAULT = 3  # index of the structure whose relaxations fail after one evaluation
CRASH_MARKED = 5  # index of the structure that crashes ORB workers in the crash tests
# kind "exit": the worker process dies at the n-th sighting of the marked structure;
# "cuda": from then on every model call in that process raises a sticky CUDA error;
# "embedding": the worker dies while embedding the marked structure.
CRASH = {"flag": None, "persistent": False, "kind": "exit", "sighting": 2}
CUDA_ERROR = "CUDA error: an illegal memory access was encountered"
CUDA_BROKEN = {"on": False}  # per process, like a CUDA context after such an error
BATCH_LOG = {"path": None}
TIME_LIMIT = 300  # seconds per test
GIB = 2**30
# device: (free, total) bytes reported to the first worker, or None for no GPU.
# faults: atom count of a structure -> memory error raised by a worker held to a share for any
#   call containing it ("oom", "cusolver", "cublas"), or by every worker ("always").
# atom_limit: a worker held to a share raises out-of-memory for calls above this many atoms.
# load_failures: per model, the first n workers run out of memory while loading the model.
MEMORY_DEFAULTS = {
    "device": None,
    "faults": {},
    "atom_limit": None,
    "load_failures": 0,
    "dir": None,
}
MEMORY = dict(MEMORY_DEFAULTS)
MEMORY_ERRORS = {
    "oom": lambda: torch.OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity "
        "of 47.38 GiB of which 4.00 MiB is free."
    ),
    "cusolver": lambda: RuntimeError(
        "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling "
        "`cusolverDnCreate(handle)`"
    ),
    "cublas": lambda: RuntimeError(
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"
    ),
}


def in_worker():
    return multiprocessing.parent_process() is not None


def memory_fault(mlip, counts):
    """Raise the memory error a worker would hit for a call on structures of these sizes."""
    if not in_worker():
        return
    capped = stage.MEMORY_FRACTION is not None
    event = "ok"
    for count in counts:
        kind = MEMORY["faults"].get(count)
        if kind == "always" or (kind is not None and capped):
            event = kind
            break
    else:
        limit = MEMORY["atom_limit"]
        if capped and limit is not None and sum(counts) > limit:
            event = "atom_limit"
    if MEMORY["dir"] is not None:
        with open(MEMORY["dir"] / "calls.txt", "a") as log:
            sizes = " ".join(map(str, counts))
            log.write(f"{os.getpid()} {mlip} {int(capped)} {sizes} {event}\n")
    if event in ("atom_limit", "always"):
        raise MEMORY_ERRORS["oom"]()
    if event != "ok":
        raise MEMORY_ERRORS[event]()


def memory_calls():
    path = MEMORY["dir"] / "calls.txt"
    calls = []
    for line in path.read_text().splitlines() if path.exists() else []:
        pid, mlip, capped, *counts, event = line.split()
        calls.append((int(pid), mlip, capped == "1", list(map(int, counts)), event))
    return calls


def recorded_limits():
    path = MEMORY["dir"] / "limits.txt"
    limits = []
    for line in path.read_text().splitlines() if path.exists() else []:
        pid, fraction = line.split()
        limits.append((int(pid), None if fraction == "None" else float(fraction)))
    return limits


def evaluate(name, atoms):
    """Energy, forces and stress of one fake MLIP; raises the injected faults."""
    if CUDA_BROKEN["on"]:
        raise RuntimeError(CUDA_ERROR)
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("matmul precision was not set after loading the model")
    element = SINGLE_POINT_FAULT.get(name)
    if element in atoms.get_chemical_symbols():
        raise RuntimeError(f"fake {name} cannot evaluate {element}")
    start = atoms.info.get("relax_fault")
    if start is not None and not np.array_equal(atoms.positions, np.asarray(start)):
        raise RuntimeError("fake relaxation fault")
    emt = atoms.copy()
    emt.calc = EMT()
    scale = SCALE[name]
    # UMA reads charge from atoms.info; the shift shows whether info reached the model.
    energy = scale * emt.get_potential_energy() + 0.01 * atoms.info.get("charge", 0)
    return float(energy), scale * emt.get_forces(), scale * emt.get_stress()


class FakeASECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, mlip):
        super().__init__()
        self.mlip = mlip  # ASE calculators have a read-only `name`

    def _validate_charge_and_spin(self, atoms):
        # fairchem's FAIRChemCalculator writes these defaults into the caller's atoms
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 0)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if self.mlip == "uma" and atoms is not None:
            self._validate_charge_and_spin(atoms)
        super().calculate(atoms, properties, system_changes)
        energy, forces, stress = evaluate(self.mlip, self.atoms)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "stress": stress,
        }


class FakeCalculator(BaseMLIPCalculator):
    """LeMat's calculator interface around a fake model, with the real wrappers' habits."""

    def __init__(self, mlip, model_type=None, task=None, **kwargs):
        self.mlip = mlip
        self.model_type = f"fake-{mlip}"
        super().__init__(**kwargs)

    def _setup_model(self, **kwargs):
        if in_worker():
            for i in range(MEMORY["load_failures"]):
                token = MEMORY["dir"] / f"load_{self.mlip}_{i}"
                try:
                    os.close(os.open(token, os.O_CREAT | os.O_EXCL))
                except FileExistsError:
                    continue
                raise MEMORY_ERRORS["oom"]()
        torch.set_float32_matmul_precision("high")  # as ORB's loader does
        self.ase_calc = FakeASECalculator(self.mlip)

    def _get_ase_calculator(self):
        return self.ase_calc

    def calculate_energy_forces(self, structure):
        atoms = structure.to_ase_atoms()
        if self.mlip == "uma":
            atoms.info = {}
        atoms.calc = self.ase_calc
        try:
            memory_fault(self.mlip, [len(atoms)])
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        except Exception:
            if self.mlip != "mace":
                raise
            return CalculationResult(energy=None, forces=None, metadata={})
        return CalculationResult(energy=energy, forces=forces, metadata={})

    def calculate_formation_energy(self, structure):
        energy = self.calculate_energy_forces(structure).energy
        return get_formation_energy_per_atom_from_total_energy(
            energy, structure.composition
        )

    def calculate_energy_above_hull(self, structure):
        energy = self.calculate_energy_forces(structure).energy
        return get_energy_above_hull_from_total_energy(
            energy, structure.composition, hull_type=self.hull_type
        )

    def extract_embeddings(self, structure):
        if (
            self.mlip == "orb"
            and CRASH["flag"] is not None
            and CRASH["kind"] == "embedding"
            and structure.properties.get("crash")
            and (CRASH["persistent"] or not CRASH["flag"].exists())
        ):
            CRASH["flag"].touch()
            os._exit(3)
        if CUDA_BROKEN["on"]:
            raise RuntimeError(CUDA_ERROR)
        memory_fault(self.mlip, [len(structure)])
        element = EMBEDDING_FAULT.get(self.mlip)
        if element in {el.symbol for el in structure.composition.elements}:
            raise RuntimeError(f"fake {self.mlip} cannot embed {element}")
        node = np.column_stack([structure.atomic_numbers, structure.cart_coords])
        node = (SCALE[self.mlip] * node).astype(np.float32)
        return EmbeddingResult(node, node.mean(axis=0), {})


class FakeBatchedModel:
    """The same fake model behind the BatchedModel interface; one call fails as a whole."""

    def __init__(self, name, calculator, *, uma_fixes=True):
        assert isinstance(calculator, FakeCalculator) and calculator.mlip == name
        if name == "uma":
            assert uma_fixes is BATCHED["uma_fixes"], "uma_fixes was not passed through"
        self.mlip = name
        self.crash_sightings = 0

    def predict(self, atoms):
        if BATCH_LOG["path"] is not None:
            with open(BATCH_LOG["path"], "a") as log:
                log.write(f"{self.mlip} {len(atoms)} {sum(map(len, atoms))}\n")
        if (
            self.mlip == "orb"
            and CRASH["flag"] is not None
            and CRASH["kind"] in ("exit", "cuda")
        ):
            self.crash_sightings += sum(bool(a.info.get("crash")) for a in atoms)
            if self.crash_sightings >= CRASH["sighting"] and (
                CRASH["persistent"] or not CRASH["flag"].exists()
            ):
                CRASH["flag"].touch()
                if CRASH["kind"] == "exit":
                    os._exit(3)
                CUDA_BROKEN["on"] = True
        memory_fault(self.mlip, [len(a) for a in atoms])
        return [Prediction(*evaluate(self.mlip, a)) for a in atoms]


def hull_tables():
    """Synthetic hull references for the EMT elements, one table per hull type."""
    tables = {}
    for name, hull_type in HULL_TYPE.items():
        offset = HULL_OFFSET[hull_type]
        entries = [({el: 1}, -0.1 * (i + 1) + offset) for i, el in enumerate(ELEMENTS)]
        entries += [
            ({"Cu": 1, "Au": 1}, 2 * (-0.55 + offset)),
            ({"Ni": 1, "Al": 1}, 2 * (-0.45 + offset)),
            ({"Ag": 1, "Pd": 1}, 2 * (-0.48 + offset)),
        ]
        entries = [(c, e) for c, e in entries if HULL_GAP.get(name) not in c]
        df = pd.DataFrame(
            {
                "species_at_sites": [
                    [el for el, n in c.items() for _ in range(n)] for c, _ in entries
                ],
                "energy": np.array([e for _, e in entries], dtype=np.float32),
            }
        )
        matrix = np.stack([one_hot_encode_composition(list(c)) for c, _ in entries])
        tables[hull_type] = (df, matrix)
    return tables


def make_structures():
    """Strained, rattled EMT cells of 1-12 atoms, with properties as earlier stages leave them."""
    cells = [
        bulk("Cu", "fcc", a=3.75),
        bulk("Cu", "fcc", a=3.55, cubic=True),
        bulk("Al", "fcc", a=4.2).repeat((2, 1, 1)),
        bulk("Ni", "fcc", a=3.45, cubic=True),
        bulk("Pd", "fcc", a=4.0).repeat((1, 1, 2)),
        bulk("Cu", "fcc", a=3.7, cubic=True),
        bulk("Ag", "fcc", a=4.0),
        bulk("Pt", "fcc", a=3.85),
        bulk("Au", "fcc", a=4.2),
        bulk("Cu", "fcc", a=3.65, cubic=True).repeat((2, 1, 1)),
    ]
    cuau = bulk("Cu", "fcc", a=3.8, cubic=True)
    cuau.symbols[[1, 2]] = "Au"
    nial = bulk("Ni", "fcc", a=3.6, cubic=True)
    nial.symbols[[0, 3]] = "Al"
    agpd = bulk("Ag", "fcc", a=4.0, cubic=True)
    agpd.symbols[[0, 1]] = "Pd"
    ptcu = bulk("Pt", "fcc", a=3.8, cubic=True)
    ptcu.symbols[[2, 3]] = "Cu"
    cells += [cuau, nial, agpd, ptcu]
    cells.append(bulk("Al", "fcc", a=4.1, cubic=True).repeat((3, 1, 1)))  # over budget

    rng = np.random.default_rng(7)
    structures = []
    for i, atoms in enumerate(cells):
        strain = np.eye(3) + rng.normal(0, 0.02, (3, 3))
        atoms.set_cell(atoms.cell @ strain, scale_atoms=True)
        atoms.rattle(0.05, seed=i)
        structure = AseAtomsAdaptor.get_structure(atoms)
        structure.properties = {"source": "fake", "id": i}
        structures.append(structure)
    structures[RELAX_FAULT].properties["relax_fault"] = structures[
        RELAX_FAULT
    ].cart_coords.tolist()
    structures[RELAX_FAULT].properties["charge"] = 1
    structures[CRASH_MARKED].properties["crash"] = True
    return structures


@pytest.fixture
def fakes(monkeypatch):
    tables = hull_tables()
    monkeypatch.setattr(
        registry,
        "get_calculator",
        lambda name, **config: FakeCalculator(name, **config),
    )
    monkeypatch.setattr(batched_models, "make_batched_model", FakeBatchedModel)
    monkeypatch.setattr(
        reference_energies,
        "_retrieve_df",
        lambda hull_type="dft", threshold=0.001: tables[hull_type][0],
    )
    monkeypatch.setattr(
        reference_energies,
        "_retrieve_matrix",
        lambda hull_type="dft", threshold=0.001: tables[hull_type][1],
    )
    monkeypatch.setattr(stage, "START_METHOD", "fork")
    monkeypatch.setattr(stage, "_device_memory", lambda device: MEMORY["device"])

    def limit_memory(device, fraction):
        if MEMORY["dir"] is not None:
            with open(MEMORY["dir"] / "limits.txt", "a") as log:
                log.write(f"{os.getpid()} {fraction}\n")

    monkeypatch.setattr(stage, "_limit_memory", limit_memory)
    MEMORY.update(MEMORY_DEFAULTS, faults={})
    multi_mlip_preprocess._PROCESS_MODEL_CACHE.clear()
    precision = torch.get_float32_matmul_precision()

    def timeout(signum, frame):
        raise TimeoutError(f"test exceeded {TIME_LIMIT} s")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(TIME_LIMIT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        for child in multiprocessing.active_children():
            child.kill()
            child.join()
        CRASH.update(flag=None, persistent=False, kind="exit", sighting=2)
        BATCH_LOG["path"] = None
        CUDA_BROKEN["on"] = False
        MEMORY.update(MEMORY_DEFAULTS, faults={})
        multi_mlip_preprocess._PROCESS_MODEL_CACHE.clear()
        torch.set_float32_matmul_precision(precision)


def run(structures, mode, stability=True, embeddings=True, **batched):
    preprocessor = MultiMLIPStabilityPreprocessor(
        mlip_names=MLIPS,
        mlip_configs={
            name: {"device": "cpu", "hull_type": HULL_TYPE[name]} for name in MLIPS
        },
        relax_structures=stability,
        relaxation_config={"fmax": 0.02, "steps": STEPS},
        calculate_formation_energy=stability,
        calculate_energy_above_hull=stability,
        extract_embeddings=embeddings,
        timeout=300,
        n_jobs=1,
        execution_mode=mode,
        matmul_precision="highest",
        batched={**BATCHED, **batched},
    )
    if mode == "batched":
        # Workers must load their own models and set the precision themselves.
        multi_mlip_preprocess._PROCESS_MODEL_CACHE.clear()
        torch.set_float32_matmul_precision("high")
    return preprocessor(structures)


def assert_same(a, b, key):
    assert type(a) is type(b), (key, type(a), type(b))
    if isinstance(a, Structure):
        assert np.array_equal(a.lattice.matrix, b.lattice.matrix), key
        assert [str(s) for s in a.species] == [str(s) for s in b.species], key
        assert np.array_equal(a.frac_coords, b.frac_coords), key
        assert_same(a.site_properties, b.site_properties, f"{key}.site_properties")
        assert_same(dict(a.properties), dict(b.properties), f"{key}.properties")
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert np.array_equal(a, b), key
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (key, sorted(a.keys() ^ b.keys()))
        for k in a:
            assert_same(a[k], b[k], f"{key}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), key
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{key}[{i}]")
    elif isinstance(a, (float, np.floating)) and key.rsplit(".", 1)[-1].startswith(
        TOLERANT_PREFIXES
    ):
        assert abs(a - b) <= TOL, (key, a, b)
    else:
        assert a == b, (key, a, b)


def assert_same_results(reference, result):
    assert result.n_input_structures == reference.n_input_structures
    assert result.failed_indices == reference.failed_indices
    assert len(result.processed_structures) == len(reference.processed_structures)
    for i, (ref, got) in enumerate(
        zip(reference.processed_structures, result.processed_structures)
    ):
        assert_same(ref, got, f"structure {i}")


def assert_reference_is_informative(result, structures, stability, embeddings):
    """The fakes exercise what they should: every fault, convergence and the step cap."""
    assert result.failed_indices == []
    steps = []
    hull_gaps = 0
    for structure, processed in zip(structures, result.processed_structures):
        props = processed.properties
        elements = {el.symbol for el in structure.composition.elements}
        for name in MLIPS:
            keys = {k for k in props if k.endswith(f"_{name}")}
            if SINGLE_POINT_FAULT.get(name) in elements and name != "mace":
                # The model's whole result is dropped; only the reason is kept.
                assert keys == {f"mlip_error_{name}"}, (name, keys)
                continue
            single_point_ok = SINGLE_POINT_FAULT.get(name) not in elements
            hull_ok = HULL_GAP.get(name) not in elements
            hull_gaps += stability and not hull_ok
            assert (f"energy_{name}" in props) == single_point_ok
            assert (f"forces_{name}" in props) == single_point_ok
            assert (f"formation_energy_{name}" in props) == (
                single_point_ok and stability
            )
            assert (f"e_above_hull_{name}" in props) == (
                single_point_ok and stability and hull_ok
            )
            embedded = embeddings and EMBEDDING_FAULT.get(name) not in elements
            assert (f"graph_embedding_{name}" in props) == embedded
            if not stability:
                assert f"relaxation_steps_{name}" not in props
                assert f"relaxation_failed_{name}" not in props
            elif not single_point_ok:
                assert props[f"relaxation_error_{name}"] == (
                    f"fake {name} cannot evaluate {SINGLE_POINT_FAULT[name]}"
                )
            elif structure is structures[RELAX_FAULT]:
                assert props[f"relaxation_error_{name}"] == "fake relaxation fault"
            else:
                assert f"relaxation_failed_{name}" not in props, name
                # a hull failure leaves the relaxation itself in place
                assert f"relaxation_rmse_{name}" in props
                assert f"relaxed_formation_energy_{name}" in props
                assert (f"relaxed_e_above_hull_{name}" in props) == hull_ok
                relaxed = props[f"relaxed_structure_{name}"]
                assert isinstance(relaxed, Structure)
                if name == "uma":  # written into the relaxing atoms by fairchem
                    assert relaxed.properties["spin"] == 0
                steps.append(props[f"relaxation_steps_{name}"])
    if stability:
        assert STEPS in steps and min(steps) < STEPS, steps  # capped and converged runs
        assert hull_gaps
    charged = result.processed_structures[RELAX_FAULT].properties
    # 0.01 eV charge shift: LeMat's ORB and MACE single points see atoms.info, UMA's do not.
    orb_without_charge = charged["energy_orb"] - 0.01
    assert abs(charged["energy_uma"] - SCALE["uma"] * orb_without_charge) < 1e-9


@pytest.mark.parametrize(
    "stability, embeddings",
    [(True, True), (True, False), (False, True)],
    ids=["all", "stability_only", "embeddings_only"],
)
def test_batched_matches_per_structure(fakes, tmp_path, stability, embeddings):
    structures = make_structures()
    reference = run(deepcopy(structures), "per_structure", stability, embeddings)
    assert_reference_is_informative(reference, structures, stability, embeddings)

    inputs = deepcopy(structures)
    BATCH_LOG["path"] = tmp_path / "batches.txt"
    result = run(inputs, "batched", stability, embeddings)
    assert_same_results(reference, result)
    calls = [line.split() for line in BATCH_LOG["path"].read_text().splitlines()]
    for name in MLIPS:
        sizes = [(int(n), int(a)) for mlip, n, a in calls if mlip == name]
        assert max(n for n, _ in sizes) > 1, f"{name} was never batched"
        assert max(n for n, _ in sizes) <= BATCHED["structures_per_worker"]
        assert all(a <= BATCHED["max_atoms_per_worker"] for n, a in sizes if n > 1)
        assert any(a > BATCHED["max_atoms_per_worker"] for n, a in sizes if n == 1)
    for original, given in zip(structures, inputs):
        assert_same(
            original, given, "input"
        )  # the caller's structures are not modified


def test_relaxation_request_cap_is_applied(fakes):
    structures = make_structures()
    result = run(deepcopy(structures), "batched", max_requests=1)
    assert result.failed_indices == []
    for structure, processed in zip(structures, result.processed_structures):
        elements = {el.symbol for el in structure.composition.elements}
        for name in MLIPS:
            if SINGLE_POINT_FAULT.get(name) in elements:
                continue
            assert processed.properties[f"relaxation_failed_{name}"] is True
            assert (
                processed.properties[f"relaxation_error_{name}"]
                == "max_requests (1) exceeded"
            )
            assert f"energy_{name}" in processed.properties


@pytest.mark.parametrize(
    "kind, sighting",
    [("exit", 1), ("exit", 2), ("cuda", 1), ("cuda", 2), ("embedding", 1)],
)
def test_dead_worker_is_replaced_once(fakes, tmp_path, kind, sighting):
    structures = make_structures()
    reference = run(deepcopy(structures), "per_structure")
    CRASH.update(flag=tmp_path / "crashed", kind=kind, sighting=sighting)
    result = run(deepcopy(structures), "batched")
    assert CRASH["flag"].exists(), "no worker crashed"
    assert_same_results(reference, result)
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("kind", ["exit", "cuda"])
def test_second_dead_worker_fails_the_run(fakes, tmp_path, kind):
    CRASH.update(flag=tmp_path / "crashed", persistent=True, kind=kind)
    with pytest.raises(RuntimeError, match="orb"):
        run(make_structures(), "batched")
    assert CRASH["flag"].exists(), "no worker crashed"
    assert multiprocessing.active_children() == []


def test_defaults():
    config = MultiMLIPStabilityPreprocessor().config
    assert config.execution_mode == "per_structure"
    assert config.matmul_precision == "highest"
    assert config.batched == {
        "workers_per_model": 8,
        "structures_per_worker": 16,
        "max_atoms_per_worker": "auto",
        "geometry_tolerance": None,
        "uma_fixes": True,
        "max_requests": 1000,
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"execution_mode": "fast"}, "execution_mode"),
        ({"matmul_precision": "low"}, "matmul_precision"),
        ({"batched": {"workers": 2}}, "workers"),
        ({"batched": {"workers_per_model": 0}}, "workers_per_model"),
        ({"batched": {"structures_per_worker": 1.5}}, "structures_per_worker"),
        ({"batched": {"max_atoms_per_worker": 0}}, "max_atoms_per_worker"),
        ({"batched": {"max_atoms_per_worker": "big"}}, "max_atoms_per_worker"),
        ({"batched": {"max_requests": True}}, "max_requests"),
        ({"batched": {"geometry_tolerance": -1.0}}, "geometry_tolerance"),
        ({"batched": {"uma_fixes": "yes"}}, "uma_fixes"),
        (
            {"execution_mode": "batched", "mlip_names": ["orb", "equiformer"]},
            "equiformer",
        ),
        (
            {"matmul_precision": "high", "batched": {"geometry_tolerance": 1e-10}},
            "geometry_tolerance",
        ),
    ],
)
def test_invalid_settings_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MultiMLIPStabilityPreprocessor(**kwargs)


def by_id(result):
    return {s.properties["id"]: s for s in result.processed_structures}


LOST = "lost after retrying with the whole GPU"
FAULTS = ("oom", "cusolver", "cublas")


def test_memory_errors_are_recovered(fakes, tmp_path):
    """Every memory error in a worker held to its share is retried; no value is lost."""
    structures = make_structures()
    reference = run(deepcopy(structures), "per_structure")
    # 12 atoms: structure 14; 8 atoms: structure 9; 2 atoms: structures 2 and 4.
    MEMORY.update(
        device=(40 * GIB, 48 * GIB),
        faults={12: "oom", 8: "cusolver", 2: "cublas"},
        atom_limit=8,
        dir=tmp_path,
    )
    result = run(deepcopy(structures), "batched")
    assert_same_results(reference, result)
    assert not any(LOST in warning for warning in result.warnings)
    calls = memory_calls()
    for name in MLIPS:
        mine = [c for c in calls if c[1] == name]
        # The faults fired in workers held to a share; the structures then ran without one.
        fired = {
            n for _, _, capped, ns, ev in mine if capped and ev in FAULTS for n in ns
        }
        assert {12, 8, 2} <= fired, (name, fired)
        free = {
            n for _, _, capped, ns, ev in mine if not capped and ev == "ok" for n in ns
        }
        assert {12, 8, 2} <= free, (name, free)
        # After running out of memory a worker stops sending calls that are too large.
        for pid in {c[0] for c in mine}:
            over = [c for c in mine if c[0] == pid and c[4] == "atom_limit"]
            assert len(over) <= 3, (name, pid, len(over))
    assert multiprocessing.active_children() == []


def test_lost_values_are_reported(fakes, tmp_path):
    """A structure that runs out of memory even with the whole GPU is reported."""
    structures = make_structures()
    reference = run(deepcopy(structures), "per_structure")
    MEMORY.update(device=(40 * GIB, 48 * GIB), faults={12: "always"}, dir=tmp_path)
    result = run(deepcopy(structures), "batched")
    lost = [warning for warning in result.warnings if LOST in warning]
    for name in MLIPS:
        assert any(w.startswith(f"[{name}] structure 14:") for w in lost), (name, lost)
    assert all(w.split(":")[0].endswith("structure 14") for w in lost), lost
    got = by_id(result)
    for ref in reference.processed_structures:
        i = ref.properties["id"]
        if i != 14:
            assert_same(ref, got[i], f"structure {i}")


def test_workers_that_cannot_load_the_model_are_dropped(fakes, tmp_path):
    structures = make_structures()
    reference = run(deepcopy(structures), "per_structure")
    MEMORY.update(device=(40 * GIB, 48 * GIB), load_failures=2, dir=tmp_path)
    result = run(deepcopy(structures), "batched", workers_per_model=3)
    assert_same_results(reference, result)


def test_one_worker_without_a_share_when_none_can_load(fakes, tmp_path):
    structures = make_structures()
    reference = run(deepcopy(structures), "per_structure")
    MEMORY.update(device=(40 * GIB, 48 * GIB), load_failures=2, dir=tmp_path)
    result = run(deepcopy(structures), "batched", workers_per_model=2)
    assert_same_results(reference, result)
    assert any(fraction is None for _, fraction in recorded_limits())


def test_model_that_never_loads_fails_the_run(fakes, tmp_path):
    MEMORY.update(device=(40 * GIB, 48 * GIB), load_failures=3, dir=tmp_path)
    with pytest.raises(RuntimeError, match="orb"):
        run(make_structures(), "batched", workers_per_model=2)
    assert multiprocessing.active_children() == []


def test_memory_share_and_atom_budget(fakes, tmp_path, monkeypatch):
    """Workers, share and atom budget follow from the free memory and the model constants."""
    per_atom = 100 * 2**20
    monkeypatch.setattr(stage, "MEMORY_MODEL", {name: (0, per_atom) for name in MLIPS})
    monkeypatch.setattr(stage, "MIN_ATOM_BUDGET", 12)
    free, total = 13 * GIB, 20 * GIB
    MEMORY.update(device=(free, total), dir=tmp_path)
    structures = make_structures()
    reference = run(deepcopy(structures), "per_structure")
    result = run(
        deepcopy(structures),
        "batched",
        workers_per_model=8,
        max_atoms_per_worker="auto",
    )
    assert_same_results(reference, result)

    reserve = max(stage.MEMORY_RESERVE_MIN, stage.MEMORY_RESERVE_FRACTION * total)

    def budget(workers):
        share = (free - reserve) / workers - stage.CUDA_CONTEXT_BYTES
        return share, int(share // (stage.ATOM_MARGIN * per_atom))

    workers = 8
    while budget(workers)[1] < 12:
        workers -= 1
    share, atoms = budget(workers)
    assert (workers, atoms) == (6, 12)
    # Single points that fail in a capped worker are retried by at most one final
    # worker per model, without a share and without an atom budget.
    limits = [fraction for _, fraction in recorded_limits()]
    capped = [fraction for fraction in limits if fraction is not None]
    assert capped and all(f == pytest.approx(share / total) for f in capped), limits
    assert limits.count(None) <= len(MLIPS), limits
    capped_calls = [call for call in memory_calls() if call[2]]
    assert max(sum(counts) for _, _, _, counts, _ in capped_calls) <= atoms
    for name in MLIPS:
        pids = {call[0] for call in capped_calls if call[1] == name}
        assert 1 < len(pids) <= workers, (name, pids)

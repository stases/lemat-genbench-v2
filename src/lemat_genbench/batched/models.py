"""Batched energy, force and stress evaluation using loaded ASE calculators."""

from __future__ import annotations

import hashlib
import inspect
from types import MethodType
from typing import Sequence

import torch
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from torch.nn import functional as F

from lemat_genbench.batched.types import BatchedModel, Prediction

# SHA-256 of inspect.getsource(function).encode("utf-8") in fairchem-core 2.3.0.
# Revalidate numerical equivalence before updating these for another version.
_UMA_SOURCE_HASHES = {
    "MOLE.forward": "447d0dea7cb525cbf8443136172b408abeaf5a471aa7224805fc1a303f7cfb75",
    "generate_graph": "2d368aa7b141755b40becd614fe922342c71a57b73319ea8d8cd13095b0b379a",
    "eSCNMDBackbone._generate_graph": (
        "1d387bb8fd1fe2df29d0a473439895a5ccebe029f5dc044f04a0f6aa310171c5"
    ),
}


def _atom_counts(atoms: Sequence[Atoms]) -> list[int]:
    counts = []
    for structure in atoms:
        if not structure.pbc.all():
            raise ValueError("Batched models require fully periodic structures.")
        if len(structure) == 0:
            raise ValueError("Atoms object has no atoms inside.")
        counts.append(len(structure))
    return counts


def _split_predictions(
    counts: Sequence[int],
    energy: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    *,
    energy_units_to_eV: float = 1.0,
    length_units_to_A: float = 1.0,
) -> list[Prediction]:
    """Detach, apply ASE's unit conversions, and split without changing dtypes.

    Stress may already be Voigt (ORB), or contain nine entries per system
    (MACE/UMA). ASE averages the off-diagonal pairs of a full stress tensor.
    Energy conversion happens after conversion to a Python float, as in MACE.
    """
    energies = energy.detach().cpu().numpy().reshape(-1)
    force_array = forces.detach().cpu().numpy() * energy_units_to_eV / length_units_to_A
    stress_array = (
        stress.detach().cpu().numpy() * energy_units_to_eV / length_units_to_A**3
    )
    if stress_array.shape != (len(counts), 6):
        # Convert each matrix separately, as ASE does. NumPy 1.x can promote
        # scalar off-diagonal arithmetic differently from batched arithmetic.
        stress_array = [
            full_3x3_to_voigt_6_stress(matrix)
            for matrix in stress_array.reshape(len(counts), 3, 3)
        ]
    if len(energies) != len(counts) or force_array.shape != (sum(counts), 3):
        raise ValueError("Model output shapes do not match the input structures.")
    predictions = []
    start = 0
    for i, count in enumerate(counts):
        predictions.append(
            Prediction(
                energy=float(energies[i]) * energy_units_to_eV,
                forces=force_array[start : start + count].copy(),
                stress=stress_array[i].copy(),
            )
        )
        start += count
    return predictions


class OrbBatchedModel:
    """Use ORB's own graph settings and selected direct/conservative heads."""

    def __init__(self, calculator):
        self.ase_calc = calculator.ase_calc
        self.model = self.ase_calc.model

    @torch.enable_grad()
    def predict(self, atoms: Sequence[Atoms]) -> list[Prediction]:
        counts = _atom_counts(atoms)
        if not counts:
            return []
        from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
        from orb_models.forcefield.base import batch_graphs

        ac = self.ase_calc
        graphs = [
            ase_atoms_to_atom_graphs(
                structure.copy(),
                system_config=ac.system_config,
                max_num_neighbors=ac.max_num_neighbors,
                edge_method=ac.edge_method,
                half_supercell=ac.half_supercell,
                device=ac.device,
            )
            for structure in atoms
        ]
        out = self.model.predict(batch_graphs(graphs).to(ac.device))
        forces_key = self.model.grad_forces_name if ac.conservative else "forces"
        stress_key = self.model.grad_stress_name if ac.conservative else "stress"
        return _split_predictions(
            counts, out["energy"], out[forces_key], out[stress_key]
        )


class MaceBatchedModel:
    """One batch forward per committee member, with ASE's averaging and units."""

    def __init__(self, calculator):
        self.ase_calc = calculator.ase_calc
        if self.ase_calc.model_type not in ("MACE", "EnergyDipoleMACE"):
            raise ValueError(
                "The MACE calculator must provide energy, forces and stress."
            )

    @torch.enable_grad()
    def predict(self, atoms: Sequence[Atoms]) -> list[Prediction]:
        counts = _atom_counts(atoms)
        if not counts:
            return []
        from mace import data
        from mace.tools import torch_geometric

        ac = self.ase_calc
        keyspec = data.KeySpecification(
            info_keys={}, arrays_keys={"charges": ac.charges_key}
        )
        dataset = [
            data.AtomicData.from_config(
                data.config_from_atoms(
                    structure.copy(), key_specification=keyspec, head_name=ac.head
                ),
                z_table=ac.z_table,
                cutoff=ac.r_max,
                heads=ac.available_heads,
            )
            for structure in atoms
        ]
        loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False
        )
        batch = next(iter(loader)).to(ac.device)
        # ASE allocates result buffers in torch's current default dtype, which
        # its loader also selects for featurisation; do not force float32 here.
        energies = torch.zeros(ac.num_models, len(counts), device=ac.device)
        forces = torch.zeros(ac.num_models, sum(counts), 3, device=ac.device)
        stresses = torch.zeros(ac.num_models, len(counts), 3, 3, device=ac.device)
        for i, model in enumerate(ac.models):
            out = model(
                ac._clone_batch(batch).to_dict(),
                compute_stress=not ac.use_compile,
                training=ac.use_compile,
                compute_edge_forces=ac.compute_atomic_stresses,
                compute_atomic_stresses=ac.compute_atomic_stresses,
            )
            energies[i] = out["energy"].detach().reshape(-1)
            forces[i] = out["forces"].detach()
            if out["stress"] is not None:
                stresses[i] = out["stress"].detach()
        if out["stress"] is None:
            raise ValueError(
                "The MACE calculator did not provide stress; compiled MACE "
                "cannot supply the required Prediction."
            )
        return _split_predictions(
            counts,
            energies.mean(dim=0),
            forces.mean(dim=0),
            stresses.mean(dim=0),
            energy_units_to_eV=ac.energy_units_to_eV,
            length_units_to_A=ac.length_units_to_A,
        )


class UmaBatchedModel:
    """Batch with FAIRChem's metadata validation and prediction unit.

    Enabled fixes also affect the shared ASE calculator and embedding extractor.
    Disabling fixes leaves the instance as-is; it cannot undo earlier fixes.
    """

    def __init__(self, calculator, *, uma_fixes: bool = True):
        self.ase_calc = calculator.ase_calc
        self.predictor = self.ase_calc.predictor
        if uma_fixes:
            _apply_uma_fixes(self.predictor.model)

    @torch.enable_grad()
    def predict(self, atoms: Sequence[Atoms]) -> list[Prediction]:
        counts = _atom_counts(atoms)
        if not counts:
            return []
        from fairchem.core.datasets import data_list_collater

        data = []
        for structure in atoms:
            structure = structure.copy()
            self.ase_calc._check_atoms_pbc(structure)
            self.ase_calc._validate_charge_and_spin(structure)
            data.append(self.ase_calc.a2g(structure))
        # fairchem rejects an entirely edgeless batch, but a mixed batch can
        # silently produce energies for edgeless systems. Preserve ASE failures,
        # using the per-system edge counts the forward pass writes to mole_sizes.
        try:
            mole_tensors = self.predictor.model.module.backbone.global_mole_tensors
            # cleared, so that a forward pass that stops writing it fails below
            mole_tensors.mole_sizes = None
        except AttributeError as exc:
            raise RuntimeError(
                "Unsupported fairchem internals: UMA per-system edge counts "
                "are unavailable in backbone.global_mole_tensors.mole_sizes."
            ) from exc
        # The predictor owns the device transfer, just as in ASE.calculate().
        out = self.predictor.predict(data_list_collater(data, otf_graph=True))
        edge_counts = mole_tensors.mole_sizes
        if not isinstance(edge_counts, torch.Tensor) or edge_counts.shape != (
            len(counts),
        ):
            raise RuntimeError(
                "Unsupported fairchem internals: expected UMA mole_sizes to contain "
                f"{len(counts)} per-system edge counts, got shape "
                f"{getattr(edge_counts, 'shape', None)}."
            )
        empty_positions = (edge_counts == 0).nonzero(as_tuple=True)[0].tolist()
        if empty_positions:
            raise ValueError(
                "No edges found in input system at batch positions "
                f"{empty_positions} (zero-based)."
            )
        return _split_predictions(counts, out["energy"], out["forces"], out["stress"])


def _mole_forward_split(self, x):
    """Stock MOLE arithmetic with one split backward instead of many slices."""
    with torch.autocast(device_type=self.weights.device.type, enabled=False):
        weights = torch.einsum(
            "eoi, be->boi",
            self.weights,
            self.global_mole_tensors.expert_mixing_coefficients,
        )
    lo = self.global_mole_tensors.ac_start_idx
    hi = lo + x.shape[0]
    indices, lengths = [], []
    start = 0
    for i, size in enumerate(self.global_mole_tensors.mole_sizes.tolist()):
        left, right = max(lo, start), min(hi, start + size)
        # Stock interval_intersection includes empty boundary intersections.
        if left <= right:
            indices.append(i)
            lengths.append(right - left)
        start += size
    parts = torch.split(x, lengths, dim=0)
    return torch.cat(
        [F.linear(part, weights[i], bias=self.bias) for part, i in zip(parts, indices)],
        dim=0,
    )


def _generate_graph_batched(self, data_dict):
    """Replace only the eligible radius-graph call, preserving backbone outputs."""
    from fairchem.core.common import gp_utils
    from fairchem.core.graph.compute import get_pbc_distances
    from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2

    # Strict truncation breaks ties by edge enumeration order, which can differ
    # between one-system and batched radius graphs. Retain stock choices there.
    if (
        not self.otf_graph
        or self.radius_pbc_version != 2
        or self.enforce_max_neighbors_strictly
    ):
        return self._lemat_stock_generate_graph(data_dict)
    pbc = (
        torch.ones(len(data_dict), 3, dtype=torch.bool)
        if self.always_use_pbc
        else data_dict.get("pbc", None)
    )
    if pbc is None or not (pbc.all() or (~pbc).all()):
        return self._lemat_stock_generate_graph(data_dict)
    edge_index, cell_offsets, neighbors = radius_graph_pbc_v2(
        data_dict,
        self.cutoff,
        self.max_neighbors,
        self.enforce_max_neighbors_strictly,
        pbc=pbc[0],
    )
    out = get_pbc_distances(
        data_dict.pos,
        edge_index,
        data_dict.cell,
        cell_offsets,
        neighbors,
        return_offsets=True,
        return_distance_vec=True,
    )
    graph = {
        "edge_index": out["edge_index"],
        "edge_distance": out["distances"],
        "edge_distance_vec": out["distance_vec"],
        "cell_offsets": cell_offsets,
        "offset_distances": out["offsets"],
        "neighbors": neighbors,
    }
    if gp_utils.initialized():
        graph = self._init_gp_partitions(graph, data_dict["atomic_numbers_full"])
        node_partition = graph["node_partition"]
        data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][node_partition]
        data_dict["batch"] = data_dict["batch_full"][node_partition]
    else:
        graph["node_offset"] = 0
        graph["edge_distance_vec_full"] = graph["edge_distance_vec"]
        graph["edge_distance_full"] = graph["edge_distance"]
        graph["edge_index_full"] = graph["edge_index"]
    return graph


def _require_attributes(obj, names, label):
    missing = [name for name in names if not hasattr(obj, name)]
    if missing:
        raise RuntimeError(
            f"Unsupported fairchem internals: {label} is missing {missing}; "
            "review UMA fixes for this fairchem version."
        )


def _apply_uma_fixes(model):
    """Validate all targets before making idempotent, instance-only bindings."""
    from fairchem.core.graph.compute import generate_graph
    from fairchem.core.models.uma.escn_md import eSCNMDBackbone
    from fairchem.core.models.uma.nn.mole import MOLE

    for name, method in (
        ("MOLE.forward", MOLE.forward),
        ("generate_graph", generate_graph),
        ("eSCNMDBackbone._generate_graph", eSCNMDBackbone._generate_graph),
    ):
        error = (
            f"Unsupported fairchem source for {name}. UMA fixes were validated "
            "for fairchem 2.3.0; uma_fixes=False (config key "
            "batched.uma_fixes: false) runs stock fairchem."
        )
        try:
            source_hash = hashlib.sha256(
                inspect.getsource(method).encode("utf-8")
            ).hexdigest()
        except (OSError, TypeError) as exc:
            raise RuntimeError(error) from exc
        if source_hash != _UMA_SOURCE_HASHES[name]:
            raise RuntimeError(error)
    modules = list(model.modules())
    backbones = [module for module in modules if isinstance(module, eSCNMDBackbone)]
    if not backbones:
        raise RuntimeError(
            "Unsupported fairchem model: no eSCNMDBackbone for UMA fixes."
        )
    moles = [module for module in modules if isinstance(module, MOLE)]
    if not moles:
        raise RuntimeError(
            "Unsupported fairchem model: no stock MOLE modules for UMA fixes; "
            "uma_fixes=False (config key batched.uma_fixes: false) runs stock fairchem."
        )
    for module in moles:
        _require_attributes(module, ("weights", "bias", "global_mole_tensors"), "MOLE")
        _require_attributes(
            module.global_mole_tensors,
            ("expert_mixing_coefficients", "mole_sizes", "ac_start_idx"),
            "MOLEGlobals",
        )
        if getattr(module.forward, "__func__", None) not in (
            MOLE.forward,
            _mole_forward_split,
        ):
            raise RuntimeError("Unsupported fairchem MOLE.forward override.")
    for backbone in backbones:
        _require_attributes(
            backbone,
            (
                "otf_graph",
                "always_use_pbc",
                "radius_pbc_version",
                "cutoff",
                "max_neighbors",
                "enforce_max_neighbors_strictly",
                "_init_gp_partitions",
            ),
            "eSCNMDBackbone",
        )
        method = getattr(backbone._generate_graph, "__func__", None)
        if method is _generate_graph_batched:
            _require_attributes(
                backbone, ("_lemat_stock_generate_graph",), "patched eSCNMDBackbone"
            )
        elif method is not eSCNMDBackbone._generate_graph:
            raise RuntimeError("Unsupported fairchem eSCNMDBackbone._generate_graph.")
    for module in moles:
        if module.forward.__func__ is not _mole_forward_split:
            module.forward = MethodType(_mole_forward_split, module)
    for backbone in backbones:
        if backbone._generate_graph.__func__ is not _generate_graph_batched:
            backbone._lemat_stock_generate_graph = backbone._generate_graph
            backbone._generate_graph = MethodType(_generate_graph_batched, backbone)


def make_batched_model(
    mlip_name: str, calculator, *, uma_fixes: bool = True
) -> BatchedModel:
    """Adapt an already-loaded LeMat calculator without loading any checkpoints."""
    if mlip_name == "orb":
        return OrbBatchedModel(calculator)
    if mlip_name == "mace":
        return MaceBatchedModel(calculator)
    if mlip_name == "uma":
        return UmaBatchedModel(calculator, uma_fixes=uma_fixes)
    raise ValueError(f"Unknown batched MLIP: {mlip_name!r}")

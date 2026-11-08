import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from pymatgen.core import Composition, Element
from scipy import sparse
from scipy.optimize import linprog
from tqdm import tqdm

CURRENT_FOLDER = os.path.dirname(Path(__file__).resolve())
_HULL_REFERENCE_ARRAYS = {}


def build_formation_energy_reference_file():
    ds_pbe = load_dataset("LeMaterial/LeMat-Bulk", "compatible_pbe")
    # ds_pbesol = load_dataset("LeMaterial/LeMat-Bulk", "compatible_pbesol")
    # ds_scan = load_dataset("LeMaterial/LeMat-Bulk", "compatible_scan")

    data = {
        "energy": [
            *[
                x / z
                for x, y, z in zip(
                    ds_pbe["train"]["energy"],
                    ds_pbe["train"]["nelements"],
                    ds_pbe["train"]["nsites"],
                )
                if y == 1
            ],
        ],
        "composition": [
            *[
                [x for x in Composition(Counter(y)).chemical_system_set][0]
                for x, y in zip(
                    ds_pbe["train"]["nelements"], ds_pbe["train"]["species_at_sites"]
                )
                if x == 1
            ],
        ],
    }

    element_chem_pot = {}
    for element, energy in (
        pd.DataFrame(data).groupby("composition").min().to_dict()["energy"].items()
    ):
        if element not in element_chem_pot:
            element_chem_pot[element] = {}
        element_chem_pot[element]["pbe"] = energy

    json.dump(
        element_chem_pot,
        open(os.path.join(CURRENT_FOLDER, "element_chem_pot.json"), "w"),
    )


def get_formation_energy_from_composition_energy(
    total_energy: float, composition: str | Composition, functional="pbe"
):
    element_chem_pot_file = os.path.join(CURRENT_FOLDER, "element_chem_pot.json")

    if not os.path.exists(element_chem_pot_file):
        raise FileNotFoundError(
            f"Reference energy file not found: {element_chem_pot_file}. "
            "Run build_formation_energy_reference_file() first."
        )

    with open(element_chem_pot_file) as f:
        element_chem_pot = json.load(f)

    try:
        res = 0
        # Handle charged species by converting to neutral elements
        neutral_composition_dict = {}
        missing_elements = []

        for element, count in composition.as_dict().items():
            # Handle charged species by extracting the base element
            # element can be a string like 'Cs+' or a Species object
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Check if reference energy is available
            if base_element not in element_chem_pot:
                missing_elements.append(base_element)
                continue

            if functional not in element_chem_pot[base_element]:
                raise ValueError(
                    f"Functional '{functional}' not available for element '{base_element}'"
                )

            # Use neutral element for chemical potential lookup
            if base_element not in neutral_composition_dict:
                neutral_composition_dict[base_element] = 0
            neutral_composition_dict[base_element] += count

        if missing_elements:
            raise ValueError(
                f"Missing reference energies for elements: {missing_elements}"
            )

        res = total_energy - sum(
            [
                element_chem_pot[k][functional] * v
                for k, v in neutral_composition_dict.items()
            ]
        )
        return res
    except Exception as e:
        print("Error in get_formation_energy_from_composition_energy: ", e)
        return None


def get_formation_energy_per_atom_from_composition_energy(
    total_energy, composition, functional="pbe"
):
    """Calculate formation energy per atom from total energy and composition.

    This function properly handles charged species by converting them
    to neutral elements before looking up reference chemical potentials.

    Parameters
    ----------
    total_energy : float
        Total energy in eV
    composition : Composition
        Pymatgen composition object (may contain charged species)
    functional : str, optional
        DFT functional to use for reference energies (default is "pbe")

    Returns
    -------
    float
        Formation energy per atom in eV/atom (intensive property, normalized per atom)

    Notes
    -----
    This follows Materials Project conventions for formation energy.
    """

    try:
        formation_energy = get_formation_energy_from_composition_energy(
            total_energy, composition, functional=functional
        )
        composition = Composition(composition)  # Ensure it's a Composition object
        formation_energy = formation_energy / composition.num_atoms
        if formation_energy is None:
            raise ValueError("Formation energy calculation returned None")
        return formation_energy
    except Exception as e:
        raise ValueError(
            f"Failed to compute formation energy for {composition.formula}: {str(e)}"
        ) from e


def one_hot_encode_composition(elements):
    one_hot = np.zeros(
        119
    )  # 119 to use direct Z indexing (0 unused, 1-118 for elements)
    for element in elements:
        try:
            # Handle charged species by extracting the base element
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Validate element and get atomic number
            element_obj = Element(base_element)
            one_hot[int(element_obj.number)] = 1
        except ValueError as e:
            print(f"Warning: Invalid element '{element}': {e}")
            continue
    return one_hot


def process_chunk(chunk):
    one_hot_compositions = []
    for elements in tqdm(chunk):
        one_hot_compositions.append(one_hot_encode_composition(elements))
    return one_hot_compositions


@lru_cache(maxsize=None)
def _retrieve_df(hull_type="dft", threshold=0.001):
    """Retrieve dataset for hull computations.

    Parameters
    ----------
    hull_type : str, optional
        Type of hull to use ('dft', 'orb', 'uma', 'mace_mp', 'mace_omat')
        Default is 'dft' for backward compatibility
    threshold : float, optional
        Energy above hull threshold in eV/atom (default 0.001)

    Returns
    -------
    pd.DataFrame
        Dataset of materials close to the hull
    """
    try:
        threshold_str = f"{threshold:.3f}".replace(".", "_")

        # Try local hull-specific path first
        local_hull_path = os.path.join(
            CURRENT_FOLDER,
            "..",
            "..",
            "..",
            "data",
            "convex_hulls",
            f"{hull_type}_above_hull_dataset.parquet",
        )

        if os.path.exists(local_hull_path):
            dataset = pd.read_parquet(local_hull_path)
            if "elements" in dataset.columns:
                dataset["elements"] = dataset["elements"].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )
            if "species_at_sites" in dataset.columns:
                dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )
            return dataset

        try:
            from datasets import load_dataset

            dataset_dict = load_dataset("LeMaterial/LeMat-Bulk-MLIP-Hull")

            if hull_type in dataset_dict:
                dataset = dataset_dict[hull_type].to_pandas()
                # species_at_sites should already be lists in the new format
                # but we'll still check just in case
                if "species_at_sites" in dataset.columns:
                    dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                        lambda x: x.tolist() if hasattr(x, "tolist") else x
                    )
                return dataset
        except Exception:
            pass

        # Fall back to file-based approach
        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(
            repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
            filename=f"threshold_{threshold_str}/{hull_type}_above_hull_dataset.parquet",
            repo_type="dataset",
            cache_dir=os.path.join(CURRENT_FOLDER, "..", "..", "..", "data", ".cache"),
        )
        dataset = pd.read_parquet(file_path)
        if "elements" in dataset.columns:
            dataset["elements"] = dataset["elements"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
        if "species_at_sites" in dataset.columns:
            dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Hull-specific dataset not found for hull type '{hull_type}' and threshold {threshold}. "
            "Tried local path and HuggingFace Hub."
            f" Error: {e}"
        )


@lru_cache(maxsize=None)
def _retrieve_matrix(hull_type="dft", threshold=0.001):
    """Retrieve composition matrix for hull computations.

    Parameters
    ----------
    hull_type : str, optional
        Type of hull to use ('dft', 'orb', 'uma', 'mace_mp', 'mace_omat')
        Default is 'dft' for backward compatibility
    threshold : float, optional
        Energy above hull threshold in eV/atom (default 0.001)

    Returns
    -------
    np.ndarray
        Composition matrix
    """
    try:
        threshold_str = f"{threshold:.3f}".replace(".", "_")

        local_hull_path = os.path.join(
            CURRENT_FOLDER,
            "..",
            "..",
            "..",
            "data",
            "convex_hulls",
            f"{hull_type}_above_hull_composition_matrix.npz",
        )

        if os.path.exists(local_hull_path):
            composition_array = sparse.load_npz(local_hull_path).toarray()
            return composition_array

        # otherwise fetch from HuggingFace Hub with hull type
        try:
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(
                repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
                filename=f"threshold_{threshold_str}/{hull_type}_above_hull_composition_matrix.npz",
                repo_type="dataset",
                cache_dir=os.path.join(
                    CURRENT_FOLDER, "..", "..", "..", "data", ".cache"
                ),
            )
            composition_array = sparse.load_npz(file_path).toarray()
            return composition_array
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(f"Failed to load composition matrix: {e}") from e


def filter_df(df, matrix, composition):
    try:
        structure_vector = one_hot_encode_composition(composition.elements).reshape(
            -1, 1
        )
        forbidden_elements = 1 - structure_vector
        intersection_elements = df.loc[(matrix @ forbidden_elements) == 0]

        if intersection_elements.empty:
            print(
                f"Warning: No reference structures found for composition {composition.formula}"
            )

        return intersection_elements
    except Exception as e:
        raise ValueError(f"Failed to filter reference dataset: {e}") from e


def hull_energy_per_atom(
    reference_fractions, reference_energies_per_atom, target_fractions
) -> float:
    """Return the minimum mixture energy at the target atomic fractions, in eV/atom.

    Reference fractions have shape (n_entries, n_elements), reference energies
    have shape (n_entries,), and target fractions have shape (n_elements,).
    Each composition is normalized to one atom. Raise ValueError if the linear
    program has no optimal solution, including an infeasible target composition.
    """
    result = linprog(
        reference_energies_per_atom,
        A_eq=np.asarray(reference_fractions).T,
        b_eq=target_fractions,
        bounds=(0, None),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not result.success:
        raise ValueError(f"Hull linear program failed: {result.message}")
    return float(result.fun)


def _hull_reference_arrays(df, hull_type, threshold):
    """Cache fractions and eV/atom by table identity, in positional row order."""
    key = (hull_type, threshold)
    cached = _HULL_REFERENCE_ARRAYS.get(key)
    if cached is not None and cached[0] is df:
        return cached[1:]

    # Sparse storage avoids a dense 119-column array for every full hull table.
    indptr = [0]
    atomic_numbers = []
    fractions = []
    num_atoms = []
    for species in df["species_at_sites"]:
        num_atoms.append(len(species))
        for symbol, count in Counter(species).items():
            atomic_numbers.append(Element(symbol).Z)
            fractions.append(count / len(species))
        indptr.append(len(atomic_numbers))

    reference_fractions = sparse.csr_matrix(
        (fractions, atomic_numbers, indptr), shape=(len(df), 119)
    )
    energies_per_atom = df["energy"].to_numpy(dtype=float) / np.asarray(num_atoms)
    _HULL_REFERENCE_ARRAYS[key] = (df, reference_fractions, energies_per_atom)
    return reference_fractions, energies_per_atom


def get_energy_above_hull(total_energy, composition, hull_type="dft", threshold=0.001):
    """Calculate energy above hull from total energy and composition.

    This function properly handles charged species by converting them
    to neutral elements before solving for the minimum reference mixture energy.

    Parameters
    ----------
    total_energy : float
        Total energy in eV
    composition : Composition
        Pymatgen composition object (may contain charged species)
    hull_type : str, optional
        Type of hull to use ('dft', 'orb', 'uma', 'mace_mp', 'mace_omat')
        Default is 'dft' for backward compatibility
    threshold : float, optional
        Energy above hull threshold in eV/atom for reference dataset (default 0.001)

    Returns
    -------
    float
        Energy above hull in eV/atom (intensive property, normalized per atom)

    Notes
    -----
    The hull energy is computed with a linear program over reference atomic
    fractions. The result is in eV/atom and may be negative, matching pymatgen's
    get_decomp_and_e_above_hull() with allow_negative=True.
    """

    try:
        reference_df = _retrieve_df(hull_type, threshold)
        intersection_elements = filter_df(
            reference_df,
            _retrieve_matrix(hull_type, threshold),
            composition,
        )

        if intersection_elements.empty:
            raise ValueError(
                f"No entries found in dataset containing any of the elements in: {composition.elements}"
            )

        # Convert charged species to a neutral composition for hull evaluation
        # This follows the same logic as get_formation_energy_from_composition_energy
        neutral_composition_dict = {}
        for element, count in composition.as_dict().items():
            # Handle charged species by extracting the base element
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Use the neutral element for hull evaluation
            if base_element not in neutral_composition_dict:
                neutral_composition_dict[base_element] = 0
            neutral_composition_dict[base_element] += count

        neutral_composition = Composition(neutral_composition_dict)

        reference_fractions, reference_energies = _hull_reference_arrays(
            reference_df, hull_type, threshold
        )
        rows = reference_df.index.get_indexer(intersection_elements.index)
        elements = neutral_composition.elements
        fractions = reference_fractions[rows][:, [el.Z for el in elements]].toarray()
        missing_elements = [
            el.symbol
            for el, present in zip(elements, (fractions == 1).any(axis=0))
            if not present
        ]
        if missing_elements:
            raise ValueError(
                f"Missing terminal entries for elements {', '.join(missing_elements)}"
            )

        target_fractions = [
            neutral_composition.get_atomic_fraction(el) for el in elements
        ]
        hull_energy = hull_energy_per_atom(
            fractions, reference_energies[rows], target_fractions
        )
        # np.float64, as pymatgen returned, keeps float32 energies from rounding the result
        return total_energy / neutral_composition.num_atoms - np.float64(hull_energy)

    except Exception as e:
        raise ValueError(
            f"Failed to compute energy above hull for {composition.formula}: {str(e)}"
        ) from e

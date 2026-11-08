"""Tests for the linear-programme convex hull in ``preprocess/reference_energies.py``.

``hull_energy_per_atom`` must agree with pymatgen's ``PhaseDiagram`` to 1e-9 eV/atom, and
``get_energy_above_hull`` must return what the ``PhaseDiagram`` version returned, errors included.
"""

import time
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Species

from lemat_genbench.preprocess import reference_energies
from lemat_genbench.preprocess.reference_energies import (
    get_energy_above_hull,
    one_hot_encode_composition,
)

POOL = ["Li", "O", "Fe", "Mn", "P", "S", "Cu", "Zn", "Ga", "N", "Mg", "Si"]
TOL = 1e-9  # eV/atom


def random_entries(rng, elements, n_compounds):
    """Elemental polymorphs for every element plus random compounds, as (counts, total energy) pairs."""
    entries = []
    for el in elements:
        for _ in range(int(rng.integers(1, 3))):
            n = int(rng.integers(1, 5))
            entries.append(({el: n}, n * rng.normal(-3.0, 1.0)))
    for _ in range(n_compounds):
        k = int(rng.integers(2, len(elements) + 1)) if len(elements) > 1 else 1
        counts = {
            el: int(rng.integers(1, 5)) for el in rng.choice(elements, k, replace=False)
        }
        entries.append((counts, sum(counts.values()) * rng.normal(-3.5, 1.0)))
    return entries


def random_target(rng, elements):
    k = int(rng.integers(1, len(elements) + 1))
    return {
        el: int(rng.integers(1, 7)) for el in rng.choice(elements, k, replace=False)
    }


def fractions(counts, elements):
    total = sum(counts.values())
    return np.array([counts.get(el, 0) / total for el in elements])


def entries_dataframe(entries):
    """A reference table in the format of LeMat's hull datasets (float32 energies), and its element matrix."""
    df = pd.DataFrame(
        {
            "species_at_sites": [
                [el for el, n in c.items() for _ in range(n)] for c, _ in entries
            ],
            "energy": np.array([e for _, e in entries], dtype=np.float32),
        }
    )
    matrix = np.stack([one_hot_encode_composition(list(c)) for c, _ in entries])
    return df, matrix


def phase_diagram_e_above_hull(entries, composition, total_energy):
    """The previous implementation of get_energy_above_hull, on the same filtered entries.

    Table energies are rounded to float32, as stored in the table, and read back as floats, as the previous
    implementation did through DataFrame.iterrows.
    """
    elements = {str(el) for el in composition.elements}
    pd_entries = [
        PDEntry(Composition(c), float(np.float32(e)))
        for c, e in entries
        if set(c) <= elements
    ]
    diagram = PhaseDiagram(pd_entries)
    return diagram.get_decomp_and_e_above_hull(
        PDEntry(composition, total_energy), allow_negative=True
    )[1]


@pytest.fixture
def patched_hull_data(monkeypatch):
    def install(entries):
        df, matrix = entries_dataframe(entries)
        monkeypatch.setattr(
            reference_energies,
            "_retrieve_df",
            lambda hull_type="dft", threshold=0.001: df,
        )
        monkeypatch.setattr(
            reference_energies,
            "_retrieve_matrix",
            lambda hull_type="dft", threshold=0.001: matrix,
        )

    return install


@pytest.mark.parametrize("n_elements", [1, 2, 3, 4, 5])
def test_hull_energy_matches_phase_diagram(n_elements):
    from lemat_genbench.preprocess.reference_energies import hull_energy_per_atom

    rng = np.random.default_rng(n_elements)
    for _ in range(10):
        elements = [str(el) for el in rng.choice(POOL, n_elements, replace=False)]
        entries = random_entries(rng, elements, n_compounds=6 * n_elements)
        diagram = PhaseDiagram([PDEntry(Composition(c), e) for c, e in entries])
        ref_x = np.stack([fractions(c, elements) for c, _ in entries])
        ref_e = np.array([e / sum(c.values()) for c, e in entries])
        targets = [random_target(rng, elements) for _ in range(10)]
        targets += [
            entries[i][0] for i in rng.choice(len(entries), 3, replace=False)
        ]  # on reference points
        for target in targets:
            expected = diagram.get_hull_energy_per_atom(Composition(target))
            got = hull_energy_per_atom(ref_x, ref_e, fractions(target, elements))
            assert abs(got - expected) <= TOL, (target, got, expected)


def test_hull_energy_raises_when_infeasible():
    from lemat_genbench.preprocess.reference_energies import hull_energy_per_atom

    ref_x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    ref_e = np.array([-3.0, -4.0, -4.2])
    with pytest.raises(ValueError):
        hull_energy_per_atom(ref_x, ref_e, np.array([0.4, 0.4, 0.2]))


@pytest.mark.parametrize("seed", range(5))
def test_energy_above_hull_matches_phase_diagram(patched_hull_data, seed):
    rng = np.random.default_rng(100 + seed)
    system = [str(el) for el in rng.choice(POOL, 6, replace=False)]
    entries = random_entries(rng, system, n_compounds=40)
    patched_hull_data(entries)
    for _ in range(20):
        counts = random_target(rng, system[:4])
        composition = Composition(counts)
        total_energy = composition.num_atoms * rng.normal(-3.3, 0.7)
        expected = phase_diagram_e_above_hull(entries, composition, total_energy)
        got = get_energy_above_hull(total_energy, composition)
        assert abs(got - expected) <= TOL, (counts, got, expected)


def test_energy_above_hull_with_charged_species(patched_hull_data):
    entries = [
        ({"Fe": 2}, -16.0),
        ({"O": 2}, -9.8),
        ({"Fe": 2, "O": 3}, -38.0),
        ({"Fe": 3, "O": 4}, -52.0),
    ]
    patched_hull_data(entries)
    charged = Composition({Species("Fe", 3): 2, Species("O", -2): 3})
    neutral = Composition({"Fe": 2, "O": 3})
    expected = phase_diagram_e_above_hull(entries, neutral, -37.5)
    assert abs(get_energy_above_hull(-37.5, charged) - expected) <= TOL
    assert abs(get_energy_above_hull(-37.5, neutral) - expected) <= TOL


def test_energy_above_hull_float32_energy(patched_hull_data):
    """A float32 total energy gives a float64 result, as with PhaseDiagram."""
    entries = [({"Fe": 2}, -16.0), ({"O": 2}, -9.8), ({"Fe": 2, "O": 3}, -38.0)]
    patched_hull_data(entries)
    composition = Composition("Fe3O4")
    energy = np.float32(-48.123457)
    got = get_energy_above_hull(energy, composition)
    assert isinstance(got, np.float64)
    assert abs(got - phase_diagram_e_above_hull(entries, composition, energy)) <= TOL


def test_energy_above_hull_missing_element_raises(patched_hull_data):
    patched_hull_data([({"Fe": 1}, -8.0), ({"O": 2}, -9.8), ({"Fe": 2, "O": 3}, -38.0)])
    with pytest.raises(ValueError):
        get_energy_above_hull(-30.0, Composition("LiFeO2"))


def test_energy_above_hull_missing_terminal_entry_raises(patched_hull_data):
    """No elemental reference for O: PhaseDiagram refused this, so the linear programme must refuse it too."""
    patched_hull_data(
        [({"Fe": 1}, -8.0), ({"Fe": 1, "O": 1}, -12.0), ({"Fe": 2, "O": 3}, -38.0)]
    )
    with pytest.raises(ValueError):
        get_energy_above_hull(-40.0, Composition("Fe3O4"))


def test_energy_above_hull_is_fast_for_many_elements(patched_hull_data):
    """Ten-element systems took 40 s or more with PhaseDiagram; the linear programme takes milliseconds."""
    rng = np.random.default_rng(7)
    system = [str(el) for el in rng.choice(POOL, 10, replace=False)]
    entries = random_entries(rng, system, n_compounds=400)
    patched_hull_data(entries)
    composition = Composition(Counter({el: 1 for el in system}))
    start = time.perf_counter()
    value = get_energy_above_hull(-35.0, composition)
    assert np.isfinite(value)
    assert time.perf_counter() - start < 5.0

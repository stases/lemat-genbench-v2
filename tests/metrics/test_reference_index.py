"""The composition index must give exactly the novelty, uniqueness and SUN results of the exhaustive loops.

Every expected value comes from an oracle that repeats the loops the metrics used before the index existed:
novelty compares against every reference row whose ``elements`` are a subset of the structure's elements,
standalone uniqueness compares each structure with every later one, and SUN uniqueness compares each structure
with every earlier one in its subset and stops at the first match.
"""

import itertools
import random

import numpy as np
import pytest
from datasets import Dataset
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Lattice, Structure

from lemat_genbench.fingerprinting import reference_index
from lemat_genbench.fingerprinting.encode_compositions import (
    lematbulk_item_to_structure,
)
from lemat_genbench.fingerprinting.utils import get_fingerprinter
from lemat_genbench.metrics.novelty_metric import NoveltyMetric
from lemat_genbench.metrics.sun_metric import MetaSUNMetric, SUNMetric
from lemat_genbench.metrics.uniqueness_metric import UniquenessMetric

METHOD = "structure-matcher"


def rocksalt(a, cation="Na", anion="Cl"):
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def zincblende(a, cation="Na", anion="Cl"):
    return Structure.from_spacegroup(
        "F-43m", Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )


def cscl(a, x="Na", y="Cl"):
    return Structure(Lattice.cubic(a), [x, y], [[0, 0, 0], [0.5, 0.5, 0.5]])


def bcc(a, element):
    return Structure(Lattice.cubic(a), [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]])


def simple_cubic(a, element):
    return Structure(Lattice.cubic(a), [element], [[0, 0, 0]])


def tetragonal_mgo(c_over_a):
    """c/a 1.0, 1.1 and 1.2 form a non-transitive chain: 1.0~1.1, 1.1~1.2, 1.0!~1.2."""
    return Structure(
        Lattice.tetragonal(4.0, 4.0 * c_over_a),
        ["Mg", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def fe2o3():
    return Structure(
        Lattice.orthorhombic(5.0, 5.5, 6.0),
        ["Fe", "Fe", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.1],
            [0.75, 0.25, 0.6],
            [0.25, 0.75, 0.35],
        ],
    )


def fe4o6():
    structure = fe2o3()
    structure.make_supercell([2, 1, 1])
    return structure


def lifeo2():
    return Structure(
        Lattice.tetragonal(4.1, 8.6),
        ["Li", "Fe", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.2], [0, 0, 0.7]],
    )


def nakcl2():
    return Structure(
        Lattice.tetragonal(5.64 / 2**0.5, 5.64),
        ["Na", "K", "Cl", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0, 0, 0.5]],
    )


def disordered(na):
    """Rock salt with Na/K disorder on the cation site; na=0.5 gives integer amounts, na=0.3 does not."""
    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.9),
        [{"Na": na, "K": 1 - na}, "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def mixed_valence():
    """Plain Fe and Fe2+ in one structure: pymatgen's composition test is not element-wise for it."""
    return Structure(
        Lattice.cubic(4.3),
        ["Fe", "Fe2+", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0.5, 0.5]],
    )


def perturbed(structure, distance, seed):
    rng = np.random.default_rng(seed)
    out = structure.copy()
    for i in range(len(out)):
        out.translate_sites([i], rng.normal(size=3) * distance, frac_coords=False)
    return out


def scaled(structure, volume_factor):
    out = structure.copy()
    out.scale_lattice(structure.volume * volume_factor)
    return out


def reference_rows():
    """Reference rows in LeMat-Bulk's format; row order matters for the fit counts below."""
    entries = [
        (rocksalt(5.64), None),
        (bcc(4.29, "Na"), None),
        (simple_cubic(3.1, "Cl"), None),
        (cscl(3.2), None),
        (rocksalt(6.29, "K", "Cl"), None),
        (bcc(5.23, "K"), None),
        (simple_cubic(3.3, "Br"), None),
        (fe2o3(), None),
        (rocksalt(4.3, "Fe", "O"), None),
        (lifeo2(), None),
        # The ``elements`` column decides today's candidates; this zinc-blende NaCl claims K as well, so a
        # NaCl query never sees it.
        (zincblende(6.0), ["Cl", "K", "Na"]),
        (tetragonal_mgo(1.1), None),
        (nakcl2(), None),
    ]
    rows = []
    for i, (structure, elements) in enumerate(entries):
        rows.append(
            {
                "immutable_id": f"ref-{i}",
                "elements": elements
                or sorted({site.specie.symbol for site in structure}),
                "chemical_formula_descriptive": structure.composition.formula,
                "energy": 0.0,
                "species_at_sites": [site.specie.symbol for site in structure],
                "cartesian_site_positions": structure.cart_coords.tolist(),
                "lattice_vectors": structure.lattice.matrix.tolist(),
            }
        )
    return rows


def queries():
    nacl_primitive = rocksalt(5.64).get_primitive_structure()
    return [
        perturbed(nacl_primitive, 0.02, 0),  # matches row 0
        scaled(cscl(3.2), 1.1),  # matches row 3
        rocksalt(6.6, "K", "Br"),  # no reference row with this composition
        fe4o6(),  # matches row 7
        cscl(2.7, "Fe", "O"),  # same composition as row 8, other polymorph
        rocksalt(5.64, "Na+", "Cl-"),  # oxidation states
        disordered(0.5),
        disordered(0.3),
        zincblende(6.0),  # its only match, row 10, is outside today's candidates
        tetragonal_mgo(1.0),  # matches row 11
        tetragonal_mgo(1.2),  # matches row 11
        lifeo2(),  # matches row 9
        mixed_valence(),
    ]


def oracle_novelty(structures, rows, limit=None):
    """Today's loop: every row whose elements are a subset, no early exit, NaN on any exception."""
    rows = rows[:limit] if limit is not None else rows
    matcher = get_fingerprinter(METHOD)
    values = []
    for structure in structures:
        try:
            elements = {element.symbol for element in structure.composition.elements}
            found = False
            for row in rows:
                if set(row["elements"]) <= elements:
                    found = (
                        matcher.is_equivalent(
                            structure, lematbulk_item_to_structure(row)
                        )
                        or found
                    )
            values.append(0.0 if found else 1.0)
        except Exception:
            values.append(float("nan"))
    return values


def oracle_unique_count(structures):
    matcher = get_fingerprinter(METHOD)
    return sum(
        not any(
            matcher.is_equivalent(structures[i], structures[j])
            for j in range(i + 1, len(structures))
        )
        for i in range(len(structures))
    )


def oracle_unique_in_set(structures):
    matcher = get_fingerprinter(METHOD)
    unique = []
    for i in range(len(structures)):
        if not any(
            matcher.is_equivalent(structures[i], structures[j]) for j in range(i)
        ):
            unique.append(i)
    return unique


def simple_key(structure):
    """Reduced composition of an ordered structure of plain elements; None for anything else."""
    if not structure.is_ordered or not all(
        isinstance(sp, Element) for sp in structure.composition
    ):
        return None
    return structure.composition.reduced_formula


def uniqueness_set():
    nacl = rocksalt(5.64)
    return [
        nacl,
        perturbed(nacl.get_primitive_structure(), 0.02, 1),
        rocksalt(5.64, "Na+", "Cl-"),
        nacl.get_primitive_structure(),
        cscl(3.2),
        tetragonal_mgo(1.0),
        tetragonal_mgo(1.1),
        tetragonal_mgo(1.2),
        fe2o3(),
        fe4o6(),
        rocksalt(4.3, "Fe", "O"),
        disordered(0.5),
        nakcl2(),
        disordered(0.3),
        rocksalt(6.6, "K", "Br"),
        mixed_valence(),
    ]


@pytest.fixture
def reference(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rows = reference_rows()
    dataset = Dataset.from_list(rows)
    loads = []

    def load(name, config):
        loads.append((name, config))
        return dataset

    monkeypatch.setattr(reference_index, "load_reference_dataset", load)
    reference_index.get_reference_index.cache_clear()
    yield rows, loads
    reference_index.get_reference_index.cache_clear()


@pytest.fixture
def fit_calls(monkeypatch):
    calls = []
    original = StructureMatcher.fit

    def counting(self, struct1, struct2, *args, **kwargs):
        calls.append((struct1, struct2))
        return original(self, struct1, struct2, *args, **kwargs)

    monkeypatch.setattr(StructureMatcher, "fit", counting)
    return calls


def test_fixture_cases_are_meaningful():
    """Guard the fixtures themselves: the chain is non-transitive and the oxidation-state case is asymmetric."""
    matcher = StructureMatcher(ltol=0.1)
    a, b, c = (tetragonal_mgo(r) for r in (1.0, 1.1, 1.2))
    assert matcher.fit(a, b) and matcher.fit(b, c) and not matcher.fit(a, c)
    plain, decorated = rocksalt(5.64), rocksalt(5.64, "Na+", "Cl-")
    assert matcher.fit(plain, decorated) and not matcher.fit(decorated, plain)


def test_match_key():
    key = reference_index.match_key
    assert key(fe2o3()) == key(fe4o6()) is not None
    assert (
        key(rocksalt(5.64))
        == key(cscl(3.2))
        == key(rocksalt(5.64, "Na+", "Cl-"))
        is not None
    )
    assert key(rocksalt(4.3, "Fe", "O")) != key(fe2o3())
    assert key(disordered(0.5)) == key(nakcl2()) is not None
    assert key(disordered(0.3)) is None
    assert key(mixed_valence()) is None


def test_novelty_matches_oracle(reference, fit_calls):
    rows, loads = reference
    structures = queries()
    expected = oracle_novelty(structures, rows)
    assert expected.count(0.0) >= 6 and expected.count(1.0) >= 5
    fit_calls.clear()
    result = NoveltyMetric(fingerprint_method=METHOD).compute(structures)
    np.testing.assert_array_equal(result.individual_values, expected)
    assert result.failed_indices == []
    assert len(loads) == 1


def test_novelty_compares_only_matching_compositions(reference, fit_calls):
    metric = NoveltyMetric(fingerprint_method=METHOD)
    fit_calls.clear()
    assert metric.compute([rocksalt(6.6, "K", "Br")]).individual_values == [1.0]
    assert fit_calls == []
    # Row 0 is the first NaCl row and matches: one comparison, then stop.
    assert metric.compute(
        [rocksalt(5.64).get_primitive_structure()]
    ).individual_values == [0.0]
    assert len(fit_calls) == 1


def test_novelty_max_reference_size(reference):
    rows, _ = reference
    structures = queries()
    for limit in (1, 4, 9):
        reference_index.get_reference_index.cache_clear()
        result = NoveltyMetric(
            fingerprint_method=METHOD, max_reference_size=limit
        ).compute(structures)
        np.testing.assert_array_equal(
            result.individual_values, oracle_novelty(structures, rows, limit)
        )


def test_novelty_shared_across_metrics(reference, fit_calls):
    rows, loads = reference
    structures = queries()
    first = NoveltyMetric(fingerprint_method=METHOD).compute(structures)
    fit_calls.clear()
    second = NoveltyMetric(fingerprint_method=METHOD).compute(
        list(reversed(structures))
    )
    assert fit_calls == []
    assert second.individual_values == list(reversed(first.individual_values))
    # SUN and MetaSUN reuse the index and the novelty answers: copies with hull energies differ only in
    # properties, so every comparison they make is between two of their own input structures.
    hulls = [
        with_hull(s, e)
        for s, e in zip(structures, np.linspace(-0.05, 0.12, len(structures)))
    ]
    for metric_class in (SUNMetric, MetaSUNMetric):
        fit_calls.clear()
        metric_class(fingerprint_method=METHOD).compute(hulls)
        inputs = {id(s) for s in hulls}
        assert all(
            id(struct1) in inputs and id(struct2) in inputs
            for struct1, struct2 in fit_calls
        )
    assert len(loads) == 1


def test_novelty_pool_matches_serial(reference, fit_calls):
    rows, _ = reference
    structures = queries()
    result = NoveltyMetric(fingerprint_method=METHOD, n_jobs=2).compute(structures)
    np.testing.assert_array_equal(
        result.individual_values, oracle_novelty(structures, rows)
    )
    fit_calls.clear()
    NoveltyMetric(fingerprint_method=METHOD).compute(structures)
    assert fit_calls == []


def test_index_rebuilt_for_another_reference(reference, monkeypatch):
    rows, _ = reference
    structures = queries()
    NoveltyMetric(fingerprint_method=METHOD).compute(structures)
    other_rows = list(reversed(rows))[:7]
    other = Dataset.from_list(other_rows)
    monkeypatch.setattr(
        reference_index, "load_reference_dataset", lambda name, config: other
    )
    reference_index.get_reference_index.cache_clear()
    result = NoveltyMetric(fingerprint_method=METHOD).compute(structures)
    np.testing.assert_array_equal(
        result.individual_values, oracle_novelty(structures, other_rows)
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_uniqueness_matches_oracle(seed, fit_calls):
    structures = uniqueness_set()
    random.Random(seed).shuffle(structures)
    expected = oracle_unique_count(structures)
    fit_calls.clear()
    result = UniquenessMetric(fingerprint_method=METHOD).compute(structures)
    assert result.metrics["unique_structures_count"] == expected
    assert result.metrics["duplicate_structures_count"] == len(structures) - expected
    assert result.metrics["total_structures_evaluated"] == len(structures)
    for struct1, struct2 in fit_calls:
        k1, k2 = simple_key(struct1), simple_key(struct2)
        assert k1 is None or k2 is None or k1 == k2, (k1, k2)
    assert len(fit_calls) < len(structures) * (len(structures) - 1) // 4


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sun_uniqueness_within_set_matches_oracle(seed):
    structures = uniqueness_set()
    random.Random(seed).shuffle(structures)
    metric = SUNMetric(fingerprint_method=METHOD)
    assert metric._compute_uniqueness_within_set(structures) == oracle_unique_in_set(
        structures
    )


def test_chain_order_rules():
    """Standalone uniqueness keeps only the last of a chain, SUN only the first."""
    chain = [tetragonal_mgo(r) for r in (1.0, 1.1, 1.2)]
    assert (
        UniquenessMetric(fingerprint_method=METHOD)
        .compute(chain)
        .metrics["unique_structures_count"]
        == 1
    )
    assert SUNMetric(fingerprint_method=METHOD)._compute_uniqueness_within_set(
        chain
    ) == [0]


def with_hull(structure, e_above_hull):
    out = structure.copy()
    out.properties["e_above_hull_mean"] = e_above_hull
    return out


def sun_structures():
    nacl_primitive = rocksalt(5.64).get_primitive_structure()
    return [
        with_hull(perturbed(nacl_primitive, 0.02, 0), -0.01),
        with_hull(rocksalt(6.6, "K", "Br"), -0.1),
        with_hull(nacl_primitive, -0.02),
        with_hull(tetragonal_mgo(1.0), 0.05),
        with_hull(tetragonal_mgo(1.1), 0.05),
        with_hull(tetragonal_mgo(1.2), 0.05),
        with_hull(cscl(2.7, "Fe", "O"), 0.08),
        with_hull(cscl(2.75, "Fe", "O"), 0.09),
        with_hull(disordered(0.5), 0.2),
        with_hull(rocksalt(5.64, "Na+", "Cl-"), 0.0),
        with_hull(zincblende(6.0), -0.05),
        with_hull(disordered(0.3), 0.07),
    ]


@pytest.mark.parametrize("metric_class", [SUNMetric, MetaSUNMetric])
def test_sun_matches_oracle(reference, metric_class):
    rows, _ = reference
    structures = sun_structures()
    metric = metric_class(fingerprint_method=METHOD)
    stable_threshold = metric.config.stability_threshold
    meta_threshold = metric.config.metastability_threshold
    hull = [s.properties["e_above_hull_mean"] for s in structures]
    stable = [i for i, e in enumerate(hull) if e <= stable_threshold]
    metastable = [
        i for i, e in enumerate(hull) if stable_threshold < e <= meta_threshold
    ]
    expected = np.zeros(len(structures))
    for subset, value in ((stable, 1.0), (metastable, 0.5)):
        unique = [
            subset[i] for i in oracle_unique_in_set([structures[i] for i in subset])
        ]
        novelty = oracle_novelty([structures[i] for i in unique], rows)
        for index, novel in zip(unique, novelty):
            if novel == 1.0:
                expected[index] = value
    assert expected.max() > 0
    result = metric.compute(structures)
    np.testing.assert_array_equal(result.individual_values, expected)


def test_structures_are_not_modified(reference):
    structures = queries()
    before = [s.as_dict() for s in structures]
    NoveltyMetric(fingerprint_method=METHOD).compute(structures)
    UniquenessMetric(fingerprint_method=METHOD).compute(structures)
    assert [s.as_dict() for s in structures] == before


def test_pairs_cover_every_ordering():
    """The pair rule is direction-aware: plain-then-decorated matches, decorated-then-plain does not."""
    plain, decorated = rocksalt(5.64), rocksalt(5.64, "Na+", "Cl-")
    for order in itertools.permutations([plain, decorated, cscl(3.2)]):
        order = list(order)
        assert UniquenessMetric(fingerprint_method=METHOD).compute(order).metrics[
            "unique_structures_count"
        ] == oracle_unique_count(order)
        assert SUNMetric(fingerprint_method=METHOD)._compute_uniqueness_within_set(
            order
        ) == oracle_unique_in_set(order)

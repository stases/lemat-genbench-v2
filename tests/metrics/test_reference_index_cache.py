"""Small reference-cache and comparison-failure regressions."""

import os
import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from datasets import Dataset, concatenate_datasets
from pymatgen.core import Lattice, Structure

from lemat_genbench.fingerprinting import reference_index


def _row(species, elements):
    return {
        "species_at_sites": species,
        "elements": elements,
        "lattice_vectors": np.eye(3).tolist(),
        "cartesian_site_positions": np.zeros((len(species), 3)).tolist(),
    }


@pytest.fixture
def cached_reference(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    dataset = Dataset.from_list([_row(["Na", "Cl"], ["Na", "Cl"])])
    loader = Mock(return_value=dataset)
    monkeypatch.setattr(reference_index, "load_reference_dataset", loader)
    reference_index.get_reference_index.cache_clear()
    yield loader
    reference_index.get_reference_index.cache_clear()


def test_loader_preserves_column_cache(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(reference_index, "load_dataset", loader)
    reference_index.load_reference_dataset("custom/reference", "config")
    loader.assert_called_once_with(
        "custom/reference",
        "config",
        split="train",
        columns=[
            "elements",
            "immutable_id",
            "chemical_formula_descriptive",
            "energy",
            "species_at_sites",
            "cartesian_site_positions",
            "lattice_vectors",
        ],
    )


def test_disk_cache_and_dataset_pickle(cached_reference, monkeypatch):
    index = reference_index.get_reference_index("test/reference", "config", None)
    restore, arguments = index.__reduce__()
    assert restore is reference_index._index_for
    assert arguments[0] is index.dataset
    assert arguments[1:] == (None, ("test/reference", "config", None))
    payload = pickle.dumps(index)
    assert pickle.loads(payload) is index
    reference_index.get_reference_index.cache_clear()
    cached_reference.side_effect = AssertionError("Restoring must not load a dataset")
    monkeypatch.setattr(
        reference_index.ReferenceIndex, "_build", Mock(side_effect=AssertionError)
    )
    restored = pickle.loads(payload)
    assert restored is not index
    np.testing.assert_array_equal(restored.candidates(index.structure(0)), [0])
    assert restored.novelty_cache == {}


def test_pickle_rebuilds_from_parent_dataset(cached_reference):
    index = reference_index.get_reference_index("test/reference", "config", None)
    payload = pickle.dumps(index)
    reference_index.get_reference_index.cache_clear()
    next(Path("data").glob("reference_index_*.npz")).unlink()
    cached_reference.side_effect = AssertionError("Restoring must not load a dataset")
    restored = pickle.loads(payload)
    assert restored.dataset._fingerprint == index.dataset._fingerprint
    np.testing.assert_array_equal(restored.candidates(index.structure(0)), [0])


def test_dataset_fingerprints_isolate_memoized_answers(cached_reference):
    index = reference_index.get_reference_index("test/reference", "config", None)
    index.novelty_cache["answer"] = 0.0
    other = Dataset.from_list([_row(["K", "Br"], ["K", "Br"])])
    changed = reference_index._index_for(other, None, index._identity)
    assert changed is not index
    assert changed.novelty_cache == {}
    assert reference_index._index_for(index.dataset, None, index._identity) is index


def test_cache_file_respects_umask(cached_reference):
    previous = os.umask(0o002)
    try:
        reference_index.get_reference_index("test/reference", "config", None)
        # Default ACLs can mask the mode; compare with a file created the usual way.
        plain = Path("data") / "plain"
        plain.touch()
    finally:
        os.umask(previous)
    path = next(Path("data").glob("reference_index_*.npz"))
    assert path.stat().st_mode & 0o777 == plain.stat().st_mode & 0o777
    assert not list(Path("data").glob("*.tmp"))


def test_failed_cache_write_keeps_the_index(cached_reference, monkeypatch):
    def fail_write(output, **arrays):
        output.write(b"partial cache")
        raise OSError("cache write failed")

    warnings = []
    monkeypatch.setattr(reference_index.np, "savez", fail_write)
    monkeypatch.setattr(reference_index.logger, "warning", warnings.append)
    index = reference_index.get_reference_index("test/reference", "config", None)
    np.testing.assert_array_equal(index.candidates(index.structure(0)), [0])
    assert len(warnings) == 1 and "cache write failed" in warnings[0]
    assert not list(Path("data").iterdir())


def test_corrupt_cache_file_is_rebuilt(cached_reference):
    index = reference_index.get_reference_index("test/reference", "config", None)
    path = next(Path("data").glob("reference_index_*.npz"))
    path.write_bytes(b"PK\x03\x04 interleaved writers")
    reference_index.get_reference_index.cache_clear()
    rebuilt = reference_index.get_reference_index("test/reference", "config", None)
    np.testing.assert_array_equal(rebuilt.candidates(index.structure(0)), [0])


@pytest.mark.parametrize("field", ["fingerprint", "row_count", "format_version"])
def test_disk_cache_metadata_invalidation(cached_reference, field):
    index = reference_index.get_reference_index("test/reference", "config", None)
    path = next(Path("data").glob("reference_index_*.npz"))
    with np.load(path, allow_pickle=False) as cached:
        arrays = dict(cached)
    arrays[field] = "old" if field == "fingerprint" else -1
    arrays["composition_hashes"] = np.zeros(1, dtype=np.uint64)
    np.savez(path, **arrays)
    reference_index.get_reference_index.cache_clear()
    rebuilt = reference_index.get_reference_index("test/reference", "config", None)
    np.testing.assert_array_equal(rebuilt.candidates(index.structure(0)), [0])


def test_chunked_rows_metadata_and_fallback(monkeypatch):
    original_iter = Dataset.iter

    def small_batches(self, batch_size):
        return original_iter(self, batch_size=2)

    monkeypatch.setattr(Dataset, "iter", small_batches)
    first = Dataset.from_list([_row(["Na", "Cl"], ["Na", "Cl"]), _row(["Na"], [])])
    second = Dataset.from_list([_row(["Na", "Cl"], ["K"]), _row(["Au"], ["Au"])])
    dataset = concatenate_datasets([first, second]).select([3, 0, 2, 1])
    index = reference_index.ReferenceIndex(dataset)
    assert index.disagreement_count == 2
    np.testing.assert_array_equal(index.candidates(index.structure(0)), [0])
    np.testing.assert_array_equal(index.candidates(index.structure(1)), [1])
    disordered = Structure(Lattice.cubic(3), [{"Na": 0.3, "Cl": 0.7}], [[0, 0, 0]])
    np.testing.assert_array_equal(index.candidates(disordered), [1, 3])
    empty = reference_index.ReferenceIndex(dataset, 0)
    assert len(empty.candidates(disordered)) == 0
    assert len(empty.candidates(index.structure(1))) == 0

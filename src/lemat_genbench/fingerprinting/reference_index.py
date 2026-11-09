"""Composition candidates and shared novelty answers for structure matching."""

import hashlib
import json
import os
import re
import socket
import time
import zipfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import lru_cache
from heapq import merge
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, load_dataset
from pymatgen.core import Element, Structure, get_el_sp

from lemat_genbench.fingerprinting.encode_compositions import (
    lematbulk_item_to_structure,
)
from lemat_genbench.utils.logging import logger

_FORMAT_VERSION = 1
_REFERENCE_INDEXES = {}
_COLUMNS = [
    "elements",
    "immutable_id",
    "chemical_formula_descriptive",
    "energy",
    "species_at_sites",
    "cartesian_site_positions",
    "lattice_vectors",
]


def match_key(structure: Structure) -> tuple | None:
    """Return a reduced integer element composition when pruning is safe.

    Parameters
    ----------
    structure : Structure
        Structure being compared, including disordered or decorated species.

    Returns
    -------
    tuple or None
        Sorted symbol/count pairs, or None for ambiguous compositions.
    """
    if len(structure) >= 10_000:
        return None
    amounts = {}
    for species, amount in structure.composition.items():
        symbol = species.symbol
        if symbol in amounts or not np.isfinite(amount):
            return None
        count = round(amount)
        if abs(amount - count) > 1e-6 or count <= 0:
            return None
        amounts[symbol] = count
    if not amounts:
        return None
    divisor = int(np.gcd.reduce(list(amounts.values())))
    return tuple(
        sorted((symbol, count // divisor) for symbol, count in amounts.items())
    )


def load_reference_dataset(reference_dataset: str, reference_config: str) -> Dataset:
    """Load the reference using the existing Hugging Face column cache.

    Parameters
    ----------
    reference_dataset : str
        Hugging Face dataset name.
    reference_config : str
        Dataset configuration.

    Returns
    -------
    Dataset
        Training split with the columns needed to reconstruct structures.
    """
    return load_dataset(
        reference_dataset, reference_config, split="train", columns=_COLUMNS
    )


def _mix(values):
    values = np.asarray(values, dtype=np.uint64)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _key_hash(key):
    codes = np.array([Element(symbol).Z for symbol, _ in key], dtype=np.uint64)
    counts = np.array([count for _, count in key], dtype=np.uint64)
    return _mix(counts * np.uint64(128) + codes).sum(dtype=np.uint64)


def _encoded_column(column):
    column = column.combine_chunks()
    flat = pc.list_flatten(column)
    encoded = pc.dictionary_encode(flat)
    symbols = np.array(
        [get_el_sp(value).Z for value in encoded.dictionary.to_pylist()], dtype=np.int64
    )
    return (
        pc.list_parent_indices(column).to_numpy(),
        symbols[encoded.indices.to_numpy()],
    )


def _element_masks(rows, codes, size):
    masks = np.zeros((size, 2), dtype=np.uint64)
    for half in range(2):
        selected = (codes - 1) // 64 == half
        np.bitwise_or.at(
            masks[:, half],
            rows[selected],
            np.left_shift(np.uint64(1), ((codes[selected] - 1) % 64).astype(np.uint64)),
        )
    return masks


def novelty_cache_key(structure, fingerprinter):
    """Identify structure content and matcher settings for cached novelty.

    Parameters
    ----------
    structure : Structure
        Structure to identify, excluding its properties.
    fingerprinter : PymatgenStructureSimilarity
        Fingerprinter containing the configured StructureMatcher.

    Returns
    -------
    tuple[str, bytes]
        Serialized matcher settings and a digest of the structure content.
    """
    digest = hashlib.sha256()
    digest.update(np.asarray(structure.lattice.matrix, dtype="<f8").tobytes())
    digest.update(np.asarray(structure.frac_coords, dtype="<f8").tobytes())
    species = [
        sorted(
            (json.dumps(sp.as_dict(), sort_keys=True), float(amount))
            for sp, amount in site.species.items()
        )
        for site in structure
    ]
    digest.update(json.dumps(species, separators=(",", ":")).encode())
    settings = json.dumps(fingerprinter.matcher.as_dict(), sort_keys=True)
    return settings, digest.digest()


class CompositionGroups:
    """Group structures for ordered comparisons of compatible compositions.

    Parameters
    ----------
    structures : list[Structure]
        Structures in their original comparison order.
    """

    def __init__(self, structures):
        self.keys = [match_key(structure) for structure in structures]
        self.groups = defaultdict(list)
        for i, key in enumerate(self.keys):
            self.groups[key].append(i)

    def candidates(self, i, earlier=False):
        key = self.keys[i]
        if key is None:
            return iter(range(i) if earlier else range(i + 1, len(self.keys)))
        # SpeciesComparator equality implies equal reduced element counts for keyed pairs.
        groups = (self.groups[key], self.groups[None])
        if earlier:
            return merge(*(islice(group, bisect_left(group, i)) for group in groups))
        return merge(*(islice(group, bisect_right(group, i), None) for group in groups))


class ReferenceIndex:
    """Index reference compositions without constructing reference Structures.

    Parameters
    ----------
    dataset : Dataset
        Reference rows in LeMat-Bulk format.
    max_reference_size : int, optional
        Only the first this many rows participate in matching.
    """

    def __init__(self, dataset: Dataset, max_reference_size: int | None = None):
        self.dataset = dataset
        self.row_count = len(dataset)
        if max_reference_size is not None:
            self.row_count = max(0, min(self.row_count, max_reference_size))
        self.novelty_cache = {}
        self._identity = ("dataset", "default", max_reference_size)
        self._build()

    def _build(self):
        hashes = np.zeros(self.row_count, dtype=np.uint64)
        self.metadata_masks = np.zeros((self.row_count, 2), dtype=np.uint64)
        self.disagreement_count = 0
        dataset = self.dataset.select(range(self.row_count)).select_columns(
            ["species_at_sites", "elements"]
        )
        offset = 0
        for batch in dataset.with_format("arrow").iter(batch_size=65_536):
            size = len(batch)
            rows, codes = _encoded_column(batch["species_at_sites"])
            pairs, counts = np.unique(rows * 128 + codes, return_counts=True)
            pair_rows, pair_codes = pairs // 128, pairs % 128
            if len(pairs):
                starts = np.r_[0, np.flatnonzero(np.diff(pair_rows)) + 1]
                divisors = np.gcd.reduceat(counts, starts)
                reduced = counts // np.repeat(
                    divisors, np.diff(np.r_[starts, len(pairs)])
                )
                hashes[offset + pair_rows[starts]] = np.add.reduceat(
                    _mix(
                        reduced.astype(np.uint64) * np.uint64(128)
                        + pair_codes.astype(np.uint64)
                    ),
                    starts,
                )
            species_masks = _element_masks(pair_rows, pair_codes, size)
            metadata_masks = _element_masks(*_encoded_column(batch["elements"]), size)
            self.metadata_masks[offset : offset + size] = metadata_masks
            self.disagreement_count += int(
                np.count_nonzero(np.any(species_masks != metadata_masks, axis=1))
            )
            offset += size

        self.composition_rows = np.argsort(hashes, kind="stable")
        self.composition_hashes = hashes[self.composition_rows]
        self.element_rows = np.lexsort(
            (self.metadata_masks[:, 1], self.metadata_masks[:, 0])
        )
        masks = self.metadata_masks[self.element_rows]
        if self.row_count:
            starts = np.r_[
                0, np.flatnonzero(np.any(masks[1:] != masks[:-1], axis=1)) + 1
            ]
        else:
            starts = np.array([], dtype=np.int64)
        self.element_masks = masks[starts]
        self.element_offsets = np.r_[starts, self.row_count]

    def candidates(self, structure: Structure) -> np.ndarray:
        """Return eligible row IDs in their original order.

        Parameters
        ----------
        structure : Structure
            First argument of the eventual matcher comparison.

        Returns
        -------
        numpy.ndarray
            Increasing reference row IDs respecting the metadata subset filter.
        """
        codes = np.array([element.Z for element in structure.composition.elements])
        allowed = _element_masks(np.zeros(len(codes), dtype=int), codes, 1)[0]
        key = match_key(structure)
        if key is not None:
            hashed = _key_hash(key)
            start, end = (
                np.searchsorted(self.composition_hashes, hashed, side="left"),
                np.searchsorted(self.composition_hashes, hashed, side="right"),
            )
            rows = self.composition_rows[start:end]
            masks = self.metadata_masks[rows]
            return rows[np.all((masks & allowed) == masks, axis=1)]
        groups = np.flatnonzero(
            np.all((self.element_masks & allowed) == self.element_masks, axis=1)
        )
        if not len(groups):
            return np.array([], dtype=np.int64)
        return np.sort(
            np.concatenate(
                [
                    self.element_rows[
                        self.element_offsets[group] : self.element_offsets[group + 1]
                    ]
                    for group in groups
                ]
            )
        )

    def structure(self, row: int) -> Structure:
        """Reconstruct one reference structure.

        Parameters
        ----------
        row : int
            Reference row ID.

        Returns
        -------
        Structure
            Structure built from the original reference row.
        """
        return lematbulk_item_to_structure(self.dataset[int(row)])

    def __reduce__(self):
        return _index_for, (self.dataset, self._identity[2], self._identity)


@lru_cache(maxsize=None)
def get_reference_index(
    reference_dataset: str, reference_config: str, max_reference_size: int | None
) -> ReferenceIndex:
    """Get a shared index and novelty cache for a reference in this process.

    Parameters
    ----------
    reference_dataset : str
        Hugging Face dataset name.
    reference_config : str
        Dataset configuration.
    max_reference_size : int or None
        Maximum number of reference rows to use.

    Returns
    -------
    ReferenceIndex
        Index shared by all metrics with these reference arguments.
    """
    identity = (reference_dataset, reference_config, max_reference_size)
    dataset = load_reference_dataset(reference_dataset, reference_config)
    return _index_for(dataset, max_reference_size, identity)


def _index_for(dataset, max_reference_size, identity):
    cache_key = (identity, dataset._fingerprint)
    if cache_key in _REFERENCE_INDEXES:
        return _REFERENCE_INDEXES[cache_key]
    start = time.monotonic()
    row_count = (
        len(dataset)
        if max_reference_size is None
        else max(0, min(len(dataset), max_reference_size))
    )
    reference_dataset, reference_config, _ = identity
    name = re.sub(r"[^\w.-]+", "_", f"{reference_dataset}_{reference_config}")
    suffix = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    path = Path("data") / f"reference_index_{name}_{suffix}.npz"
    arrays = (
        "metadata_masks",
        "composition_rows",
        "composition_hashes",
        "element_rows",
        "element_masks",
        "element_offsets",
        "disagreement_count",
    )
    index = None
    try:
        with np.load(path, allow_pickle=False) as cached:
            if (
                cached["format_version"].item() == _FORMAT_VERSION
                and cached["fingerprint"].item() == dataset._fingerprint
                and cached["row_count"].item() == row_count
                and cached["dataset_row_count"].item() == len(dataset)
            ):
                index = ReferenceIndex.__new__(ReferenceIndex)
                for attribute in arrays:
                    setattr(index, attribute, cached[attribute])
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        index = None
    loaded = index is not None
    if index is None:
        index = ReferenceIndex(dataset, max_reference_size)
        # Unique across the nodes that share this directory.
        temporary = path.with_name(
            f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
        )
        try:
            path.parent.mkdir(exist_ok=True)
            with temporary.open("wb") as output:
                np.savez(
                    output,
                    format_version=_FORMAT_VERSION,
                    fingerprint=dataset._fingerprint,
                    row_count=row_count,
                    dataset_row_count=len(dataset),
                    **{attribute: getattr(index, attribute) for attribute in arrays},
                )
            temporary.replace(path)
        except OSError as exc:
            logger.warning(f"Could not write the reference index cache {path}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)
    index.dataset = dataset
    index.row_count = row_count
    index.novelty_cache = {}
    index._identity = identity
    _REFERENCE_INDEXES[cache_key] = index
    logger.info(
        f"{'Loaded' if loaded else 'Built'} reference index for {row_count} rows in "
        f"{time.monotonic() - start:.2f}s; "
        f"{int(index.disagreement_count)} rows have elements/species_at_sites disagreement"
    )
    return index


_clear_reference_loader_cache = get_reference_index.cache_clear


def _clear_reference_caches():
    _clear_reference_loader_cache()
    _REFERENCE_INDEXES.clear()


get_reference_index.cache_clear = _clear_reference_caches

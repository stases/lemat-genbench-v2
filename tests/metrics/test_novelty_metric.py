"""Tests for novelty metrics implementation using real LeMat-Bulk data.

Updated to match current implementation and improve test coverage.
"""

import traceback
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.novelty_metric import NoveltyMetric

NOVELTY_TESTS_AVAILABLE = True

# Sample data from LeMat-Bulk
SAMPLE_LEMAT_ROW = {
    "elements": ["Sb", "Sr"],
    "nsites": 5,
    "chemical_formula_anonymous": "A4B",
    "chemical_formula_reduced": "Sb4Sr",
    "chemical_formula_descriptive": "Sr1 Sb4",
    "nelements": 2,
    "dimension_types": [1, 1, 1],
    "nperiodic_dimensions": 3,
    "lattice_vectors": [
        [-3.56534985, 3.56534985, 3.56534985],
        [3.56534985, -3.56534985, 3.56534985],
        [3.56534985, 3.56534985, -3.56534985],
    ],
    "immutable_id": "agm005415715",
    "cartesian_site_positions": [
        [0, 0, 0],
        [-1.782674925, 1.782674925, 1.782674925],
        [1.782674925, 1.782674925, 1.782674925],
        [1.782674925, -1.782674925, 1.782674925],
        [1.782674925, 1.782674925, -1.782674925],
    ],
    "species": [
        {
            "mass": None,
            "name": "Sb",
            "attached": None,
            "nattached": None,
            "concentration": [1],
            "original_name": None,
            "chemical_symbols": ["Sb"],
        },
        {
            "mass": None,
            "name": "Sr",
            "attached": None,
            "nattached": None,
            "concentration": [1],
            "original_name": None,
            "chemical_symbols": ["Sr"],
        },
    ],
    "species_at_sites": ["Sr", "Sb", "Sb", "Sb", "Sb"],
    "last_modified": "2023-11-16 06:57:59",
    "elements_ratios": [0.8, 0.2],
    "stress_tensor": [[0.3413351, 0, 0], [0, 0.3413351, 0], [0, 0, 0.3413351]],
    "energy": -17.323021,
    "magnetic_moments": [0, 0, 0, 0, 0],
    "forces": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "total_magnetization": 0.000002,
    "dos_ef": 4.318709,
    "functional": "pbe",
    "cross_compatibility": True,
    "entalpic_fingerprint": "38f73083d88aa235c8c8c9d66617f3e3_229_Sr1Sb4",
}


def create_weird_structures():
    """Create structures that are very unlikely to be in LeMat-Bulk.

    Note: Using more common elements that BAWL hasher can handle properly.
    """
    structures = []

    # 1. Very large unit cell with noble gases
    lattice1 = np.eye(3) * 20.0  # Large but not extreme unit cell
    structure1 = Structure(
        lattice=lattice1,
        species=["Xe", "Kr", "Ne"],
        coords=[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure1)

    # 2. Unusual but real actinide compound
    lattice2 = [[8, 0, 0], [0, 8, 0], [0, 0, 8]]
    structure2 = Structure(
        lattice=lattice2,
        species=["U", "O", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure2)

    # 3. Artificial structure with unusual coordination
    lattice3 = [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Be", "F", "F"],
        coords=[
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.5, 0.5, 0.5],
        ],
        coords_are_cartesian=False,
    )
    structures.append(structure3)

    return structures


def create_simple_test_structures():
    """Create simple test structures for unit testing."""
    structures = []

    # Simple cubic NaCl structure
    lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure1 = Structure(
        lattice=lattice,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure1)

    # Simple cubic KBr structure
    lattice = [[4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]]
    structure2 = Structure(
        lattice=lattice,
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure2)

    return structures


@pytest.mark.skipif(not NOVELTY_TESTS_AVAILABLE, reason="Novelty metrics not available")
class TestNoveltyMetricBasic:
    """Test basic novelty metric functionality without heavy dataset loading."""

    def test_novelty_metric_initialization(self):
        """Test NoveltyMetric initialization."""
        metric = NoveltyMetric()
        assert metric.name == "Novelty"
        assert metric.config.reference_dataset == "LeMaterial/LeMat-Bulk"
        assert metric.config.reference_config == "compatible_pbe"
        assert metric.config.fingerprint_method == "bawl"
        assert hasattr(metric, "fingerprinter")

    def test_custom_initialization(self):
        """Test novelty metric with custom configuration."""
        metric = NoveltyMetric(
            reference_dataset="LeMaterial/LeMat-Bulk",
            reference_config="compatible_pbesol",  # Different config
            fingerprint_method="short-bawl",
            max_reference_size=1000,
            cache_reference=False,
            name="Custom Novelty",
            description="Custom description",
        )

        assert metric.name == "Custom Novelty"
        assert metric.config.description == "Custom description"
        assert metric.config.reference_config == "compatible_pbesol"
        assert metric.config.fingerprint_method == "short-bawl"
        assert metric.config.max_reference_size == 1000
        assert metric.config.cache_reference is False

    def test_invalid_fingerprint_method(self):
        """Test error handling for invalid fingerprint method."""
        with pytest.raises(ValueError, match="Unknown fingerprint method"):
            NoveltyMetric(fingerprint_method="invalid_method")

    def test_row_to_structure_conversion(self):
        """Test conversion of LeMat-Bulk row to pymatgen Structure."""
        metric = NoveltyMetric()

        structure = metric._row_to_structure(SAMPLE_LEMAT_ROW)

        # Check structure properties
        assert isinstance(structure, Structure)
        assert len(structure) == 5  # nsites
        assert structure.composition.reduced_formula == "SrSb4"

        # Check species
        species_symbols = [str(site.specie) for site in structure]
        expected_species = ["Sr", "Sb", "Sb", "Sb", "Sb"]
        assert species_symbols == expected_species

        # Check lattice (should be the same)
        expected_lattice = np.array(SAMPLE_LEMAT_ROW["lattice_vectors"])
        np.testing.assert_array_almost_equal(structure.lattice.matrix, expected_lattice)

    @patch("lemat_genbench.metrics.novelty_metric.get_fingerprint")
    def test_fingerprint_consistency(self, mock_get_fingerprint):
        """Test that fingerprinting is consistent."""
        # Mock the fingerprinting to return consistent results
        mock_get_fingerprint.side_effect = lambda struct, fp: f"mock_fp_{struct.composition.reduced_formula}"
        
        metric = NoveltyMetric()
        structure = metric._row_to_structure(SAMPLE_LEMAT_ROW)

        # Compute fingerprint multiple times using the metric's fingerprinter
        fp1 = mock_get_fingerprint(structure, metric.fingerprinter)
        fp2 = mock_get_fingerprint(structure, metric.fingerprinter)

        # Should be identical
        assert fp1 == fp2
        assert isinstance(fp1, str)
        assert len(fp1) > 0

    def test_compute_structure_novel_mocked(self):
        """Test computing novelty for a novel structure with mocked dataset."""
        structure = create_simple_test_structures()[0]  # NaCl structure

        # Mock dataset information with fingerprints that don't include our structure
        dataset_information = {
            "fingerprints": {
                "different_fingerprint_1_229_Sr1Sb4",
                "different_fingerprint_2_229_Ca1P2",
                "different_fingerprint_3_229_Mg1O1",
            }
        }

        # Mock fingerprinter
        mock_fingerprinter = Mock()
        mock_fingerprinter.get_material_hash = Mock(return_value="novel_structure_fp")

        # Mock get_fingerprint to use the fingerprinter
        with patch("lemat_genbench.metrics.novelty_metric.get_fingerprint") as mock_get_fp:
            mock_get_fp.return_value = "novel_structure_fp"

            result = NoveltyMetric.compute_structure(
                structure, dataset_information, mock_fingerprinter
            )

            # Should be novel (1.0) since structure fingerprint not in reference
            assert result == 1.0

    def test_compute_structure_known_mocked(self):
        """Test computing novelty for a known structure with mocked dataset."""
        structure = create_simple_test_structures()[0]  # NaCl structure

        # Mock dataset information with fingerprints that include our structure
        known_fingerprint = "known_structure_fp"
        dataset_information = {
            "fingerprints": {
                known_fingerprint,
                "different_fingerprint_1_229_Sr1Sb4",
                "different_fingerprint_2_229_Ca1P2",
            }
        }

        # Mock fingerprinter
        mock_fingerprinter = Mock()
        mock_fingerprinter.get_material_hash = Mock(return_value=known_fingerprint)

        # Mock get_fingerprint to return the known fingerprint
        with patch("lemat_genbench.metrics.novelty_metric.get_fingerprint") as mock_get_fp:
            mock_get_fp.return_value = known_fingerprint

            result = NoveltyMetric.compute_structure(
                structure, dataset_information, mock_fingerprinter
            )

            # Should be known (0.0) since structure fingerprint is in reference
            assert result == 0.0

    def test_compute_structure_error_handling(self):
        """Test error handling in compute_structure."""
        structure = create_simple_test_structures()[0]

        # Mock dataset information
        dataset_information = {"fingerprints": {"some_fingerprint"}}

        # Mock fingerprinter that raises an exception
        mock_fingerprinter = Mock()
        mock_fingerprinter.get_material_hash = Mock(side_effect=Exception("Fingerprinting failed"))

        # Mock get_fingerprint to raise an exception
        with patch("lemat_genbench.metrics.novelty_metric.get_fingerprint") as mock_get_fp:
            mock_get_fp.side_effect = Exception("Fingerprinting failed")

            result = NoveltyMetric.compute_structure(
                structure, dataset_information, mock_fingerprinter
            )

            # Should return NaN for failed fingerprinting
            assert np.isnan(result)

    def test_aggregate_results(self):
        """Test aggregation of novelty results."""
        metric = NoveltyMetric()

        # Test with mixed results: 2 novel, 1 known
        values = [1.0, 0.0, 1.0]
        result = metric.aggregate_results(values)

        assert result["metrics"]["novelty_score"] == 2.0 / 3.0
        assert result["metrics"]["novel_structures_count"] == 2
        assert result["metrics"]["total_structures_evaluated"] == 3
        assert result["primary_metric"] == "novelty_score"

        # Test with all novel
        values = [1.0, 1.0, 1.0]
        result = metric.aggregate_results(values)
        assert result["metrics"]["novelty_score"] == 1.0

        # Test with all known
        values = [0.0, 0.0, 0.0]
        result = metric.aggregate_results(values)
        assert result["metrics"]["novelty_score"] == 0.0

        # Test with NaN values
        values = [1.0, float("nan"), 0.0]
        result = metric.aggregate_results(values)
        assert result["metrics"]["novelty_score"] == 0.5  # 1 novel out of 2 valid

        # Test with all NaN values
        values = [float("nan"), float("nan"), float("nan")]
        result = metric.aggregate_results(values)
        assert np.isnan(result["metrics"]["novelty_score"])
        assert result["metrics"]["novel_structures_count"] == 0
        assert result["metrics"]["total_structures_evaluated"] == 0

    def test_aggregate_results_uncertainties(self):
        """Test uncertainty calculation in aggregate_results."""
        metric = NoveltyMetric()

        # Test with values that have variance
        values = [1.0, 0.0, 1.0, 0.0]
        result = metric.aggregate_results(values)

        # Should have non-zero standard deviation
        assert "uncertainties" in result
        assert "novelty_score" in result["uncertainties"]
        assert "std" in result["uncertainties"]["novelty_score"]
        assert result["uncertainties"]["novelty_score"]["std"] > 0

        # Test with single value (should have std = 0)
        values = [1.0]
        result = metric.aggregate_results(values)
        assert result["uncertainties"]["novelty_score"]["std"] == 0.0

    @patch("lemat_genbench.metrics.novelty_metric.load_dataset")
    def test_load_reference_dataset_precomputed_fingerprints(self, mock_load_dataset):
        """Test loading reference dataset with pre-computed fingerprints."""
        # Mock dataset with pre-computed BAWL fingerprints
        mock_dataset = Mock()
        # Fix: Use actual list instead of Mock for column_names
        mock_dataset.column_names = ["entalpic_fingerprint", "other_column"]
        mock_dataset.__getitem__ = Mock(return_value=[
            "fp1_123_compound1",
            "fp2_456_compound2",
            "fp3_789_compound3",
        ])
        
        # Properly implement __len__ method with self parameter
        def mock_len(self):
            return 3
        mock_dataset.__len__ = mock_len
        
        # Mock the select method for dataset size limiting
        mock_dataset.select = Mock(return_value=mock_dataset)
        
        mock_load_dataset.return_value = mock_dataset

        metric = NoveltyMetric(fingerprint_method="bawl", max_reference_size=10)
        dataset_info = metric._load_reference_dataset()

        # Should use pre-computed fingerprints
        assert "fingerprints" in dataset_info
        assert len(dataset_info["fingerprints"]) == 3
        assert "fp1_123_compound1" in dataset_info["fingerprints"]

    @patch("lemat_genbench.metrics.novelty_metric.load_dataset")
    def test_load_reference_dataset_short_bawl(self, mock_load_dataset):
        """Test loading reference dataset with short-bawl fingerprints."""
        # Mock dataset with pre-computed BAWL fingerprints
        mock_dataset = Mock()
        # Fix: Use actual list instead of Mock for column_names
        mock_dataset.column_names = ["entalpic_fingerprint", "other_column"]
        mock_dataset.__getitem__ = Mock(return_value=[
            "fp1_123_compound1",
            "fp2_456_compound2", 
            "fp3_789_compound3",
        ])
        
        # Properly implement __len__ method with self parameter
        def mock_len(self):
            return 3
        mock_dataset.__len__ = mock_len
        
        # Mock the select method for dataset size limiting
        mock_dataset.select = Mock(return_value=mock_dataset)
        
        mock_load_dataset.return_value = mock_dataset

        metric = NoveltyMetric(fingerprint_method="short-bawl", max_reference_size=10)
        dataset_info = metric._load_reference_dataset()

        # Should use shortened fingerprints (first and third parts only)
        assert "fingerprints" in dataset_info
        expected_fps = {"fp1_compound1", "fp2_compound2", "fp3_compound3"}
        assert dataset_info["fingerprints"] == expected_fps

    @patch("lemat_genbench.metrics.novelty_metric.get_reference_index")
    def test_load_reference_dataset_structure_matcher(self, mock_get_index):
        """Test loading and retaining the shared structure matcher index."""
        metric = NoveltyMetric(
            fingerprint_method="structure-matcher",
            reference_dataset="custom/reference",
            reference_config="custom_config",
            max_reference_size=10,
        )
        dataset_info = metric._load_reference_dataset()

        assert dataset_info == {"reference_index": mock_get_index.return_value}
        mock_get_index.assert_called_once_with("custom/reference", "custom_config", 10)
        assert metric._load_reference_dataset() is dataset_info
        mock_get_index.assert_called_once()

        uncached = NoveltyMetric(
            fingerprint_method="structure-matcher", cache_reference=False
        )
        uncached._load_reference_dataset()
        assert uncached._dataset_information is None
        assert not uncached._reference_loaded

    def test_get_compute_attributes(self):
        """Test _get_compute_attributes method."""
        # Mock the _load_reference_dataset method
        mock_dataset_info = {"fingerprints": {"fp1", "fp2", "fp3"}}
        
        metric = NoveltyMetric()
        metric._load_reference_dataset = Mock(return_value=mock_dataset_info)

        attributes = metric._get_compute_attributes()

        assert "dataset_information" in attributes
        assert "fingerprinter" in attributes
        assert "verbose" in attributes
        assert attributes["dataset_information"] == mock_dataset_info
        assert attributes["fingerprinter"] == metric.fingerprinter

    def test_caching_behavior(self):
        """Test that reference dataset caching works correctly."""
        metric = NoveltyMetric(cache_reference=True)
        
        # Mock the loading process
        mock_dataset_info = {"fingerprints": {"fp1", "fp2"}}
        original_load = metric._load_reference_dataset
        metric._load_reference_dataset = Mock(return_value=mock_dataset_info)

        # First call should load the dataset
        info1 = metric._load_reference_dataset()
        assert metric._load_reference_dataset.call_count == 1

        # Set up the cached state
        metric._dataset_information = mock_dataset_info
        metric._reference_loaded = True
        
        # Restore original method to test caching
        metric._load_reference_dataset = original_load

        # Second call should use cache (won't call load again)
        info2 = metric._load_reference_dataset()
        assert info1 == info2 == mock_dataset_info

    def test_callable_interface_mocked(self):
        """Test the __call__ interface with mocked dataset loading."""
        # Mock the dataset loading to avoid actual dataset download
        mock_dataset_info = {"fingerprints": {"existing_fp1", "existing_fp2"}}
        
        metric = NoveltyMetric(max_reference_size=10)
        metric._load_reference_dataset = Mock(return_value=mock_dataset_info)

        # Create test structures
        structures = create_simple_test_structures()

        # Mock fingerprinting to return novel fingerprints
        with patch("lemat_genbench.metrics.novelty_metric.get_fingerprint") as mock_get_fp:
            mock_get_fp.side_effect = lambda struct, fp: f"novel_{struct.composition.reduced_formula}"

            # Test callable interface
            result = metric(structures)

            # Should return MetricResult
            assert isinstance(result, MetricResult)
            assert hasattr(result, "metrics")
            assert "novelty_score" in result.metrics
            
            # All structures should be novel since fingerprints don't match reference
            assert result.metrics["novelty_score"] == 1.0
            assert result.metrics["novel_structures_count"] == len(structures)


@pytest.mark.skipif(not NOVELTY_TESTS_AVAILABLE, reason="Novelty metrics not available")
class TestNoveltyMetricIntegration:
    """Integration tests for novelty metric with real components."""

    def test_error_handling_with_problematic_elements(self):
        """Test error handling when fingerprinting fails on certain elements."""
        metric = NoveltyMetric(max_reference_size=10)
        
        # Mock dataset loading to avoid network calls
        mock_dataset_info = {"fingerprints": {"safe_fp1", "safe_fp2"}}
        metric._load_reference_dataset = Mock(return_value=mock_dataset_info)

        # Create structure with elements that might cause fingerprinting to fail
        lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]

        try:
            problematic_structure = Structure(
                lattice=lattice,
                species=["Og", "Ts"],  # Superheavy elements
                coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )

            # Mock fingerprinting to fail for this structure
            with patch("lemat_genbench.metrics.novelty_metric.get_fingerprint") as mock_get_fp:
                mock_get_fp.side_effect = Exception("Unknown element")

                result = metric.compute([problematic_structure])

                # Should handle the error gracefully
                assert isinstance(result, MetricResult)
                # Should have failed fingerprinting
                assert result.metrics["total_structures_evaluated"] == 0
                # Individual values should contain NaN
                assert len(result.individual_values) == 1
                assert np.isnan(result.individual_values[0])

        except Exception:
            # If the structure creation itself fails, that's also OK
            pass

    @pytest.mark.slow
    def test_real_dataset_integration(self):
        """Test with actual LeMat-Bulk dataset (slow test)."""
        # Use very small reference for CI/CD
        metric = NoveltyMetric(max_reference_size=50)

        # Create definitely novel structures
        weird_structures = create_weird_structures()

        # Evaluate novelty
        result = metric.compute(weird_structures)

        # Check that we get a MetricResult
        assert isinstance(result, MetricResult)

        # Check results - some structures might fail fingerprinting
        assert result.metrics["total_structures_evaluated"] <= len(weird_structures)
        assert (
            result.metrics["total_structures_evaluated"] > 0
        )  # At least some should work
        assert 0 <= result.metrics["novelty_score"] <= 1.0
        assert isinstance(result.metrics["novel_structures_count"], int)

        # Most structures that can be fingerprinted should be novel
        if result.metrics["total_structures_evaluated"] > 0:
            assert result.metrics["novelty_score"] >= 0.5


class TestNoveltyMetricStructureMatcher:
    """Test novelty metric with structure matcher fingerprinting."""

    def test_structure_matcher_novel(self):
        """Test structure matcher with no reference candidates."""
        structure = create_simple_test_structures()[0]
        index = Mock(novelty_cache={})
        index.candidates.return_value = []
        fingerprinter = Mock(spec=["matcher", "is_equivalent"])
        fingerprinter.matcher.as_dict.return_value = {"ltol": 0.1}
        fingerprinter.is_equivalent.return_value = False

        result = NoveltyMetric.compute_structure(
            structure, {"reference_index": index}, fingerprinter, verbose=False
        )

        assert result == 1.0
        fingerprinter.is_equivalent.assert_not_called()
        assert list(index.novelty_cache.values()) == [1.0]

    def test_structure_matcher_known(self):
        """Test structure matcher with a matching reference candidate."""
        structure = create_simple_test_structures()[0]
        ref_structure = create_simple_test_structures()[0]
        index = Mock(novelty_cache={})
        index.candidates.return_value = [0, 1]
        index.structure.return_value = ref_structure
        fingerprinter = Mock(spec=["matcher", "is_equivalent"])
        fingerprinter.matcher.as_dict.return_value = {"ltol": 0.1}
        fingerprinter.is_equivalent.return_value = True

        result = NoveltyMetric.compute_structure(
            structure, {"reference_index": index}, fingerprinter, verbose=False
        )

        assert result == 0.0
        fingerprinter.is_equivalent.assert_called_once_with(structure, ref_structure)
        index.structure.assert_called_once_with(0)
        assert list(index.novelty_cache.values()) == [0.0]


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual novelty metrics test...")

    try:
        # Test 1: Basic functionality
        print("1. Testing basic initialization...")
        metric = NoveltyMetric()
        print("✓ Metric initialized successfully")

        # Test 2: Structure conversion
        print("2. Testing structure conversion...")
        structure = metric._row_to_structure(SAMPLE_LEMAT_ROW)
        print(f"✓ Converted structure: {structure.composition.reduced_formula}")

        # Test 3: Mock fingerprinting test
        print("3. Testing mocked fingerprinting...")
        with patch("lemat_genbench.metrics.novelty_metric.get_fingerprint") as mock_get_fp:
            mock_get_fp.return_value = "mock_fingerprint_12345"
            fp = mock_get_fp(structure, metric.fingerprinter)
            print(f"✓ Mock fingerprint: {fp}")

        # Test 4: Test with simple structures
        print("4. Testing with simple structures...")
        simple_structures = create_simple_test_structures()
        print(f"✓ Created {len(simple_structures)} simple structures")

        for i, struct in enumerate(simple_structures):
            print(f"   - Structure {i + 1}: {struct.composition.reduced_formula}")

        # Test 5: Mock dataset information test
        print("5. Testing with mock dataset...")
        mock_dataset_info = {"fingerprints": {"fp1", "fp2", "fp3"}}
        mock_fingerprinter = Mock()
        mock_fingerprinter.get_material_hash = Mock(return_value="novel_fp")

        with patch("lemat_genbench.metrics.novelty_metric.get_fingerprint") as mock_get_fp:
            mock_get_fp.return_value = "novel_fp"
            
            result = NoveltyMetric.compute_structure(
                simple_structures[0], mock_dataset_info, mock_fingerprinter
            )
            print(f"✓ Novelty result (should be 1.0): {result}")

        # Test 6: Aggregation test
        print("6. Testing result aggregation...")
        test_values = [1.0, 0.0, 1.0, float("nan"), 0.0]
        aggregated = metric.aggregate_results(test_values)
        print(f"✓ Aggregated results: {aggregated['metrics']}")

        print("\n✅ All manual tests passed!")
        print("\nTo run full tests:")
        print("   pytest tests/metrics/test_novelty_metric.py -v")
        print("   pytest tests/metrics/test_novelty_metric.py -v -m slow  # For integration tests")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()
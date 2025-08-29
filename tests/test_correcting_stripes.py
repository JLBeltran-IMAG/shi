import numpy as np
import pytest
import sys
import tempfile
import tifffile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import correcting_stripes


class TestDetectorStripeCorrection:
    """Test suite for detector stripe correction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image_size = (100, 120)
        self.test_image = self.create_test_image_with_stripes()
        self.known_stripe_rows = [10, 25, 50]
        self.known_stripe_cols = [15, 30, 45, 80]

    def create_test_image_with_stripes(self):
        """Create a synthetic test image with simulated stripe artifacts."""
        # Create base image with random values
        image = np.random.randint(100, 200, self.test_image_size, dtype=np.uint16)
        
        # Add stripe artifacts (abnormal intensity rows and columns)
        stripe_rows = [10, 25, 50]
        stripe_cols = [15, 30, 45, 80]
        
        # Make stripe rows have abnormally high intensity
        for row in stripe_rows:
            if row < image.shape[0]:
                image[row, :] = np.random.randint(250, 300, image.shape[1])
        
        # Make stripe columns have abnormally low intensity  
        for col in stripe_cols:
            if col < image.shape[1]:
                image[:, col] = np.random.randint(10, 50, image.shape[0])
                
        return image

    def create_detector_image(self, shape=(3000, 2500)):
        """Create a realistic detector-sized image."""
        return np.random.randint(50, 200, shape, dtype=np.uint16)

    def test_delete_detector_stripes_basic_functionality(self):
        """Test basic stripe deletion functionality."""
        original_shape = self.test_image.shape
        stripe_rows = [10, 25]
        stripe_cols = [15, 30]
        
        cleaned_image = correcting_stripes.delete_detector_stripes(
            self.test_image, stripe_rows, stripe_cols
        )
        
        # Check dimensions are reduced correctly
        expected_height = original_shape[0] - len(stripe_rows)
        expected_width = original_shape[1] - len(stripe_cols)
        
        assert cleaned_image.shape == (expected_height, expected_width), \
            f"Expected shape {(expected_height, expected_width)}, got {cleaned_image.shape}"

    def test_delete_detector_stripes_empty_lists(self):
        """Test stripe deletion with empty stripe lists."""
        original_image = self.test_image.copy()
        
        cleaned_image = correcting_stripes.delete_detector_stripes(
            original_image, [], []
        )
        
        # Should return identical image
        np.testing.assert_array_equal(cleaned_image, original_image)
        assert cleaned_image.shape == original_image.shape

    def test_delete_detector_stripes_boundary_indices(self):
        """Test stripe deletion with boundary indices."""
        image = np.random.randint(0, 255, (50, 60), dtype=np.uint16)
        
        # Test with first and last rows/columns
        stripe_rows = [0, 49]  # First and last row
        stripe_cols = [0, 59]  # First and last column
        
        cleaned_image = correcting_stripes.delete_detector_stripes(
            image, stripe_rows, stripe_cols
        )
        
        expected_shape = (48, 58)  # 50-2, 60-2
        assert cleaned_image.shape == expected_shape

    def test_delete_detector_stripes_out_of_bounds(self):
        """Test stripe deletion with out-of-bounds indices."""
        image = np.random.randint(0, 255, (10, 15), dtype=np.uint16)
        
        # Test that current implementation fails gracefully with out-of-bounds indices
        stripe_rows = [2, 5, 100]  # 100 is out of bounds
        stripe_cols = [3, 7, 200]  # 200 is out of bounds
        
        # The current implementation doesn't handle out-of-bounds gracefully
        # This test documents the current behavior (raising IndexError)
        with pytest.raises(IndexError):
            correcting_stripes.delete_detector_stripes(
                image, stripe_rows, stripe_cols
            )
        
        # Test with only valid indices to ensure normal operation works
        valid_stripe_rows = [2, 5]
        valid_stripe_cols = [3, 7]
        
        cleaned_image = correcting_stripes.delete_detector_stripes(
            image, valid_stripe_rows, valid_stripe_cols
        )
        
        assert cleaned_image.shape[0] == image.shape[0] - 2  # 2 rows removed
        assert cleaned_image.shape[1] == image.shape[1] - 2  # 2 cols removed

    def test_delete_detector_stripes_duplicate_indices(self):
        """Test stripe deletion with duplicate indices."""
        image = np.random.randint(0, 255, (20, 25), dtype=np.uint16)
        
        # Include duplicate indices
        stripe_rows = [5, 10, 5, 15]  # 5 is duplicated
        stripe_cols = [8, 12, 8, 18]  # 8 is duplicated
        
        cleaned_image = correcting_stripes.delete_detector_stripes(
            image, stripe_rows, stripe_cols
        )
        
        # Should handle duplicates correctly (numpy.delete removes each occurrence)
        expected_rows_removed = len(set(stripe_rows))  # Unique rows
        expected_cols_removed = len(set(stripe_cols))  # Unique columns
        
        assert cleaned_image.shape[0] == image.shape[0] - expected_rows_removed
        assert cleaned_image.shape[1] == image.shape[1] - expected_cols_removed

    def test_delete_detector_stripes_data_type_preservation(self):
        """Test that data type is preserved during stripe removal."""
        # Test with different data types
        for dtype in [np.uint8, np.uint16, np.int32, np.float32, np.float64]:
            image = np.random.randint(0, 100, (30, 40)).astype(dtype)
            
            cleaned_image = correcting_stripes.delete_detector_stripes(
                image, [5, 10], [8, 15]
            )
            
            assert cleaned_image.dtype == dtype, f"Data type not preserved for {dtype}"

    def test_delete_detector_stripes_known_detector_indices(self):
        """Test with the actual detector stripe indices used in production."""
        # Create detector-sized image
        detector_image = self.create_detector_image()
        
        # Use the actual stripe indices from the code
        stripe_rows = [2944, 2945]
        stripe_cols = [295, 722, 1167, 1388, 1541, 2062, 2302, 2303]
        
        cleaned_image = correcting_stripes.delete_detector_stripes(
            detector_image, stripe_rows, stripe_cols
        )
        
        # Check correct dimensions
        expected_height = detector_image.shape[0] - len(stripe_rows)
        expected_width = detector_image.shape[1] - len(stripe_cols)
        
        assert cleaned_image.shape == (expected_height, expected_width)

    def test_delete_detector_stripes_content_verification(self):
        """Test that correct rows and columns are actually removed."""
        # Create a simple test image where we can verify content
        image = np.arange(60).reshape(6, 10)  # 6x10 image with values 0-59
        
        stripe_rows = [1, 4]  # Remove rows 1 and 4
        stripe_cols = [2, 7]   # Remove columns 2 and 7
        
        cleaned_image = correcting_stripes.delete_detector_stripes(
            image, stripe_rows, stripe_cols
        )
        
        # Expected result: should be missing original rows 1,4 and cols 2,7
        assert cleaned_image.shape == (4, 8)  # 6-2 rows, 10-2 cols
        
        # First row of cleaned image should be original row 0
        expected_first_row = np.delete(image[0, :], [2, 7])  # Remove cols 2,7 from row 0
        np.testing.assert_array_equal(cleaned_image[0, :], expected_first_row)

    @patch('correcting_stripes.QApplication')
    @patch('correcting_stripes.QFileDialog.getExistingDirectory')
    def test_correcting_stripes_folder_selection_dialog(self, mock_dialog, mock_app):
        """Test folder selection dialog functionality."""
        # Mock QApplication properly - it should return an instance, not a list
        mock_app_instance = MagicMock()
        mock_app.return_value = mock_app_instance
        
        # Create temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_tiff_path = temp_path / "test.tif"
            
            # Create a test TIFF file
            test_image = np.random.randint(0, 255, (100, 120), dtype=np.uint16)
            tifffile.imwrite(test_tiff_path, test_image)
            
            # Mock the dialog to return our temp directory
            mock_dialog.return_value = str(temp_path)
            
            # Mock tifffile functions to avoid actual file operations
            with patch('correcting_stripes.tifffile.imread') as mock_imread, \
                 patch('correcting_stripes.tifffile.imwrite') as mock_imwrite, \
                 patch('correcting_stripes.Path.rglob') as mock_rglob:
                
                mock_imread.return_value = test_image
                mock_rglob.return_value = [test_tiff_path]
                
                # Test the function
                correcting_stripes.correcting_stripes(folder=None)
                
                # Verify dialog was called
                mock_dialog.assert_called_once()
                mock_app.assert_called_once()
                mock_app_instance.quit.assert_called_once()

    def test_correcting_stripes_with_provided_folder(self):
        """Test correcting_stripes with explicitly provided folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create subdirectory structure
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            
            # Create test TIFF files
            test_images = []
            test_files = []
            
            for i, directory in enumerate([temp_path, sub_dir]):
                test_image = np.random.randint(0, 255, (3000, 2500), dtype=np.uint16)
                test_file = directory / f"test_{i}.tif"
                tifffile.imwrite(test_file, test_image)
                test_images.append(test_image)
                test_files.append(test_file)
            
            # Mock file operations to avoid actual modification
            with patch('correcting_stripes.tifffile.imwrite') as mock_imwrite:
                correcting_stripes.correcting_stripes(folder=str(temp_path))
                
                # Should have processed both files
                assert mock_imwrite.call_count == len(test_files)

    def test_correcting_stripes_error_handling(self):
        """Test error handling in correcting_stripes function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a file that will cause imread to fail
            corrupt_file = temp_path / "corrupt.tif"
            corrupt_file.write_text("not a valid tiff")
            
            # Mock tifffile.imread to raise an exception
            with patch('correcting_stripes.tifffile.imread') as mock_imread:
                mock_imread.side_effect = Exception("Corrupted file")
                
                # Should handle the exception gracefully
                try:
                    correcting_stripes.correcting_stripes(folder=str(temp_path))
                except Exception:
                    pytest.fail("correcting_stripes should handle file read errors gracefully")

    def test_correcting_stripes_no_tiff_files(self):
        """Test correcting_stripes behavior with no TIFF files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create non-TIFF files
            (temp_path / "test.txt").write_text("not a tiff")
            (temp_path / "image.jpg").write_text("also not a tiff")
            
            # Should complete without errors
            correcting_stripes.correcting_stripes(folder=str(temp_path))

    def test_correcting_stripes_cancelled_dialog(self):
        """Test behavior when user cancels the folder selection dialog."""
        with patch('correcting_stripes.QApplication'), \
             patch('correcting_stripes.QFileDialog.getExistingDirectory') as mock_dialog:
            
            # Mock dialog returning empty string (user cancelled)
            mock_dialog.return_value = ""
            
            # Should return early without processing
            result = correcting_stripes.correcting_stripes(folder=None)
            
            # Function should complete without errors
            assert result is None

    def test_stripe_removal_preserves_image_statistics(self):
        """Test that stripe removal doesn't dramatically change image statistics."""
        # Create image with known statistics
        base_intensity = 128
        image = np.full((100, 120), base_intensity, dtype=np.uint16)
        
        # Add moderate noise
        noise = np.random.normal(0, 10, image.shape)
        image = (image + noise).astype(np.uint16)
        
        # Add a few stripe artifacts
        stripe_rows = [25, 75]
        stripe_cols = [30, 90]
        
        # Make stripes have different intensity
        for row in stripe_rows:
            image[row, :] += 50
        for col in stripe_cols:
            image[:, col] -= 30
        
        # Remove stripes
        cleaned_image = correcting_stripes.delete_detector_stripes(
            image, stripe_rows, stripe_cols
        )
        
        # Original image stats (excluding stripe regions)
        mask = np.ones(image.shape, dtype=bool)
        mask[stripe_rows, :] = False
        mask[:, stripe_cols] = False
        clean_region_original = image[mask]
        
        # Compare statistics
        original_mean = np.mean(clean_region_original)
        cleaned_mean = np.mean(cleaned_image)
        
        # Should be similar (within 5% tolerance)
        relative_error = abs(cleaned_mean - original_mean) / (original_mean + 1e-10)
        assert relative_error < 0.05, f"Mean changed too much: {relative_error:.3f}"


class TestStripeDetectionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_pixel_image(self):
        """Test with single pixel image."""
        image = np.array([[255]], dtype=np.uint16)
        
        # Try to remove non-existent stripes
        cleaned = correcting_stripes.delete_detector_stripes(image, [0], [0])
        
        # Should result in empty array
        assert cleaned.size == 0

    def test_single_row_image(self):
        """Test with single row image."""
        image = np.random.randint(0, 255, (1, 100), dtype=np.uint16)
        
        # Remove some columns
        cleaned = correcting_stripes.delete_detector_stripes(image, [], [10, 20, 30])
        
        assert cleaned.shape == (1, 97)  # 100 - 3 columns

    def test_single_column_image(self):
        """Test with single column image."""
        image = np.random.randint(0, 255, (100, 1), dtype=np.uint16)
        
        # Remove some rows
        cleaned = correcting_stripes.delete_detector_stripes(image, [10, 20, 30], [])
        
        assert cleaned.shape == (97, 1)  # 100 - 3 rows

    def test_remove_all_rows(self):
        """Test removing all rows."""
        image = np.random.randint(0, 255, (5, 10), dtype=np.uint16)
        
        # Remove all rows
        all_rows = list(range(5))
        cleaned = correcting_stripes.delete_detector_stripes(image, all_rows, [])
        
        # Should result in empty array
        assert cleaned.size == 0

    def test_remove_all_columns(self):
        """Test removing all columns."""
        image = np.random.randint(0, 255, (10, 5), dtype=np.uint16)
        
        # Remove all columns
        all_cols = list(range(5))
        cleaned = correcting_stripes.delete_detector_stripes(image, [], all_cols)
        
        # Should result in empty array
        assert cleaned.size == 0

    def test_unsorted_stripe_indices(self):
        """Test with unsorted stripe indices."""
        image = np.arange(100).reshape(10, 10)
        
        # Provide unsorted indices
        stripe_rows = [8, 2, 5, 1]
        stripe_cols = [7, 3, 9, 0]
        
        cleaned = correcting_stripes.delete_detector_stripes(
            image, stripe_rows, stripe_cols
        )
        
        # Should still work correctly
        expected_shape = (6, 6)  # 10-4 rows, 10-4 cols
        assert cleaned.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
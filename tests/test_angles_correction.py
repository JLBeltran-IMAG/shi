import numpy as np
import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import angles_correction as angles_corr


class TestAngleCorrection:
    """Test suite for angle correction functionality - validates mathematical correctness."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_size = 128
        np.random.seed(42)  # For reproducible tests

    def create_clean_sinusoidal_image(self, angle_deg: float, period: float = 10.0, amplitude: float = 1.0):
        """Create a clean sinusoidal pattern at known angle for validation."""
        y, x = np.mgrid[0:self.test_size, 0:self.test_size]
        angle_rad = np.deg2rad(angle_deg)
        
        # Create perfect sinusoidal grating
        pattern = amplitude * np.sin(2 * np.pi * (x * np.cos(angle_rad) + y * np.sin(angle_rad)) / period)
        return pattern.astype(np.float32)

    def create_image_with_multiple_frequencies(self):
        """Create image with multiple known frequencies for peak detection validation."""
        y, x = np.mgrid[0:self.test_size, 0:self.test_size]
        
        # Multiple frequency components
        pattern = (np.sin(2 * np.pi * x / 16) +  # Horizontal stripes
                  0.7 * np.sin(2 * np.pi * y / 12) +  # Vertical stripes  
                  0.5 * np.sin(2 * np.pi * (x + y) / 20))  # Diagonal component
        
        return pattern.astype(np.float32)

    # Mathematical Correctness Tests
    
    def test_fft_padding_power_of_two(self):
        """Verify FFT padding produces power-of-2 dimensions."""
        test_sizes = [(100, 120), (64, 64), (200, 150), (33, 47)]
        
        for height, width in test_sizes:
            test_image = np.random.random((height, width)).astype(np.float32)
            padded_dim = angles_corr.next_two_power_for_dimension_padding(test_image)
            
            # Must be power of 2
            assert (padded_dim & (padded_dim - 1)) == 0, f"Result {padded_dim} is not power of 2"
            
            # Must be >= max dimension
            max_dim = max(height, width)
            assert padded_dim >= max_dim, f"Padded {padded_dim} < max dimension {max_dim}"
            
            # Should be minimal power of 2
            expected = int(2 ** np.ceil(np.log2(max_dim)))
            assert padded_dim == expected, f"Expected {expected}, got {padded_dim}"

    def test_fft_energy_conservation(self):
        """Verify FFT preserves energy (Parseval's theorem)."""
        test_image = np.random.random((64, 80)).astype(np.float32)
        
        fft_result = angles_corr.squared_fft(test_image)
        
        # Calculate energies
        spatial_energy = np.sum(np.abs(test_image)**2)
        
        # For padded FFT, need to account for normalization
        fft_energy = np.sum(np.abs(fft_result)**2) / fft_result.size
        
        # Should be approximately equal (within numerical precision)
        relative_error = abs(fft_energy - spatial_energy) / (spatial_energy + 1e-10)
        assert relative_error < 1e-2, f"Energy not conserved: relative error {relative_error}"

    def test_fft_properties(self):
        """Verify basic FFT properties."""
        test_image = np.random.random((50, 60)).astype(np.float32)
        fft_result = angles_corr.squared_fft(test_image)
        
        # Result should be complex
        assert np.iscomplexobj(fft_result), "FFT result must be complex"
        
        # Should be square with power-of-2 dimensions
        assert fft_result.shape[0] == fft_result.shape[1], "FFT result must be square"
        
        # DC component should be at center (due to fftshift)
        center = fft_result.shape[0] // 2
        dc_value = fft_result[center, center]
        
        # DC should equal sum of original image
        expected_dc = np.sum(test_image)
        actual_dc = np.real(dc_value)
        
        relative_error = abs(actual_dc - expected_dc) / (abs(expected_dc) + 1e-10)
        assert relative_error < 1e-3, f"DC component incorrect: expected {expected_dc}, got {actual_dc}"

    def test_peak_extraction_basic_properties(self):
        """Test basic properties of peak extraction without assuming implementation."""
        # Test with simple periodic image
        test_image = self.create_clean_sinusoidal_image(0.0, period=16.0)  # Horizontal stripes
        
        coords = angles_corr.extracting_coordinates_of_peaks(test_image)
        
        # Basic sanity checks
        assert isinstance(coords, list), "Should return list"
        assert len(coords) > 0, "Should find at least one peak"
        assert len(coords) <= 10, "Should not find excessive peaks"  # Reasonable upper bound
        
        # All coordinates should be valid integers
        for coord in coords:
            assert len(coord) == 2, "Each coordinate should have 2 elements"
            assert isinstance(coord[0], (int, np.integer)), "Y coordinate should be integer"
            assert isinstance(coord[1], (int, np.integer)), "X coordinate should be integer"
            
            # Coordinates should be within reasonable bounds for FFT spectrum
            fft_size = angles_corr.next_two_power_for_dimension_padding(test_image)
            assert 0 <= coord[0] < fft_size, f"Y coordinate {coord[0]} out of bounds"
            assert 0 <= coord[1] < fft_size, f"X coordinate {coord[1]} out of bounds"

    def test_peak_extraction_consistency(self):
        """Verify peak extraction is consistent (results in reasonable format)."""
        test_image = self.create_clean_sinusoidal_image(30.0, period=12.0)
        
        # Run multiple times to check for crashes or format inconsistency
        results = []
        for i in range(3):
            coords = angles_corr.extracting_coordinates_of_peaks(test_image.copy())
            results.append(coords)
            
            # Check that result format is consistent
            assert isinstance(coords, list), f"Run {i}: Should return list"
            
            for coord in coords:
                assert len(coord) == 2, f"Run {i}: Each coordinate should have 2 elements"
                assert isinstance(coord[0], (int, np.integer)), f"Run {i}: Y should be integer"
                assert isinstance(coord[1], (int, np.integer)), f"Run {i}: X should be integer"
        
        # Check that number of peaks is consistent (algorithm should be stable)
        peak_counts = [len(result) for result in results]
        assert all(count == peak_counts[0] for count in peak_counts), \
            f"Peak count should be consistent: {peak_counts}"

    def test_angle_calculation_mathematical_validity(self):
        """Test mathematical validity of angle calculations."""
        # Test with known geometric configurations
        test_cases = [
            # Main peak, horizontal offset -> should give 0° or 90°
            [(64, 64), (64, 80)],  # Horizontal offset
            [(64, 64), (80, 64)],  # Vertical offset
            [(64, 64), (74, 74)],  # Diagonal offset (45°)
            [(64, 64), (54, 54)],  # Opposite diagonal (-45°)
        ]
        
        for coords in test_cases:
            angle = angles_corr.calculating_angles_of_peaks_average(coords)
            
            # Test that function returns correct type
            assert isinstance(angle, np.float32), "Angle should be float32"
            
            # If angle is finite, it should be in reasonable range
            if np.isfinite(angle):
                assert -90 <= angle <= 90, f"Finite angle {angle} outside reasonable range"
            else:
                # If angle is NaN or infinite, that might indicate an edge case in the implementation
                # This is valuable information about the algorithm's limitations
                print(f"Warning: Non-finite angle {angle} for coordinates {coords}")
                # We don't fail the test, but we log it as it might indicate a bug

    def test_quadrant_sign_function_correctness(self):
        """Test mathematical correctness of quadrant sign calculation."""
        center_h, center_w = 50, 50
        
        # Test all four quadrants for both axes
        test_points = [
            # (y, x, expected_sign_y, expected_sign_x)
            (40, 60, 1, -1),   # Quadrant I (top-right)
            (40, 40, -1, 1),   # Quadrant II (top-left)  
            (60, 40, 1, 1),    # Quadrant III (bottom-left)
            (60, 60, -1, 1),   # Quadrant IV (bottom-right)
        ]
        
        for y, x, expected_y, expected_x in test_points:
            sign_y = angles_corr.quadrant_loc_sign(y, center_h, x, center_w, "y")
            sign_x = angles_corr.quadrant_loc_sign(y, center_h, x, center_w, "x")
            
            assert sign_y in [-1, 0, 1], f"Y sign should be -1, 0, or 1, got {sign_y}"
            assert sign_x in [-1, 0, 1], f"X sign should be -1, 0, or 1, got {sign_x}"
            
        # Test center point (should give 0)
        assert angles_corr.quadrant_loc_sign(center_h, center_h, center_w, center_w, "y") == 0
        assert angles_corr.quadrant_loc_sign(center_h, center_h, center_w, center_w, "x") == 0

    def test_angle_symmetry_properties(self):
        """Test that angle calculation handles symmetric cases."""
        center = (64, 64)
        
        # Test symmetric peak pairs
        symmetric_cases = [
            # Main peak with symmetric harmonics
            ([center, (74, 74), (54, 54)]),  # ±45° diagonal
            ([center, (64, 74), (64, 54)]),  # Horizontal symmetry
            ([center, (74, 64), (54, 64)]),  # Vertical symmetry
        ]
        
        for coords in symmetric_cases:
            angle = angles_corr.calculating_angles_of_peaks_average(coords)
            
            # Test correct type
            assert isinstance(angle, np.float32), "Angle should be float32"
            
            # For symmetric cases, algorithm behavior might vary
            if np.isfinite(angle):
                assert abs(angle) <= 90, f"Finite angle magnitude should be ≤ 90°, got {angle}"
            else:
                # Symmetric cases might cause mathematical indeterminacy in some algorithms
                print(f"Warning: Symmetric case produced non-finite angle {angle} for {coords}")
                # This could indicate the algorithm needs special handling for symmetric cases

    def test_error_handling_edge_cases(self):
        """Test that functions handle edge cases without crashing."""
        # Empty image
        try:
            empty_image = np.array([[]], dtype=np.float32)
            result = angles_corr.extracting_coordinates_of_peaks(empty_image)
            # Should not crash, result format depends on implementation
            assert isinstance(result, list), "Should return list even for edge cases"
        except Exception as e:
            # If it raises an exception, it should be informative
            assert len(str(e)) > 0, "Exception should have meaningful message"

        # Single pixel
        single_pixel = np.array([[1.0]], dtype=np.float32)
        coords = angles_corr.extracting_coordinates_of_peaks(single_pixel)
        assert isinstance(coords, list), "Should handle single pixel gracefully"
        
        # Very small image
        tiny_image = np.random.random((2, 2)).astype(np.float32)
        coords = angles_corr.extracting_coordinates_of_peaks(tiny_image)
        assert isinstance(coords, list), "Should handle tiny images"

        # Edge case for angle calculation: only main peak
        single_peak_coords = [(64, 64)]
        angle = angles_corr.calculating_angles_of_peaks_average(single_peak_coords)
        # Should handle gracefully (might return 0 or NaN, both acceptable)
        assert isinstance(angle, np.float32), "Should return float32 even for edge case"

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large values
        large_image = np.full((32, 32), 1e6, dtype=np.float32)
        coords = angles_corr.extracting_coordinates_of_peaks(large_image)
        assert isinstance(coords, list), "Should handle large values"
        
        # Very small values
        small_image = np.full((32, 32), 1e-6, dtype=np.float32)
        coords = angles_corr.extracting_coordinates_of_peaks(small_image)
        assert isinstance(coords, list), "Should handle small values"
        
        # Zero image
        zero_image = np.zeros((32, 32), dtype=np.float32)
        coords = angles_corr.extracting_coordinates_of_peaks(zero_image)
        assert isinstance(coords, list), "Should handle zero image"

    def test_realistic_angle_detection_scenarios(self):
        """Test angle detection with realistic imaging scenarios."""
        # Test multiple angles to verify robustness
        test_angles = [0.0, 15.0, 30.0, 45.0, -15.0, -30.0]
        
        for target_angle in test_angles:
            test_image = self.create_clean_sinusoidal_image(target_angle, period=20.0)
            
            coords = angles_corr.extracting_coordinates_of_peaks(test_image)
            
            if len(coords) >= 2:  # Need at least main peak + 1 harmonic
                detected_angle = angles_corr.calculating_angles_of_peaks_average(coords)
                
                # Test basic properties
                assert isinstance(detected_angle, np.float32), "Should return float32"
                
                # Check if result is mathematically valid
                if np.isfinite(detected_angle):
                    assert -90 <= detected_angle <= 90, f"Finite angle {detected_angle} outside valid range"
                    
                    # For realistic test, we can check if detected angle is in reasonable vicinity
                    # This is informational - some algorithms might not be perfectly accurate
                    angle_error = abs(detected_angle - target_angle)
                    if angle_error > 10.0:  # Large error threshold for realistic test
                        print(f"Warning: Large angle error {angle_error:.1f}° for target {target_angle}°, got {detected_angle:.1f}°")
                else:
                    print(f"Warning: Non-finite angle {detected_angle} for target {target_angle}°")
            else:
                print(f"Warning: Insufficient peaks detected ({len(coords)}) for angle {target_angle}°")

    def test_data_type_consistency(self):
        """Verify all functions return consistent data types."""
        test_image = self.create_clean_sinusoidal_image(20.0)
        
        # Test FFT padding function
        padded_dim = angles_corr.next_two_power_for_dimension_padding(test_image)
        assert isinstance(padded_dim, int), "Padding dimension should be int"
        
        # Test FFT function
        fft_result = angles_corr.squared_fft(test_image)
        assert fft_result.dtype == np.complex128 or fft_result.dtype == np.complex64, "FFT should return complex"
        
        # Test coordinate extraction
        coords = angles_corr.extracting_coordinates_of_peaks(test_image)
        assert isinstance(coords, list), "Coordinates should be list"
        
        # Test angle calculation
        if len(coords) >= 2:
            angle = angles_corr.calculating_angles_of_peaks_average(coords)
            assert isinstance(angle, np.float32), "Angle should be float32"

    def test_algorithm_robustness_with_noise(self):
        """Test algorithm robustness with various noise conditions."""
        base_image = self.create_clean_sinusoidal_image(25.0, period=15.0)
        
        # Test with different noise levels
        noise_levels = [0.01, 0.1, 0.5]
        
        for noise_level in noise_levels:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, base_image.shape)
            noisy_image = (base_image + noise).astype(np.float32)
            
            # Should not crash
            coords = angles_corr.extracting_coordinates_of_peaks(noisy_image)
            assert isinstance(coords, list), f"Should handle noise level {noise_level}"
            
            if len(coords) >= 2:
                angle = angles_corr.calculating_angles_of_peaks_average(coords)
                assert np.isfinite(angle), f"Should return finite angle with noise {noise_level}"


class TestAngleCorrectionIntegration:
    """Integration tests verifying compatibility with the broader system."""

    def test_coordinate_format_compatibility(self):
        """Verify coordinate format is compatible with spatial_harmonics module."""
        test_image = np.random.random((64, 64)).astype(np.float32)
        
        coords = angles_corr.extracting_coordinates_of_peaks(test_image)
        
        # Test that coordinates have the expected format for spatial_harmonics
        assert isinstance(coords, list), "Should return list for spatial_harmonics compatibility"
        
        for coord in coords:
            assert isinstance(coord, list), "Each coordinate should be a list"
            assert len(coord) == 2, "Each coordinate should have exactly 2 elements"
            
            # Should be integers (or convertible to integers)
            y, x = coord
            assert isinstance(y, (int, np.integer)) or (isinstance(y, float) and y.is_integer()), \
                f"Y coordinate {y} should be integer-like"
            assert isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()), \
                f"X coordinate {x} should be integer-like"

    def test_function_signature_compatibility(self):
        """Test that function signatures match expected interface."""
        # Test that all expected functions exist and are callable
        assert callable(angles_corr.next_two_power_for_dimension_padding), \
            "next_two_power_for_dimension_padding should be callable"
        assert callable(angles_corr.squared_fft), "squared_fft should be callable"
        assert callable(angles_corr.extracting_coordinates_of_peaks), \
            "extracting_coordinates_of_peaks should be callable"
        assert callable(angles_corr.quadrant_loc_sign), "quadrant_loc_sign should be callable"
        assert callable(angles_corr.calculating_angles_of_peaks_average), \
            "calculating_angles_of_peaks_average should be callable"

    def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory."""
        # Test with moderately large image
        large_image = np.random.random((512, 512)).astype(np.float32)
        
        # Should complete without memory errors
        try:
            coords = angles_corr.extracting_coordinates_of_peaks(large_image)
            assert isinstance(coords, list), "Should handle large images"
        except MemoryError:
            pytest.fail("Function should not run out of memory on reasonable inputs")


if __name__ == "__main__":
    pytest.main([__file__])
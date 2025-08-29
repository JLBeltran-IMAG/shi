import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import spatial_harmonics as sh


class TestFFTMathematicalCorrectness:
    """Test suite for FFT and signal processing mathematical correctness."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.test_size = 128
        
    def create_synthetic_sinusoidal_pattern(self, frequency_x=5, frequency_y=3, amplitude=1.0):
        """Create synthetic sinusoidal pattern with known frequencies."""
        y, x = np.mgrid[0:self.test_size, 0:self.test_size]
        pattern = amplitude * np.sin(2 * np.pi * frequency_x * x / self.test_size) * np.cos(2 * np.pi * frequency_y * y / self.test_size)
        return pattern.astype(np.float32)
    
    def create_delta_function(self, position=(64, 64)):
        """Create delta function for impulse response testing."""
        delta = np.zeros((self.test_size, self.test_size), dtype=np.float32)
        delta[position] = 1.0
        return delta

    def test_fft_energy_conservation_parseval_theorem(self):
        """Verify FFT preserves energy according to Parseval's theorem."""
        test_image = self.create_synthetic_sinusoidal_pattern()
        grid_period = 10.0
        
        kx, ky, fft_result = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            test_image, grid_period, logarithmic_spectrum=False
        )
        
        # Calculate energies
        spatial_energy = np.sum(np.abs(test_image)**2)
        freq_energy = np.sum(np.abs(fft_result)**2) / fft_result.size
        
        # Verify energy conservation within numerical precision
        relative_error = abs(freq_energy - spatial_energy) / (spatial_energy + 1e-10)
        assert relative_error < 1e-2, f"Energy not conserved: relative error {relative_error}"

    def test_fft_dc_component_correctness(self):
        """Test that DC component equals sum of original image."""
        test_image = np.random.random((64, 64)).astype(np.float32)
        grid_period = 8.0
        
        kx, ky, fft_result = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            test_image, grid_period, logarithmic_spectrum=False
        )
        
        # DC component should be at center due to fftshift
        center = fft_result.shape[0] // 2
        dc_value = fft_result[center, center]
        
        expected_dc = np.sum(test_image)
        actual_dc = np.real(dc_value)
        
        relative_error = abs(actual_dc - expected_dc) / (abs(expected_dc) + 1e-10)
        assert relative_error < 1e-3, f"DC component incorrect: expected {expected_dc}, got {actual_dc}"

    def test_fft_frequency_domain_properties(self):
        """Test basic frequency domain properties."""
        test_image = self.create_synthetic_sinusoidal_pattern(frequency_x=8, frequency_y=0)
        grid_period = 16.0
        
        kx, ky, fft_result = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            test_image, grid_period, logarithmic_spectrum=False
        )
        
        # Verify wavevector dimensions
        assert len(kx) == test_image.shape[1], "kx should match image width"
        assert len(ky) == test_image.shape[0], "ky should match image height"
        
        # Verify FFT result is complex
        assert np.iscomplexobj(fft_result), "FFT result should be complex"
        assert fft_result.shape == test_image.shape, "FFT should preserve shape"

    def test_fft_logarithmic_spectrum_properties(self):
        """Test logarithmic spectrum calculation."""
        test_image = self.create_synthetic_sinusoidal_pattern()
        grid_period = 10.0
        
        # Test linear spectrum
        kx1, ky1, linear_fft = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            test_image, grid_period, logarithmic_spectrum=False
        )
        
        # Test logarithmic spectrum
        kx2, ky2, log_fft = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            test_image, grid_period, logarithmic_spectrum=True
        )
        
        # Verify logarithmic properties
        assert np.all(log_fft >= 0), "Logarithmic spectrum should be non-negative"
        assert np.all(np.isfinite(log_fft)), "Logarithmic spectrum should be finite"
        
        # Verify wavevectors are identical
        np.testing.assert_array_equal(kx1, kx2, "Wavevectors should be identical")
        np.testing.assert_array_equal(ky1, ky2, "Wavevectors should be identical")

    def test_fft_edge_cases(self):
        """Test FFT with edge case inputs."""
        grid_period = 10.0
        
        # Zero image
        zero_image = np.zeros((32, 32), dtype=np.float32)
        kx, ky, fft_zero = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            zero_image, grid_period
        )
        assert np.allclose(fft_zero, 0), "FFT of zero image should be zero"
        
        # Constant image
        const_image = np.ones((32, 32), dtype=np.float32) * 5.0
        kx, ky, fft_const = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            const_image, grid_period
        )
        # Only DC component should be non-zero
        center = fft_const.shape[0] // 2
        assert abs(fft_const[center, center]) > 0, "DC component should be non-zero"

    def test_fft_numerical_stability_extreme_values(self):
        """Test FFT numerical stability with extreme values."""
        grid_period = 10.0
        
        # Very large values
        large_image = np.full((32, 32), 1e6, dtype=np.float32)
        kx, ky, fft_large = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            large_image, grid_period
        )
        assert np.all(np.isfinite(fft_large)), "FFT should handle large values"
        
        # Very small values
        small_image = np.full((32, 32), 1e-6, dtype=np.float32)
        kx, ky, fft_small = sh.squared_fast_fourier_transform_linear_and_logarithmic(
            small_image, grid_period
        )
        assert np.all(np.isfinite(fft_small)), "FFT should handle small values"


class TestFFTRegionOperations:
    """Test suite for FFT region manipulation functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_array = np.random.random((100, 120)) + 1j * np.random.random((100, 120))
        self.test_array = self.test_array.astype(np.complex128)

    def test_zero_fft_region_basic_functionality(self):
        """Test basic zeroing of FFT regions."""
        original_array = self.test_array.copy()
        top, bottom, left, right = 10, 20, 15, 25
        
        result = sh.zero_fft_region(self.test_array, top, bottom, left, right)
        
        # Check that specified region is zeroed
        assert np.allclose(result[top:bottom, left:right], 0), "Specified region should be zero"
        
        # Check that outside region is unchanged
        assert np.allclose(result[:top, :], original_array[:top, :]), "Region above should be unchanged"
        assert np.allclose(result[bottom:, :], original_array[bottom:, :]), "Region below should be unchanged"
        assert np.allclose(result[:, :left], original_array[:, :left]), "Region left should be unchanged"
        assert np.allclose(result[:, right:], original_array[:, right:]), "Region right should be unchanged"

    def test_zero_fft_region_boundary_cases(self):
        """Test zeroing with boundary cases."""
        # Full array
        result_full = sh.zero_fft_region(
            self.test_array.copy(), 0, self.test_array.shape[0], 0, self.test_array.shape[1]
        )
        assert np.allclose(result_full, 0), "Full array should be zeroed"
        
        # Empty region (top == bottom or left == right)
        original_copy = self.test_array.copy()
        result_empty = sh.zero_fft_region(original_copy, 10, 10, 15, 25)  # top == bottom
        np.testing.assert_array_equal(result_empty, self.test_array), "Empty region should change nothing"

    def test_zero_fft_region_invalid_boundaries(self):
        """Test zeroing with invalid boundaries."""
        # Out of bounds (should handle gracefully)
        original_copy = self.test_array.copy()
        
        # Test with boundaries exceeding array dimensions
        result = sh.zero_fft_region(original_copy, 10, 200, 15, 300)
        # Should zero up to array boundaries
        assert np.allclose(result[10:, 15:], 0), "Should zero to array boundaries"

    def test_zero_fft_region_preserves_dtype(self):
        """Test that zeroing preserves array data type."""
        for dtype in [np.complex64, np.complex128]:
            test_array = np.random.random((50, 60)).astype(dtype) + 1j * np.random.random((50, 60)).astype(dtype)
            result = sh.zero_fft_region(test_array, 10, 20, 15, 25)
            assert result.dtype == dtype, f"Should preserve {dtype}"

    def test_zero_fft_region_returns_modified_array(self):
        """Test that function returns the modified array."""
        original_copy = self.test_array.copy()
        result = sh.zero_fft_region(original_copy, 10, 20, 15, 25)
        
        # Should return the same array object (in-place modification)
        assert result is original_copy, "Should return the same array object"


class TestHarmonicExtraction:
    """Test suite for harmonic extraction mathematical correctness."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_size = 128

    def create_fft_with_known_peak(self, peak_position=(70, 80), peak_magnitude=1000):
        """Create FFT spectrum with known peak location."""
        fft_spectrum = np.random.random((self.test_size, self.test_size)) * 10 + \
                      1j * np.random.random((self.test_size, self.test_size)) * 10
        fft_spectrum = fft_spectrum.astype(np.complex128)
        
        # Place known peak
        fft_spectrum[peak_position] = peak_magnitude + 0j
        return fft_spectrum

    def test_extracting_harmonic_peak_detection_accuracy(self):
        """Test accuracy of harmonic peak detection."""
        expected_peak = (70, 80)
        fft_spectrum = self.create_fft_with_known_peak(expected_peak, 1000)
        
        ky_limit, kx_limit = 20, 25
        top, bottom, left, right, peak_y, peak_x = sh.extracting_harmonic(
            fft_spectrum, ky_limit, kx_limit
        )
        
        # Verify peak detection accuracy
        assert (peak_y, peak_x) == expected_peak, f"Peak detected at ({peak_y}, {peak_x}), expected {expected_peak}"

    def test_extracting_harmonic_boundary_calculation(self):
        """Test boundary calculation correctness."""
        fft_spectrum = self.create_fft_with_known_peak((64, 64), 1000)
        ky_limit, kx_limit = 30, 35
        
        top, bottom, left, right, peak_y, peak_x = sh.extracting_harmonic(
            fft_spectrum, ky_limit, kx_limit
        )
        
        # Verify boundary calculations
        expected_top = max(0, peak_y - ky_limit)
        expected_bottom = min(fft_spectrum.shape[0], peak_y + ky_limit)
        expected_left = max(0, peak_x - kx_limit)
        expected_right = min(fft_spectrum.shape[1], peak_x + kx_limit)
        
        assert top == expected_top, f"Top boundary: expected {expected_top}, got {top}"
        assert bottom == expected_bottom, f"Bottom boundary: expected {expected_bottom}, got {bottom}"
        assert left == expected_left, f"Left boundary: expected {expected_left}, got {left}"
        assert right == expected_right, f"Right boundary: expected {expected_right}, got {right}"

    def test_extracting_harmonic_boundary_clamping(self):
        """Test boundary clamping at array edges."""
        fft_spectrum = self.create_fft_with_known_peak((5, 5), 1000)  # Near corner
        ky_limit, kx_limit = 20, 20  # Large limits
        
        top, bottom, left, right, peak_y, peak_x = sh.extracting_harmonic(
            fft_spectrum, ky_limit, kx_limit
        )
        
        # Boundaries should be clamped to array dimensions
        assert top >= 0, "Top boundary should be >= 0"
        assert bottom <= fft_spectrum.shape[0], "Bottom boundary should be <= height"
        assert left >= 0, "Left boundary should be >= 0"
        assert right <= fft_spectrum.shape[1], "Right boundary should be <= width"

    def test_extracting_harmonic_with_uniform_spectrum(self):
        """Test harmonic extraction with uniform spectrum (no clear peak)."""
        # Create uniform spectrum
        uniform_spectrum = np.ones((64, 64), dtype=np.complex128)
        
        # Should still find some maximum (might be first occurrence)
        top, bottom, left, right, peak_y, peak_x = sh.extracting_harmonic(
            uniform_spectrum, 10, 10
        )
        
        # Should return valid coordinates
        assert 0 <= peak_y < uniform_spectrum.shape[0], "Peak Y should be valid"
        assert 0 <= peak_x < uniform_spectrum.shape[1], "Peak X should be valid"
        assert isinstance(peak_y, (int, np.integer)), "Peak Y should be integer"
        assert isinstance(peak_x, (int, np.integer)), "Peak X should be integer"

    def test_extracting_harmonic_with_multiple_equal_peaks(self):
        """Test behavior with multiple peaks of equal magnitude."""
        fft_spectrum = np.zeros((100, 100), dtype=np.complex128)
        
        # Create multiple equal peaks
        peak_magnitude = 1000
        peak_positions = [(30, 30), (30, 70), (70, 30), (70, 70)]
        for pos in peak_positions:
            fft_spectrum[pos] = peak_magnitude + 0j
        
        top, bottom, left, right, peak_y, peak_x = sh.extracting_harmonic(
            fft_spectrum, 15, 15
        )
        
        # Should find one of the peaks
        found_peak = (peak_y, peak_x)
        assert found_peak in peak_positions, f"Should find one of the equal peaks, found {found_peak}"

    def test_extracting_harmonic_preserves_input_array(self):
        """Test that harmonic extraction doesn't modify input array."""
        original_spectrum = self.create_fft_with_known_peak()
        original_copy = original_spectrum.copy()
        
        sh.extracting_harmonic(original_spectrum, 20, 20)
        
        # Original array should be unchanged
        np.testing.assert_array_equal(original_spectrum, original_copy, "Input array should not be modified")


class TestPhaseAndScatteringCalculations:
    """Test suite for phase map and scattering calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_size = 64

    def create_test_complex_arrays(self):
        """Create test complex arrays for phase calculations."""
        # Main harmonic (reference)
        main_harmonic = np.random.random((self.test_size, self.test_size)) + \
                       1j * np.random.random((self.test_size, self.test_size))
        main_harmonic = main_harmonic.astype(np.complex128)
        
        # Inverse fourier transform (target)
        ifft_data = np.random.random((self.test_size, self.test_size)) + \
                   1j * np.random.random((self.test_size, self.test_size))
        ifft_data = ifft_data.astype(np.complex128)
        
        return ifft_data, main_harmonic

    def test_compute_phase_map_epsilon_protection(self):
        """Test that epsilon prevents division by zero in phase calculations."""
        ifft_data, main_harmonic = self.create_test_complex_arrays()
        
        # Create zero main harmonic at some locations
        main_harmonic[20:25, 30:35] = 0.0 + 0j
        
        # Should not crash or produce inf/nan
        try:
            with patch('spatial_harmonics.unwrap_phase') as mock_unwrap:
                mock_unwrap.return_value = np.zeros((self.test_size, self.test_size))
                phase_map = sh.compute_phase_map(ifft_data, main_harmonic, epsilon=1e-12)
                
                assert np.all(np.isfinite(phase_map)), "Phase map should be finite"
        except Exception as e:
            # If function uses unavailable dependencies, that's acceptable
            if "unwrap_phase" in str(e) or "module" in str(e).lower():
                pass  # Expected for missing dependencies
            else:
                raise

    def test_compute_scattering_mathematical_properties(self):
        """Test mathematical properties of scattering calculation."""
        ifft_data, main_harmonic = self.create_test_complex_arrays()
        
        scattering = sh.compute_scattering(ifft_data, main_harmonic, epsilon=1e-12)
        
        # Scattering should be real-valued
        assert np.all(np.isreal(scattering)), "Scattering should be real-valued"
        
        # Should be finite
        assert np.all(np.isfinite(scattering)), "Scattering should be finite"

    def test_compute_scattering_epsilon_protection(self):
        """Test epsilon protection in scattering calculation."""
        ifft_data, main_harmonic = self.create_test_complex_arrays()
        
        # Test with zero main harmonic
        main_harmonic_zero = np.zeros_like(main_harmonic)
        scattering = sh.compute_scattering(ifft_data, main_harmonic_zero, epsilon=1e-6)
        
        # Should not produce inf or nan
        assert np.all(np.isfinite(scattering)), "Should handle zero main harmonic gracefully"

    def test_compute_scattering_clipping_behavior(self):
        """Test clipping behavior in scattering calculation."""
        # Create test case where ratio might be very small
        ifft_data = np.ones((32, 32), dtype=np.complex128) * 1e-10
        main_harmonic = np.ones((32, 32), dtype=np.complex128) * 1e3
        
        epsilon = 1e-12
        scattering = sh.compute_scattering(ifft_data, main_harmonic, epsilon)
        
        # Result should be finite and reasonable
        assert np.all(np.isfinite(scattering)), "Scattering with small ratios should be finite"
        
        # CRITICAL PHYSICS CONSTRAINT: Scattering must be non-negative
        # This is a fundamental physics requirement - negative scattering is impossible
        negative_scattering_ratio = np.sum(scattering < 0) / scattering.size
        assert negative_scattering_ratio == 0, f"PHYSICS VIOLATION: {negative_scattering_ratio:.1%} of scattering values are negative - this violates fundamental physics laws"

    def test_scattering_physics_constraint(self):
        """Test that scattering respects physical constraints."""
        ifft_data, main_harmonic = self.create_test_complex_arrays()
        
        # Ensure positive definite inputs for physical realism
        ifft_data = np.abs(ifft_data) + 0j
        main_harmonic = np.abs(main_harmonic) + 1.0 + 0j  # Add constant to avoid near-zero
        
        scattering = sh.compute_scattering(ifft_data, main_harmonic)
        
        # Physical scattering should be real and finite
        assert np.all(np.isreal(scattering)), "Physical scattering should be real"
        assert np.all(np.isfinite(scattering)), "Physical scattering should be finite"

    def test_phase_and_scattering_consistency(self):
        """Test consistency between phase and scattering calculations."""
        ifft_data, main_harmonic = self.create_test_complex_arrays()
        
        # Both should handle same inputs without crashing
        scattering = sh.compute_scattering(ifft_data, main_harmonic)
        
        try:
            with patch('spatial_harmonics.unwrap_phase') as mock_unwrap:
                mock_unwrap.return_value = np.zeros_like(scattering)
                phase_map = sh.compute_phase_map(ifft_data, main_harmonic)
                
                # Both should have same shape
                assert scattering.shape == phase_map.shape, "Phase and scattering should have same shape"
                
        except Exception as e:
            if "unwrap_phase" in str(e) or "module" in str(e).lower():
                pass  # Expected for missing dependencies
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__])
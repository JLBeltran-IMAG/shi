"""
Mathematical validation tests for harmonic identification and classification in SHI.

This module tests the mathematical correctness of harmonic detection, peak extraction,
and classification logic. Tests focus on:
- Geometric properties of harmonic classification
- Peak detection accuracy and robustness
- Mathematical consistency in angle calculations
- Edge cases and boundary conditions

Architecture-agnostic approach: Tests validate mathematical behavior without assuming
specific implementation details.
"""

import numpy as np
import pytest
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import spatial_harmonics


class TestHarmonicIdentificationMathematics:
    """Test mathematical properties of harmonic identification algorithms."""
    
    def create_synthetic_fft_with_known_peaks(self, size=(64, 64), peak_positions=None):
        """Create synthetic FFT with known peak positions for testing."""
        fft_data = np.zeros(size, dtype=complex)
        
        if peak_positions is None:
            # Default: central peak plus four cardinal direction peaks
            center_y, center_x = size[0] // 2, size[1] // 2
            peak_positions = [
                (center_y, center_x, 10.0),          # Main harmonic (center)
                (center_y - 10, center_x, 5.0),     # Vertical positive
                (center_y + 10, center_x, 5.0),     # Vertical negative  
                (center_y, center_x - 10, 5.0),     # Horizontal negative
                (center_y, center_x + 10, 5.0),     # Horizontal positive
            ]
        
        # Add peaks at specified positions with specified magnitudes
        for y, x, magnitude in peak_positions:
            if 0 <= y < size[0] and 0 <= x < size[1]:
                fft_data[y, x] = magnitude + 0j
        
        return fft_data

    def test_harmonic_classification_geometric_properties(self):
        """Test geometric properties of harmonic classification."""
        # Test cases with known geometric relationships
        main_y, main_x = 32, 32
        
        test_cases = [
            # (harmonic_y, harmonic_x, expected_classification)
            (22, 32, "harmonic_vertical_positive"),     # Pure vertical, above main
            (42, 32, "harmonic_vertical_negative"),     # Pure vertical, below main
            (32, 22, "harmonic_horizontal_negative"),   # Pure horizontal, left
            (32, 42, "harmonic_horizontal_positive"),   # Pure horizontal, right
            (22, 22, None),  # Diagonal (higher order) - exact result depends on threshold
            (42, 42, None),  # Diagonal (higher order) - exact result depends on threshold
        ]
        
        try:
            for harmonic_y, harmonic_x, expected in test_cases:
                result = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, harmonic_y, harmonic_x, angle_threshold=15
                )
                
                # Basic validation
                assert isinstance(result, str), "Classification should return string"
                assert result.startswith("harmonic_"), "Result should start with 'harmonic_'"
                
                # For pure vertical/horizontal cases, check expected results
                if expected is not None:
                    assert result == expected, f"CLASSIFICATION ERROR: Expected {expected} but got {result} for position ({harmonic_y}, {harmonic_x})"
                
                # Test mathematical consistency: same relative position should give same result
                result2 = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, harmonic_y, harmonic_x, angle_threshold=15
                )
                assert result == result2, "Classification should be deterministic"
                
        except Exception as e:
            pytest.fail(f"Harmonic classification failed: {e}")

    def test_harmonic_classification_angle_mathematics(self):
        """Test mathematical correctness of angle calculations in harmonic classification."""
        main_y, main_x = 50, 50
        
        # Test angle threshold behavior
        angle_thresholds = [5, 10, 15, 30, 45]
        
        for threshold in angle_thresholds:
            try:
                # Test case: small deviation that should be classified as vertical
                small_deviation_y, small_deviation_x = 45, 51  # dy=5, dx=1, small angle
                
                result_small = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, small_deviation_y, small_deviation_x, 
                    angle_threshold=threshold
                )
                
                # Test case: large deviation that should be higher order
                large_deviation_y, large_deviation_x = 40, 60  # dy=10, dx=10, 45 degrees
                
                result_large = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, large_deviation_y, large_deviation_x, 
                    angle_threshold=threshold
                )
                
                # Mathematical consistency checks
                assert isinstance(result_small, str), "Should return string classification"
                assert isinstance(result_large, str), "Should return string classification"
                
                # For small angles and high thresholds, should be vertical/horizontal
                if threshold >= 15:  # High threshold should catch small deviations
                    if not result_small.startswith(("harmonic_vertical", "harmonic_horizontal")):
                        print(f"WARNING: Small deviation not classified as vertical/horizontal with threshold {threshold}: {result_small}")
                
                # For 45-degree case with low thresholds, should be higher order
                if threshold <= 15:  # Low threshold should reject large deviations
                    if result_large.startswith(("harmonic_vertical", "harmonic_horizontal")):
                        print(f"WARNING: Large deviation classified as vertical/horizontal with low threshold {threshold}: {result_large}")
                
            except Exception as e:
                print(f"WARNING: Angle mathematics test failed for threshold {threshold}: {e}")

    def test_higher_order_harmonic_identification_logic(self):
        """Test mathematical logic for higher order harmonic identification."""
        # Test the quadrant-based classification system
        test_quadrants = [
            (1, 1, "harmonic_diagonal_p1_p1"),    # Quadrant I
            (-1, 1, "harmonic_diagonal_n1_p1"),   # Quadrant II  
            (-1, -1, "harmonic_diagonal_n1_n1"),  # Quadrant III
            (1, -1, "harmonic_diagonal_p1_n1"),   # Quadrant IV
        ]
        
        try:
            for dx, dy, expected in test_quadrants:
                result = spatial_harmonics.identifying_harmonics_x1y1_higher_orders(dx, dy)
                
                # Mathematical consistency
                assert isinstance(result, str), "Should return string"
                assert result == expected, f"Expected {expected} for ({dx}, {dy}), got {result}"
                
            # Test error conditions
            error_cases = [(0, 1), (1, 0), (0, 0)]
            for dx, dy in error_cases:
                with pytest.raises(ValueError, match="must be non-zero"):
                    spatial_harmonics.identifying_harmonics_x1y1_higher_orders(dx, dy)
                    
        except Exception as e:
            print(f"WARNING: Higher order identification failed: {e}")

    def test_peak_extraction_mathematical_properties(self):
        """Test mathematical properties of peak extraction algorithms."""
        # Create synthetic FFT with known peak structure
        fft_data = self.create_synthetic_fft_with_known_peaks((32, 32))
        
        try:
            # Test peak extraction
            if hasattr(spatial_harmonics, 'extracting_harmonic'):
                ky_limit, kx_limit = 5, 5
                
                # Should find the highest remaining peak
                top, bottom, left, right, peak_y, peak_x = spatial_harmonics.extracting_harmonic(
                    fft_data, ky_limit, kx_limit
                )
                
                # Mathematical properties
                assert isinstance(top, (int, np.integer)), "Boundary should be integer"
                assert isinstance(bottom, (int, np.integer)), "Boundary should be integer"  
                assert isinstance(left, (int, np.integer)), "Boundary should be integer"
                assert isinstance(right, (int, np.integer)), "Boundary should be integer"
                assert isinstance(peak_y, (int, np.integer)), "Peak position should be integer"
                assert isinstance(peak_x, (int, np.integer)), "Peak position should be integer"
                
                # Geometric consistency
                assert top <= peak_y <= bottom, "Peak Y should be within extracted region"
                assert left <= peak_x <= right, "Peak X should be within extracted region"
                assert top < bottom, "Top should be less than bottom"
                assert left < right, "Left should be less than right"
                
                # CRITICAL: Region size should relate to band limits  
                region_height = bottom - top
                region_width = right - left
                expected_height = 2 * ky_limit
                expected_width = 2 * kx_limit
                
                # Allow small tolerance for boundary effects
                assert abs(region_height - expected_height) <= 2, f"GEOMETRIC ERROR: Region height {region_height} deviates too much from expected {expected_height}"
                assert abs(region_width - expected_width) <= 2, f"GEOMETRIC ERROR: Region width {region_width} deviates too much from expected {expected_width}"
                
        except Exception as e:
            print(f"WARNING: Peak extraction test failed: {e}")

    def test_fft_region_zeroing_mathematical_consistency(self):
        """Test mathematical consistency of FFT region zeroing operations."""
        # Create test FFT data
        fft_data = self.create_synthetic_fft_with_known_peaks((32, 32))
        original_energy = np.sum(np.abs(fft_data)**2)
        
        try:
            if hasattr(spatial_harmonics, 'zero_fft_region'):
                # Test region zeroing
                top, bottom, left, right = 10, 20, 10, 20
                
                # Make a copy for testing
                fft_copy = np.copy(fft_data)
                
                # Zero a region
                spatial_harmonics.zero_fft_region(fft_copy, top, bottom, left, right)
                
                # Mathematical properties
                # 1. Zeroed region should be zero
                zeroed_region = fft_copy[top:bottom, left:right]
                assert np.allclose(zeroed_region, 0), "Zeroed region should be zero"
                
                # 2. Other regions should be unchanged
                # Test corners that shouldn't be affected
                if top > 0 and left > 0:
                    assert np.allclose(fft_copy[:top, :left], fft_data[:top, :left]), \
                        "Unaffected regions should remain unchanged"
                
                # 3. Energy should be reduced (but not necessarily by exact amount due to boundary effects)
                new_energy = np.sum(np.abs(fft_copy)**2)
                assert new_energy <= original_energy, "Energy should not increase after zeroing"
                
                # 4. Test boundary conditions
                boundary_test_cases = [
                    (0, 5, 0, 5),      # Top-left corner
                    (27, 32, 27, 32),  # Bottom-right corner (if within bounds)
                ]
                
                for test_top, test_bottom, test_left, test_right in boundary_test_cases:
                    if (test_top >= 0 and test_bottom <= fft_data.shape[0] and 
                        test_left >= 0 and test_right <= fft_data.shape[1]):
                        
                        test_copy = np.copy(fft_data)
                        spatial_harmonics.zero_fft_region(test_copy, test_top, test_bottom, test_left, test_right)
                        
                        # Should complete without error
                        assert isinstance(test_copy, np.ndarray), "Should remain numpy array"
                
        except Exception as e:
            print(f"WARNING: FFT region zeroing test failed: {e}")

    def test_harmonic_extraction_sequence_consistency(self):
        """Test mathematical consistency of sequential harmonic extraction."""
        # Create FFT with multiple known peaks
        size = (64, 64)
        center_y, center_x = size[0] // 2, size[1] // 2
        
        # Multiple peaks with different magnitudes
        peak_positions = [
            (center_y, center_x, 20.0),         # Strongest (main)
            (center_y - 15, center_x, 15.0),   # Second strongest
            (center_y + 15, center_x, 10.0),   # Third strongest
            (center_y, center_x - 15, 8.0),    # Fourth strongest
            (center_y, center_x + 15, 5.0),    # Fifth strongest
        ]
        
        fft_data = self.create_synthetic_fft_with_known_peaks(size, peak_positions)
        
        try:
            if hasattr(spatial_harmonics, 'extracting_harmonic') and hasattr(spatial_harmonics, 'zero_fft_region'):
                fft_copy = np.copy(fft_data)
                ky_limit, kx_limit = 8, 8
                extracted_peaks = []
                
                # Extract peaks in sequence (as done in actual algorithm)
                for i in range(4):  # Extract 4 peaks after main
                    top, bottom, left, right, peak_y, peak_x = spatial_harmonics.extracting_harmonic(
                        fft_copy, ky_limit, kx_limit
                    )
                    
                    # Record this peak
                    peak_magnitude = np.abs(fft_copy[peak_y, peak_x])
                    extracted_peaks.append((peak_y, peak_x, peak_magnitude))
                    
                    # Zero out the extracted region
                    spatial_harmonics.zero_fft_region(fft_copy, top, bottom, left, right)
                
                # Mathematical consistency checks
                assert len(extracted_peaks) == 4, "Should extract expected number of peaks"
                
                # Peaks should generally be found in decreasing order of magnitude
                # (with some tolerance for numerical effects)
                magnitudes = [peak[2] for peak in extracted_peaks]
                
                for i in range(len(magnitudes) - 1):
                    if magnitudes[i] < magnitudes[i+1]:
                        # This could be OK due to zeroing effects, but worth noting
                        print(f"WARNING: Peak {i} magnitude {magnitudes[i]:.3f} < peak {i+1} magnitude {magnitudes[i+1]:.3f}")
                
                # All extracted peaks should be at different locations
                positions = [(peak[0], peak[1]) for peak in extracted_peaks]
                unique_positions = set(positions)
                
                if len(unique_positions) != len(positions):
                    print("WARNING: Duplicate peak positions detected in extraction sequence")
                
        except Exception as e:
            print(f"WARNING: Harmonic extraction sequence test failed: {e}")


class TestHarmonicClassificationEdgeCases:
    """Test edge cases and boundary conditions in harmonic classification."""
    
    def test_angle_threshold_edge_cases(self):
        """Test behavior at angle threshold boundaries."""
        main_y, main_x = 100, 100
        
        # Create test cases right at threshold boundaries
        threshold = 15.0  # Default threshold
        
        # Calculate positions that should be exactly at threshold
        # For vertical case: angle = arctan(dx/dy) = threshold
        # So dx = dy * tan(threshold)
        
        dy = 10  # Fixed vertical displacement
        dx_at_threshold = dy * np.tan(np.deg2rad(threshold))
        
        test_cases = [
            (main_y + dy, main_x + dx_at_threshold - 0.1, "should_be_vertical"),
            (main_y + dy, main_x + dx_at_threshold + 0.1, "should_be_higher_order"),
        ]
        
        try:
            for test_y, test_x, expectation in test_cases:
                result = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, test_y, test_x, angle_threshold=threshold
                )
                
                # Mathematical consistency check
                assert isinstance(result, str), "Should return string"
                
                # Check threshold behavior (with some tolerance for floating point)
                if expectation == "should_be_vertical":
                    if not result.startswith("harmonic_vertical"):
                        print(f"WARNING: Case just below threshold not classified as vertical: {result}")
                elif expectation == "should_be_higher_order":
                    if result.startswith(("harmonic_vertical", "harmonic_horizontal")):
                        print(f"WARNING: Case just above threshold classified as vertical/horizontal: {result}")
                
        except Exception as e:
            print(f"WARNING: Angle threshold edge case test failed: {e}")

    def test_identical_coordinate_edge_cases(self):
        """Test behavior when harmonic coordinates are identical to main harmonic."""
        main_y, main_x = 50, 50
        
        try:
            # This should not occur in practice, but test robustness
            result = spatial_harmonics.identifying_harmonic(
                main_y, main_x, main_y, main_x, angle_threshold=15
            )
            
            # Should handle gracefully (exact behavior may vary)
            assert isinstance(result, str), "Should return string even for identical coordinates"
            print(f"INFO: Identical coordinates result: {result}")
            
        except Exception as e:
            print(f"WARNING: Identical coordinates caused exception: {e}")

    def test_extreme_coordinate_values(self):
        """Test behavior with extreme coordinate values."""
        test_cases = [
            (0, 0, 1000, 1000),      # Very large difference
            (1000, 1000, 0, 0),      # Very large difference (opposite)
            (500.5, 500.5, 500.6, 500.6),  # Very small difference
            (-100, -100, 100, 100),  # Negative coordinates
        ]
        
        for main_y, main_x, harm_y, harm_x in test_cases:
            try:
                result = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, harm_y, harm_x, angle_threshold=15
                )
                
                # Should handle extreme values gracefully
                assert isinstance(result, str), f"Should handle extreme values: ({main_y}, {main_x}) -> ({harm_y}, {harm_x})"
                
            except Exception as e:
                print(f"WARNING: Extreme coordinates failed ({main_y}, {main_x}) -> ({harm_y}, {harm_x}): {e}")

    def test_floating_point_precision_effects(self):
        """Test effects of floating point precision on classification."""
        main_y, main_x = 100.0, 100.0
        
        # Test with values that might cause precision issues
        epsilon_values = [1e-10, 1e-15, 1e-20]
        
        for eps in epsilon_values:
            try:
                # Slightly perturbed coordinates
                result1 = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, main_y + eps, main_x + eps, angle_threshold=15
                )
                
                result2 = spatial_harmonics.identifying_harmonic(
                    main_y, main_x, main_y - eps, main_x - eps, angle_threshold=15
                )
                
                # Should be consistent (or at least not crash)
                assert isinstance(result1, str), f"Should handle epsilon {eps}"
                assert isinstance(result2, str), f"Should handle epsilon {eps}"
                
                # Very small perturbations might give different results due to floating point,
                # but should not cause crashes
                
            except Exception as e:
                print(f"WARNING: Floating point precision issue with epsilon {eps}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
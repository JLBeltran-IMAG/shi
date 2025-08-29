# Test Implementation Tasks

## Phase 1: Test Infrastructure Setup

### Task 1: Setup Testing Environment
**Priority:** High  
**Estimated Time:** 2 hours

#### Subtasks:
1. **Create test directory structure**
   - [ ] Create `/tests` directory
   - [ ] Create `/tests/fixtures` directory
   - [ ] Create `/tests/__init__.py` file
   - [ ] Add `.gitignore` entries for test artifacts

2. **Install testing dependencies**
   - [ ] Create `requirements-dev.txt` with:
     - pytest==7.4.0
     - pytest-cov==4.1.0
     - pytest-mock==3.11.1
     - numpy==1.24.3
     - scipy==1.11.1
     - scikit-image==0.21.0
   - [ ] Install dependencies: `pip install -r requirements-dev.txt`

3. **Configure pytest**
   - [ ] Create `pytest.ini` with test discovery patterns
   - [ ] Configure coverage settings
   - [ ] Set up test markers for different test categories

**Testing Step:** Run `pytest --version` to verify installation

---

### Task 2: Create Test Fixtures and Utilities
**Priority:** High  
**Estimated Time:** 3 hours

#### Subtasks:
1. **Create conftest.py with base fixtures**
   - [ ] Implement `synthetic_image_2d()` fixture
   - [ ] Implement `synthetic_fft_spectrum()` fixture
   - [ ] Implement `wrapped_phase_map()` fixture
   - [ ] Implement `known_harmonic_peaks()` fixture

2. **Create test data generators**
   - [ ] Function to generate sinusoidal patterns
   - [ ] Function to generate noise patterns
   - [ ] Function to create edge cases (zeros, NaN, infinity)

3. **Create assertion helpers**
   - [ ] `assert_array_almost_equal()` for numerical comparisons
   - [ ] `assert_phase_continuity()` for unwrapped phase validation
   - [ ] `assert_energy_conservation()` for FFT validation

**Testing Step:** Run `pytest tests/conftest.py --collect-only` to verify fixtures are discoverable

---

## Phase 2: Core Algorithm Tests

### Task 3: Implement FFT and Harmonic Extraction Tests
**Priority:** Critical  
**Estimated Time:** 4 hours

#### Subtasks:
1. **Create test_core_algorithms.py**
   - [ ] Create file with proper imports
   - [ ] Setup TestFFTProcessing class
   - [ ] Setup TestHarmonicExtraction class

2. **Implement FFT tests**
   - [ ] `test_fft_computation_known_pattern()`
     - Create 128x128 image with sin(2x) + cos(3y)
     - Verify peaks at frequencies 2 and 3
   - [ ] `test_fft_shift_centering()`
     - Verify DC component at center after fftshift
   - [ ] `test_wavevector_calculation()`
     - Test with mask_period=10, verify spacing
   - [ ] `test_fft_parseval_theorem()`
     - Verify energy conservation

3. **Implement harmonic extraction tests**
   - [ ] `test_harmonic_peak_detection()`
     - Create spectrum with peaks at (64,80), (80,64)
     - Verify correct identification
   - [ ] `test_harmonic_classification()`
     - Test all classifications: vertical, horizontal, diagonal
   - [ ] `test_harmonic_boundary_calculation()`
     - Verify boundary limits don't exceed array dimensions
   - [ ] `test_zero_fft_region()`
     - Verify region is zeroed, rest unchanged

**Testing Step:** Run `pytest tests/test_core_algorithms.py -v` and verify all tests pass

---

### Task 4: Implement Phase Unwrapping Tests
**Priority:** Critical  
**Estimated Time:** 5 hours

#### Subtasks:
1. **Create test_phase_unwrapping.py**
   - [ ] Import unwrapping_phase module
   - [ ] Setup TestPhaseUnwrapping class
   - [ ] Setup TestPhaseGradient class

2. **Implement basic phase tests**
   - [ ] `test_wrapped_phase_calculation()`
     - Test np.angle() behavior with complex inputs
   - [ ] `test_phase_gradient_computation()`
     - Verify gradient calculation for known phase slopes
   - [ ] `test_phase_wrapping_detection()`
     - Detect 2π discontinuities

3. **Implement unwrapping algorithm tests**
   - [ ] `test_goldstein_branch_cut_unwrap()`
     - Create phase with known branch cuts
     - Verify unwrapped result is continuous
   - [ ] `test_ls_unwrap_phase()`
     - Test least squares with synthetic wrapped phase
     - Compare with analytical solution
   - [ ] `test_quality_guided_unwrap()`
     - Create phase with varying quality regions
     - Verify high-quality regions unwrapped first

4. **Implement validation tests**
   - [ ] `test_unwrapping_consistency()`
     - Parametrized test for all algorithms
     - Same input should give consistent output
   - [ ] `test_phase_continuity_check()`
     - No 2π jumps in unwrapped phase

**Testing Step:** Run `pytest tests/test_phase_unwrapping.py -v --tb=short`

---

### Task 5: Implement Image Correction Tests
**Priority:** High  
**Estimated Time:** 3 hours

#### Subtasks:
1. **Create test_image_processing.py**
   - [ ] Import corrections module
   - [ ] Setup TestImageCorrections class

2. **Implement correction algorithm tests**
   - [ ] `test_dark_field_subtraction()`
     - Image: [100, 150], Dark: [10, 10]
     - Result should be [90, 140]
   - [ ] `test_flat_field_division()`
     - Test with zero-division handling
     - Verify epsilon prevents infinity
   - [ ] `test_bright_field_normalization()`
     - Test normalization preserves relative intensities
   - [ ] `test_combined_corrections()`
     - Apply dark, then flat, then bright
     - Verify order of operations

3. **Implement angle correction tests**
   - [ ] `test_rotation_angle_calculation()`
     - Create image with known tilt
     - Verify detected angle matches
   - [ ] `test_rotation_preservation()`
     - Verify image properties preserved after rotation

**Testing Step:** Run `pytest tests/test_image_processing.py --cov=src.corrections`

---

### Task 6: Implement Contrast Calculation Tests
**Priority:** High  
**Estimated Time:** 3 hours

#### Subtasks:
1. **Create test_numerical_computations.py**
   - [ ] Setup TestContrastCalculations class
   - [ ] Setup TestDifferentialPhase class

2. **Implement contrast tests**
   - [ ] `test_absorption_calculation()`
     - Test log(1/|IFFT|) with known inputs
     - Verify numerical stability with small values
   - [ ] `test_scattering_computation()`
     - Test harmonic ratio calculations
     - Verify bounds [0, inf)
   - [ ] `test_absorption_positivity()`
     - All absorption values should be ≥ 0

3. **Implement differential phase tests**
   - [ ] `test_differential_phase_sobel()`
     - Create edge image, apply Sobel
     - Compare with expected edge response
   - [ ] `test_differential_phase_gradient()`
     - Test numpy gradient implementation
     - Verify against analytical gradient

**Testing Step:** Run `pytest tests/test_numerical_computations.py -v`

---

## Phase 3: Integration and Edge Cases

### Task 7: Implement Integration Tests
**Priority:** Medium  
**Estimated Time:** 4 hours

#### Subtasks:
1. **Create test_integration.py**
   - [ ] Setup TestProcessingPipeline class
   - [ ] Import all necessary modules

2. **Implement pipeline tests**
   - [ ] `test_full_harmonic_extraction_pipeline()`
     - Image → FFT → Harmonics → Contrast
     - Verify each stage output
   - [ ] `test_correction_chain()`
     - Apply all corrections in sequence
     - Verify cumulative effect
   - [ ] `test_multi_harmonic_processing()`
     - Process 9 harmonics simultaneously
     - Verify memory efficiency

3. **Implement cross-module tests**
   - [ ] `test_fft_to_phase_unwrapping()`
     - FFT → harmonic extraction → phase unwrapping
   - [ ] `test_corrections_to_fft()`
     - Corrections → FFT → verify spectrum changes

**Testing Step:** Run `pytest tests/test_integration.py --tb=long`

---

### Task 8: Implement Edge Case and Performance Tests
**Priority:** Medium  
**Estimated Time:** 3 hours

#### Subtasks:
1. **Create test_edge_cases.py**
   - [ ] Setup TestEdgeCases class
   - [ ] Setup TestNumericalStability class

2. **Implement edge case tests**
   - [ ] `test_zero_image_handling()`
     - All zeros input, verify no crashes
   - [ ] `test_single_pixel_image()`
     - 1x1 image through full pipeline
   - [ ] `test_nan_handling()`
     - Images with NaN values
   - [ ] `test_infinity_handling()`
     - Test division by zero scenarios

3. **Implement performance tests**
   - [ ] `test_large_image_processing()`
     - 2048x2048 image processing
     - Verify memory usage < threshold
   - [ ] `test_fft_different_sizes()`
     - Parametrized: [1, 2, 64, 128, 512, 1024]
   - [ ] `test_numerical_precision()`
     - Verify float32 vs float64 differences

**Testing Step:** Run `pytest tests/test_edge_cases.py --benchmark-only`

---

## Phase 4: Statistical and Morphological Tests

### Task 9: Implement Statistical Analysis Tests
**Priority:** Low  
**Estimated Time:** 3 hours

#### Subtasks:
1. **Create test_statistical_operations.py**
   - [ ] Import correlation and morphostructural modules
   - [ ] Setup TestStatisticalAnalysis class

2. **Implement statistical tests**
   - [ ] `test_gaussian_fitting()`
     - Generate known Gaussian, fit, verify parameters
   - [ ] `test_correlation_calculation()`
     - Known correlated data, verify coefficient
   - [ ] `test_confidence_ellipse()`
     - 2D Gaussian data, verify 95% confidence ellipse

3. **Implement morphological tests**
   - [ ] `test_region_extraction()`
     - Rectangle, ellipse, polygon regions
   - [ ] `test_pixel_wise_statistics()`
     - Mean, std calculation for regions

**Testing Step:** Run `pytest tests/test_statistical_operations.py`

---

## Phase 5: Validation and Documentation

### Task 10: Create Validation Suite
**Priority:** Low  
**Estimated Time:** 2 hours

#### Subtasks:
1. **Create test_validation.py**
   - [ ] Implement physics validation tests
   - [ ] Implement mathematical consistency tests

2. **Implement validation tests**
   - [ ] `test_physical_constraints()`
     - Absorption ≥ 0, Phase ∈ [-π, π]
   - [ ] `test_mathematical_consistency()`
     - FFT(IFFT(x)) ≈ x
   - [ ] `test_algorithm_determinism()`
     - Same input → same output

**Testing Step:** Run full test suite: `pytest tests/ -v --cov=src --cov-report=html`

---

### Task 11: Setup Continuous Integration
**Priority:** Low  
**Estimated Time:** 2 hours

#### Subtasks:
1. **Create GitHub Actions workflow**
   - [ ] Create `.github/workflows/tests.yml`
   - [ ] Configure Python versions [3.8, 3.9, 3.10, 3.11]
   - [ ] Setup test matrix for different OS

2. **Configure coverage reporting**
   - [ ] Setup codecov integration
   - [ ] Create coverage badges
   - [ ] Set minimum coverage threshold (80%)

3. **Create test documentation**
   - [ ] Create `tests/README.md`
   - [ ] Document test running procedures
   - [ ] Document how to add new tests

**Testing Step:** Push to GitHub and verify CI runs successfully

---

## Phase 6: Final Integration

### Task 12: Complete Test Suite Validation
**Priority:** High  
**Estimated Time:** 2 hours

#### Subtasks:
1. **Run complete test suite**
   - [ ] Run all tests with coverage
   - [ ] Generate HTML coverage report
   - [ ] Identify uncovered code paths

2. **Performance benchmarking**
   - [ ] Run performance tests
   - [ ] Document baseline metrics
   - [ ] Create performance regression tests

3. **Final documentation**
   - [ ] Update CLAUDE.md with test commands
   - [ ] Create troubleshooting guide
   - [ ] Document known limitations

**Testing Step:** 
```bash
# Final validation commands
pytest tests/ -v --cov=src --cov-report=term-missing
pytest tests/ --benchmark-only
pytest tests/ -v -m "not slow"  # Run fast tests only
```

---

## Summary

**Total Estimated Time:** 40 hours

**Task Dependencies:**
- Tasks 1-2 must be completed first (infrastructure)
- Tasks 3-6 can be done in parallel (core tests)
- Tasks 7-8 depend on 3-6 (integration)
- Tasks 9-11 can be done anytime after infrastructure
- Task 12 must be done last (validation)

**Success Criteria:**
- [ ] All tests passing
- [ ] Code coverage > 80%
- [ ] No performance regressions
- [ ] CI/CD pipeline functional
- [ ] Documentation complete
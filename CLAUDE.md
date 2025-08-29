# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHI (Spatial Harmonic Imaging) is a CLI software for multi-contrast X-ray imaging that produces high-resolution images in absorption, scattering, and differential phase modes. The software is designed for users new to the technique, including students and companies seeking effective data analysis tools.

## Key Commands

### Installation
```bash
./install.sh          # Install SHI as a system-wide CLI tool
```

### Main Operations
```bash
shi calculate -m <mask_period> [options]     # Execute the SHI method
shi morphostructural                         # Perform morphostructural analysis  
shi preprocessing                             # Correct angle alignment
shi clean                                     # Remove temporary files
```

### Common Usage Examples
```bash
# Test the installation
shi test

# Run SHI-2D method in current folder
shi calculate -m <mask_period> --all-2d

# Run SHI-CT method in current folder  
shi calculate -m <mask_period> --all-3d

# With specific images
shi calculate -m <mask_period> -i <path> -f <flat> -d <dark> -b <bright>
```

## Architecture

### Directory Structure
- **`/src`**: Core processing modules
  - `spatial_harmonics.py`: FFT-based harmonic analysis and phase extraction
  - `corrections.py`: Dark field, flat field, and bright field corrections
  - `angles_correction.py`: Optical component alignment correction
  - `correcting_stripes.py`: Detector stripe artifact removal
  - `crop_tk.py`: Interactive ROI selection using tkinter
  - `unwrapping_phase.py`: Multiple phase unwrapping algorithms
  - `distributions_fit.py`: Statistical distribution fitting
  - `directories.py`: Directory management utilities
  - `export.py`: Data export functionality
  - `utils.py`: Common utility functions

- **`/scripts`**: Advanced analysis tools
  - `morphostructural.py`: Combined scattering/absorption analysis
  - `morphos.py`: Morphological analysis
  - `correlation.py`: Image correlation analysis

- **`/cache`**: Stores pixel-wise analysis results
- **`/tmp`**: Temporary processing files (harmonics.pkl)
- **`/icons`**: UI icons for tkinter interfaces

### Data Processing Pipeline

1. **Input Validation**: Expects folder structure with sample images, dark images, flat images, and optional bright images
2. **Preprocessing**: Angle correction and stripe removal if needed
3. **Core SHI Processing**: 
   - FFT analysis to extract harmonics
   - Phase retrieval and unwrapping
   - Multi-contrast image generation (absorption, scattering, differential phase)
4. **Output**: Results saved to `Documents/CXI/CXI-DATA-ANALYSIS/`

### Key Processing Features

- **Phase Unwrapping Methods**: branch_cut, least_squares, quality_guided, min_lp, or default reliability-based
- **Image Corrections**: Dark field, flat field, bright field corrections with optional cropping
- **Angle Alignment**: Automatic detection and correction of optical component misalignment
- **Stripe Correction**: Removes detector artifacts that create false features

## Development Notes

### Dependencies
The project uses standard scientific Python libraries (numpy, scipy, scikit-image, tifffile). ImageJ is required for certain visualization features.

### Testing
No formal test suite found. Use `shi test` to verify installation.

### Important Considerations
- The software expects specific folder structures for experimental data (see README for acquisition scheme)
- Results are automatically saved to a fixed location in Documents folder
- Temporary files in `/tmp` can be cleaned with `shi clean` command
- The software uses pickle files for caching harmonic calculations

### Code Style
- Uses pathlib for file path operations
- Type hints in newer functions (Python 3.7+)
- Logging configured at INFO level
- Functions generally have comprehensive docstrings
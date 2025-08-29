# SHI (Spatial Harmonic Imaging) - Technical Documentation

## Project Overview

SHI is a sophisticated X-ray imaging analysis software that implements Spatial Harmonic Imaging (SHI), a multi-contrast imaging modality. The software processes X-ray images through Fourier analysis to extract three distinct contrast modes from a single dataset:

1. **Absorption contrast** - Traditional X-ray absorption imaging
2. **Scattering contrast** - Small-angle X-ray scattering information
3. **Differential phase contrast** - Phase shifts in the X-ray wavefront

## Core Architecture

### 1. Data Processing Pipeline

The software follows a systematic pipeline for image processing:

```
Raw Images → Preprocessing → FFT Analysis → Harmonic Extraction → Contrast Retrieval → Export
```

#### Stage 1: Preprocessing
- **Dark field correction**: Removes sensor noise by subtracting dark images (no X-ray exposure)
- **Flat field correction**: Normalizes detector response using reference images
- **Bright field correction**: Optional additional normalization step
- **Angle correction**: Aligns optical components to correct for misalignment
- **Stripe correction**: Removes detector artifacts that appear as vertical/horizontal stripes

#### Stage 2: Fourier Analysis
The core mathematical engine uses Fast Fourier Transform (FFT) to decompose images into frequency components:
- Converts spatial domain images to frequency domain
- Identifies harmonic peaks in the Fourier spectrum
- Extracts spatial frequency information based on mask periodicity

#### Stage 3: Harmonic Extraction
- **Zero-order harmonic (H00)**: Contains absorption information
- **First-order harmonics (H±1,0 and H0,±1)**: Contain phase gradient information
- **Higher-order harmonics**: Contain scattering information
- The software automatically identifies and classifies up to 9 harmonics based on their position relative to the main peak

#### Stage 4: Contrast Retrieval
From the extracted harmonics, three types of contrast are computed:
- **Absorption**: `log(1/|IFFT(H00)|)`
- **Scattering**: Computed from ratio of higher harmonics to main harmonic
- **Phase map**: Unwrapped phase from complex ratio of harmonics

### 2. Module Structure

#### Core Processing Modules (`/src`)

**`spatial_harmonics.py`** - Heart of the FFT processing
- Performs 2D FFT on input images
- Identifies and extracts harmonic components
- Implements phase unwrapping algorithms
- Computes differential phase contrast using Sobel filters or gradients

**`corrections.py`** - Image preprocessing
- Dark field subtraction
- Flat field division
- Bright field normalization
- Handles cropping and rotation corrections

**`angles_correction.py`** - Optical alignment
- Detects peaks in Fourier space
- Calculates misalignment angles
- Applies rotation corrections

**`correcting_stripes.py`** - Artifact removal
- Identifies periodic stripe artifacts
- Applies filtering to remove detector defects

**`unwrapping_phase.py`** - Advanced phase processing
- Multiple algorithms: branch cut, least squares, quality-guided
- Handles phase wrapping discontinuities

**`crop_tk.py`** - Interactive ROI selection
- Tkinter-based GUI for selecting regions of interest
- Saves crop coordinates for batch processing

**`directories.py`** - File management
- Creates standardized output directory structure
- Manages file paths and organization

**`utils.py`** - Helper functions
- Common utilities for file handling
- Data validation and conversion

#### Advanced Analysis Scripts (`/scripts`)

**`morphostructural.py`** - Combined analysis tool
- Interactive GUI for selecting regions in absorption/scattering images
- Performs pixel-wise correlation analysis
- Generates scatter plots and statistical distributions
- Fits Gaussian distributions to data
- Creates confidence ellipses for data clustering

**`correlation.py`** - Statistical analysis
- Computes correlation between absorption and scattering
- Statistical confidence calculations

**`morphos.py`** - Morphological analysis
- Shape and structure analysis of features

### 3. Data Flow

1. **Input Requirements**:
   - Sample images (actual X-ray images of the specimen)
   - Dark images (sensor readout with no X-rays)
   - Flat images (X-rays without sample)
   - Optional: Bright images (additional calibration)

2. **Processing Flow**:
   ```
   Input Images → Dark Correction → Flat Correction → FFT Transform
        ↓
   Harmonic Extraction → Contrast Calculation → Phase Unwrapping
        ↓
   Export Results (TIFF format with ImageJ compatibility)
   ```

3. **Output Structure**:
   ```
   Documents/CXI/CXI-DATA-ANALYSIS/
   └── [experiment_name]/
       ├── absorption/
       ├── scattering/
       ├── phase/
       └── phasemap/
   ```

### 4. Key Algorithms

#### FFT-based Harmonic Extraction
The software uses a grid period parameter (mask period) to identify harmonics in Fourier space. The main harmonic (H00) is located at the DC component, while higher harmonics appear at integer multiples of the fundamental frequency.

#### Phase Unwrapping
Multiple algorithms are implemented:
- **Goldstein's branch cut**: Identifies branch cuts to avoid phase discontinuities
- **Least squares**: Minimizes the difference between wrapped and unwrapped gradients
- **Quality-guided**: Uses quality maps to guide unwrapping path
- **Reliability-based**: Default algorithm sorting pixels by reliability

#### Differential Phase Contrast
Two methods for computing phase gradients:
- **Sobel filters**: Edge detection filters for robust gradient estimation
- **NumPy gradient**: Direct numerical differentiation

### 5. Command-Line Interface

The main entry point `shi.py` provides subcommands:

- **`shi calculate`**: Main processing command
  - `-m/--mask_period`: Required, defines the grid period
  - `--all-2d`: Process 2D images automatically
  - `--all-3d`: Process CT (3D) datasets
  - `--unwrap-phase`: Select phase unwrapping algorithm
  - `--average`: Apply averaging to reduce noise
  - `--export`: Enable data export

- **`shi morphostructural`**: Launch interactive analysis tool
  - Allows selection of regions for statistical analysis
  - Generates correlation plots between absorption and scattering

- **`shi preprocessing`**: Angle correction preprocessing
  - Detects and corrects optical component misalignment

- **`shi clean`**: Maintenance command
  - Removes temporary files in `/tmp` directory
  - Clears cached harmonic data

### 6. Caching System

The software uses pickle files to cache harmonic extraction parameters:
- Stored in `/tmp/harmonics.pkl`
- Contains boundaries and positions of harmonics
- Enables consistent processing across multiple images
- First image establishes harmonic positions, subsequent images reuse these positions

### 7. Interactive Features

**Morphostructural Analysis GUI**:
- Dual-panel display for absorption and scattering images
- Three selection tools: Rectangle, Ellipse, Polygon
- Real-time histogram display with Gaussian fitting
- Scatter plot generation with confidence ellipses
- Export functionality for analysis results

**Crop Tool**:
- Tkinter-based interface
- Visual selection of region of interest
- Saves coordinates for batch processing

## Technical Implementation Details

### Image Format
- Input: TIFF format (16-bit or 32-bit float)
- Output: 32-bit float TIFF with ImageJ metadata
- Preserves full dynamic range of processed data

### Performance Optimizations
- NumPy arrays for efficient computation
- FFT shift operations for centered frequency domain
- Masked arrays for selective pixel processing
- Batch processing capabilities for large datasets

### Error Handling
- Logging system for debugging (INFO level)
- Validation of input directory structure
- Graceful handling of missing calibration files
- Prevention of division by zero with epsilon values

## Use Cases

1. **Materials Science**: Analyzing internal structure of materials
2. **Biomedical Imaging**: Soft tissue contrast enhancement
3. **Non-destructive Testing**: Detecting defects and cracks
4. **Research**: Quantitative phase imaging studies

## Dependencies

- **NumPy**: Numerical computations and array operations
- **SciPy**: Statistical functions and signal processing
- **scikit-image**: Image processing algorithms and phase unwrapping
- **Matplotlib**: Visualization and plotting
- **tifffile**: TIFF image I/O with metadata support
- **PySide6**: Qt-based GUI components
- **pandas**: Data manipulation for statistical analysis
- **pathlib**: Modern file path handling

## Installation and Usage

The software is installed via `install.sh` which creates a system-wide symlink, making the `shi` command available globally. Users need Anaconda Python environment and ImageJ for full functionality.

The typical workflow:
1. Organize experimental data in required folder structure
2. Run preprocessing if needed (`shi preprocessing`)
3. Execute main analysis (`shi calculate -m [period]`)
4. Perform advanced analysis (`shi morphostructural`)
5. Results automatically saved to Documents folder

This architecture provides a robust, modular system for X-ray image analysis with both automated batch processing and interactive exploration capabilities.
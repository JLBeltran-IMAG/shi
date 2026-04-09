![SHI logo](docs/logo_shi.png)

# SHI: A Framework for Spatial Harmonic X-ray Imaging

**SHI** is a scientific framework for Spatial Harmonic Imaging (SHI), a multi-contrast X-ray imaging modality. It supports the complete workflow, including acquisition, reconstruction of absorption, scattering, and differential phase contrast, as well as postprocessing routines such as morphostructural analysis and CT reconstruction. It is intended for users new to the technique, including students and companies seeking practical tools for data acquisition and analysis.

---

## System Requirements

- **Operating System:** Linux (Ubuntu, Linux Mint, Debian)
- **Hardware Requirements:**
  - **Processor:** At least 2 GHz
  - **RAM:** Minimum 4 GB (8 GB or higher recommended for optimal performance)
  - **Disk Space:** At least 20 GB of free disk space for installation and data storage


## Installation

```bash
git clone https://github.com/JLBeltran-IMAG/shi.git 
cd shi
```

### Recommended (uv)

```bash
uv venv  
uv sync
source .venev/bin/activate
```

### Alternative (pip)

```bash
python3 -m venv .venv
source .venev/bin/activate
pip install .
```

### Verify installation:

```bash
shi --help
morphos --help
```

## Running SHI

The software provides three main command-line tools:

1. `acq` - Main tool for acquisition
2. `shi` - Main tool for SHI processing
3. `morphos` - Tool for morphostructural analysis

### SHI Processing

To see all available options for SHI processing:

```bash
shi calculate --help
```

Basic usage with automatic mode (2D):

```bash
shi calculate -m MASK_PERIOD --all-2d
```

Basic usage with automatic mode (3D):

```bash
shi calculate -m MASK_PERIOD --all-3d
```

To clean up temporary files:

```bash
shi clean --extra
```

### Morphostructural Analysis

The morphostructural analysis tool provides two main commands:

1. `analyze`: Run the morphostructural analysis

```bash
morphos analyze --left path/to/absorption.tif --right path/to/scattering.tif --contrast linear
```

Arguments for analyze:

- `--left`: Path to the absorption image

- `--right`: Path to the scattering/phase image

- `--contrast`: Contrast type (linear or log)

2. `clean`: Clean temporary and annotation files

```bash
# Clean temporary files
morphos clean --temp
```

## Citation

The authors kindly request that, if you use **SHI** in your research or work, you cite the following publication:

@article{Diaz2026,
 author = {Diaz, Jorge Luis Beltran and Korvink, Jan G. and Kunka, Danays},
 title = {SHI: a framework for spatial harmonic imaging},
 journal = {Scientific Reports},
 year = {2026},
 volume = {16},
 number = {1},
 pages = {4338},
 doi = {10.1038/s41598-026-37029-5},
 url = {https://doi.org/10.1038/s41598-026-37029-5},
 issn = {2045-2322}
}

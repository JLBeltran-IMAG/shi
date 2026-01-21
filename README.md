![Descripción de la imagen](docs/logo_shi.png)

# SHI: A CLI-software for Spatial Harmonic X-ray Imaging

**SHI: Spatial Harmonic Imaging** is a user-friendly software designed to facilitate Spatial Harmonic Imaging (SHI), a multi-contrast X-ray imaging modality. It produces high-resolution images in absorption, scattering, and differential phase modes very fast. The software is intended for users who are new to the technique, including students and companies seeking effective data analysis tools.

---

## System Requirements

- **Operating System:** Linux (Ubuntu, Linux Mint, Debian)
- **Hardware Requirements:**
  - **Processor:** At least 2 GHz
  - **RAM:** Minimum 4 GB (8 GB or higher recommended for optimal performance)
  - **Disk Space:** At least 20 GB of free disk space for installation and data storage

---

## Installation

### Installing Anaconda

To create an appropriate environment for Python and the necessary scientific libraries:

1. **Download Anaconda:**
   
   - Visit the official [Anaconda website](https://www.anaconda.com/products/distribution) and download the Linux version with Python 3.7 or higher.

2. **Install Anaconda:**
   
   - Open a terminal and navigate to the directory where the Anaconda installer was downloaded.
   
   - Run the installer with:
     
     ```bash
     bash Anaconda3-xxxx.xx-Linux-x86_64.sh
     ```
     
     *Replace `Anaconda3-xxxx.xx-Linux-x86_64.sh` with the actual filename.*

3. **Set Up the Environment:**
   
   - After installation, update your `PATH` by adding the following line at the end of your `.bashrc` file:
     
     ```bash
     export PATH="/home/your_username/anaconda3/bin:$PATH"
     ```
     
     *Replace `/home/your_username/anaconda3` with your actual installation path.*

4. **Verify Installation:**
   
   - Close and reopen the terminal, then run:
     
     ```bash
     conda --version
     ```
     
     This should display the installed version of Anaconda.

### Installing ImageJ

**ImageJ** is a widely used image processing software that complements the functionalities of SHI. Although ImageJ is an optional requirement, it is useful for visualizing the final results.

1. ```bash
   sudo apt install imagej
   ```

2. **Verify Installation:**
   
   - Launch ImageJ from the terminal to ensure it starts without issues.

---

### Installing SHI

Download the ZIP file, extract its contents, and run the application from the terminal:

```bash
pip install .
```

## Running SHI

The software provides two main command-line tools:

1. `shi` - Main tool for SHI processing
2. `morphos` - Tool for morphostructural analysis

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

## Contact Information

For additional support, to report issues, or to provide suggestions, please contact:

- **Author:** Jorge Luis Beltran Diaz and Danays Kunka

---

![Descripción de la imagen](docs/logo_shi.png)

# SHI: A CLI-software for Spatial Harmonic X-ray Imaging

**SHI: Spatial Harmonic Imaging** is a user-friendly software designed to facilitate Spatial Harmonic Imaging (SHI), a multi-contrast X-ray imaging modality. It produces high-resolution images in absorption, scattering, and differential phase modes within seconds per image. The software is intended for users who are new to the technique, including students and companies seeking effective data analysis tools.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Installing Anaconda](#installing-anaconda)
  - [Installing ImageJ](#installing-imagej)
  - [Installing SHI](#installing-shi)
- [Web Interface](#web-interface)
  - [Quick Start](#quick-start-web)
  - [Features](#features)
- [Using the Software](#using-the-software)
  - [Command Line Usage](#command-line-usage)
  - [Web Interface Usage](#web-interface-usage)
  - [Testing Examples](#testing-examples)
  - [Running Real Experiments](#running-real-experiments)
- [Advanced Features](#advanced-features)
  - [Averaging Tool](#averaging-tool)
  - [Scattering and Absorption Analysis Tool](#scattering-and-absorption-analysis-tool)
  - [Line Profile Plot Tool](#line-profile-plot-tool)
  - [Detector Stripes Correction Tool](#detector-stripes-correction-tool)
- [Support and Additional Resources](#support-and-additional-resources)
- [Contact Information](#contact-information)

---

## System Requirements

- **Operating System:** Linux (Ubuntu, CentOS, etc.), macOS, Windows (via WSL or Anaconda)
- **Python:** 3.7 or higher (recommended: 3.9+)
- **Hardware Requirements:**
  - **Processor:** At least 2 GHz (multi-core recommended for faster processing)
  - **RAM:** Minimum 8 GB (16 GB or higher recommended for large datasets)
  - **Disk Space:** At least 20 GB of free disk space for installation and data storage
- **Dependencies:** ImageJ (for visualization), Anaconda or pip for Python package management

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

3. **Follow the Installation Instructions:**
   
   - Accept the terms and conditions and choose an appropriate installation location (typically `/home/your_username/anaconda3`).

4. **Set Up the Environment:**
   
   - After installation, update your `PATH` by adding the following line at the end of your `.bashrc` file:
     
     ```bash
     export PATH="/home/your_username/anaconda3/bin:$PATH"
     ```
     
     *Replace `/home/your_username/anaconda3` with your actual installation path.*

5. **Verify Installation:**
   
   - Close and reopen the terminal, then run:
     
     ```bash
     conda --version
     ```
     
     This should display the installed version of Anaconda.

### Installing ImageJ

ImageJ is a widely used image processing software that complements SHI functionalities.

1. ```bash
   sudo apt install imagej
   ```

2. **Verify Installation:**
   
   - Launch ImageJ from the terminal to ensure it starts without issues.

---

### Installing SHI

#### Option 1: System Installation (Recommended)

Clone or download the SHI repository and install it globally:

```bash
# Clone the repository (or download and extract)
git clone <repository-url>
cd shi

# Install Python dependencies
pip install -r requirements.txt

# Install SHI as a system-wide CLI tool
./install.sh

# Verify installation
shi test
```

#### Option 2: Development Installation

For development or if you want to modify the source code:

```bash
# Install in development mode
pip install -e .

# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest -v
```

#### Option 3: Web Interface Only

If you only want to use the web interface:

```bash
# Install backend dependencies
cd shi-react-web/backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

**Note:** If using a USB distribution, don't remove the device while running the software. 

## Using the Software

SHI can be used in two ways:
1. **Command Line Interface (CLI)** - For expert users and automated workflows
2. **Web Interface** - For easy-to-use graphical interaction

### Command Line Usage

1. Open a terminal and navigate to the directory containing your data.

2. Run one of the main commands:
   
   ```bash
   # Basic SHI processing
   shi calculate -m <mask_period> --all-2d
   
   # With specific image paths
   shi calculate -m <mask_period> -i <images> -f <flat> -d <dark> -b <bright>
   
   # Additional processing options
   shi morphostructural --morphostructural
   shi preprocessing --stripes
   shi clean --clear-cache
   ```

3. After execution, check the folder `Documents/CXI/CXI-DATA-ANALYSIS` for the results.

### Web Interface Usage

1. Start the web application:
   ```bash
   # Start backend
   cd shi-react-web/backend
   python main.py
   
   # Start frontend (new terminal)
   cd shi-react-web/frontend
   npm start
   ```

2. Open http://localhost:3000 in your browser
3. Upload TIFF files and configure processing options through the web interface
4. Monitor processing progress and download results

### Testing Examples

After installing and ensuring that Anaconda and ImageJ work correctly, test the software functionality:

**CLI Testing:**
```bash
# Test installation
shi test

# Run complete test suite
pytest -v

# Run specific tests
pytest tests/test_spatial_harmonics_math.py
```

**Web Interface Testing:**
```bash
# Test backend API
cd shi-react-web/backend && python main.py
# Visit http://localhost:8000/docs for API documentation

# Test frontend
cd shi-react-web/frontend && npm test
```

These verify that the system processes test data and produces expected outputs.

### Running Real Experiments

For real experiments, configure the input directory as follow:

![Descripción de la imagen](docs/acq_scheme.png)

If the folder where you saved your experimental data has no the same structure above, the software will stop with error.

**Example configuration file (`experiment_config.txt`):**

The results will be saved in `Documents/CXI/CXI-DATA-ANALYSIS/foldername`.

---

## Advanced Features

The SHI software includes additional tools for advanced data processing, each implemented as separate scripts:

### Morphostructural Analysis Tool

Analyzes structural characteristics based on scattering and absorption data.

**CLI Usage:**
```bash
shi morphostructural --morphostructural
```
When prompted, select the two files corresponding to the absorption and scattering images.

**Web Interface:** Select "morphostructural" command and upload your analysis files through the web interface.


### Preprocessing and Stripe Correction

Corrects detector stripes and performs angle alignment corrections.

**CLI Usage:**
```bash
# Correct detector stripes
shi preprocessing --stripes

# Or use the dedicated script
python src/correcting_stripes.py
```

**Web Interface:** Select "preprocessing" command, enable "Correct Stripes", and upload your raw data.

A subfolder named `corrected_images` will be created containing the processed images.

---

## Web Interface

SHI includes a modern web application that provides the same functionality as the CLI tool through an intuitive graphical interface.

### Quick Start (Web) {#quick-start-web}

1. **Start the Backend:**
   ```bash
   cd shi-react-web/backend
   pip install -r requirements.txt
   python main.py
   ```
   Backend runs on: http://localhost:8000

2. **Start the Frontend:**
   ```bash
   cd shi-react-web/frontend
   npm install
   npm start
   ```
   Frontend runs on: http://localhost:3000

3. **Use the Application:**
   - Open http://localhost:3000 in your browser
   - Select a command (calculate, morphostructural, preprocessing, clean)
   - Upload your TIFF files using drag-and-drop
   - Configure processing options through the web forms
   - Submit and monitor progress in real-time
   - Download results when complete

### Features

- **Complete CLI Equivalence:** All shi.py commands available through web interface
- **Drag-and-Drop File Upload:** Automatic file categorization (sample, dark, flat, bright)
- **Real-time Progress Monitoring:** Live updates during processing
- **Job Management:** View, monitor, download, and delete processing jobs
- **Modern Interface:** Responsive design that works on desktop and mobile
- **Background Processing:** Long-running operations don't block the interface
- **Error Handling:** Clear error messages and validation

### API Access

The backend provides a complete REST API for programmatic access:
- API documentation: http://localhost:8000/docs (when backend is running)
- All CLI functionality accessible via HTTP endpoints
- JSON-based communication for easy integration

---

## Support and Additional Resources

- **Documentation:** Please refer to the complete documentation for detailed instructions on software configuration and usage.
- **Online Resources:** Visit forums and specialized websites for additional information on SHI and image processing techniques.
- **Updates:** Stay informed about new versions or improvements to the software.

---

## Contact Information

For additional support, to report issues, or to provide suggestions, please contact:

- **Author:** Jorge Luis Beltran Diaz and Danays Kunka

---

This README provides a summary of the key aspects of the user manual for CXI: Spatial Harmonic Imaging. For detailed instructions on each section, please refer to the complete documentation provided in the LaTeX file.

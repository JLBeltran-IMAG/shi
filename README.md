# SHI: A CLI-software for Spatial Harmonic X-ray Imaging

**SHI: Spatial Harmonic Imaging** is a user-friendly software designed to facilitate Spatial Harmonic Imaging (SHI), a multi-contrast X-ray imaging modality. It produces high-resolution images in absorption, scattering, and differential phase modes within seconds per image. The software is intended for users who are new to the technique, including students and companies seeking effective data analysis tools.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Installing Anaconda](#installing-anaconda)
  - [Installing ImageJ](#installing-imagej)
- [Using the Software](#using-the-software)
  - [Running from USB](#running-from-usb)
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

- **Operating System:** Linux (Ubuntu, CentOS, etc.)
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

The SHI: Spatial Harmonic Imaging software can be provided on a USB stick. You can run it directly from the USB stick without formal installation or copy the `shi` folder to any directory on your computer.

For installing the software, run on your terminal

```bash
./install.sh
```

If you are using a USB stick, don't remove the usb-device while running the software 

## Running SHI

1. Open a terminal and navigate to the directory containing the data to proccess.

2. Run on the terminal:
   
   ```bash
   shi ...(instructions)
   ```

3. After execution, check the folder `Documents/CXI/CXI-DATA-ANALYSIS` for the results.

### Testing Examples

After installing and ensuring that Anaconda and ImageJ work correctly, test the software functionality by running (missing test):

```bash
shi test
```

This verifies that the system processes the test data and produces the expected outputs.

### Running Real Experiments

For real experiments, configure the input directory as follow:

![Descripción de la imagen](docs/acq_scheme.png)

If the folder where you saved your experimental data has no the same structure above, the software will stop with error.

**Example configuration file (`experiment_config.txt`):**

The results will be saved in `Documents/CXI/CXI-DATA-ANALYSIS/foldername`.

---

## Advanced Features

The SHI software includes additional tools for advanced data processing, each implemented as separate scripts:

### Scattering and Absorption Analysis Tool

Analyzes structural characteristics based on scattering and absorption data.

- Run:
  
  ```bash
  shi morphostructural --morphostructural
  ```

- When prompted, select the two files corresponding to the absorption and scattering images.

### Line Profile Plot Tool

Generates line profile plots to enhance image details.

- Run:
  
  ```bash
  python plot_profile.py
  ```

- Select the image file to analyze when prompted.

### Detector Stripes Correction Tool

Corrects detector stripes that might introduce false features in the final images.

1. Run:
   
   ```bash
   python correcting_stripes.py
   ```

2. Select the folder containing all raw experimental data (input images, dark images, and flat images).

3. A subfolder named `no stripe` will be created in each subfolder of the selected directory.

4. Update the configuration file (`experiment_config.txt`) with the path to the new folder containing the corrected images.

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

Based on my analysis of the codebase, here's a comprehensive breakdown of how the SHI (Spatial Harmonic Imaging) application executes:

  1. Application Entry Point (shi.py:130-304)

  The application starts when a user runs:
  shi calculate --mask_period 10 --all-2d --images sample/ --flat flat/ --dark dark/ --bright bright/

  Key Components:
  - Parser Setup (shi.py:34-122): Configures command-line interface with subcommands (calculate, morphostructural, preprocessing, clean)
  - Argument Processing (shi.py:131-137): Extracts mask_period and unwrapping method from user input

  2. Image Path Detection & Validation (shi.py:140-161)

  The application determines image paths based on user mode:
  - Automatic Mode (--all-2d or --all-3d): Detects standard directory structure (sample/, dark/, flat/, bright/)
  - Manual Mode: Uses user-specified paths for each image type

  Directory Structure Expected:
  experiment/
  ├── sample/measurement_01/*.tif
  ├── dark/*.tif
  ├── flat/*.tif
  └── bright/*.tif

  3. Image Processing Loop (shi.py:162-271)

  For each measurement directory, the system performs:

  3.1 Initial Setup & ROI Cropping (shi.py:166-190)

  - File Validation: Ensures .tif files exist in measurement directory
  - Angle Correction Calculation: If --angle-after is enabled, calculates rotation angle from flat field images (angles_correction.py:184-185)
  - ROI Selection: Uses crop_tk.cropImage() to define Region of Interest interactively

  3.2 Dark Field & Bright Field Corrections (shi.py:192-218)

  The system applies preprocessing corrections:

  Dark Field Correction (corrections.py:42-91):
  corrected_image = raw_image - average_dark_image

  Bright Field Correction (corrections.py:175-216):
  corrected_image = dark_corrected_image / average_bright_image

  Images are saved to either corrected_images/ or crop_without_correction/ folders.

  3.3 Directory Structure Creation (shi.py:220-237)

  - Creates result directories using directories.create_result_subfolders()
  - Establishes output hierarchy: ~/Documents/CXI/CXI-DATA-ANALYSIS/measurement_name/

  4. Core SHI Processing (spatial_harmonics.py:587-634)

  4.1 Spatial Harmonics Extraction

  For each corrected image, the system performs:

  1. FFT Computation (spatial_harmonics.py:18-64):
  fft_image = np.fft.fftshift(np.fft.fft2(image))
  2. Harmonic Identification (spatial_harmonics.py:240-334):
    - Identifies main harmonic (0-order)
    - Extracts up to 8 higher-order harmonics using peak detection
    - Saves harmonic positions to tmp/harmonics.pkl for consistency
  3. Harmonic Classification (spatial_harmonics.py:175-238):
    - Categorizes harmonics as: vertical_positive/negative, horizontal_positive/negative, or diagonal combinations
    - Uses geometric analysis based on peak positions

  4.2 Contrast Retrieval (spatial_harmonics.py:528-584)

  For each harmonic, the system computes different contrast types:

  Absorption Contrast:
  absorption = np.log(1 / np.abs(ifft_harmonic))

  Scattering Contrast (spatial_harmonics.py:437-474):
  scattering = np.log(1 / np.abs(ratio))

  Phase Map (spatial_harmonics.py:385-434):
  phase_map = unwrap_phase(np.angle(ratio))

  Differential Phase (spatial_harmonics.py:477-526):
  diff_phase_horizontal = sobel_h(absorption)
  diff_phase_vertical = sobel_v(absorption)

  4.3 Results Export (directories.py:108-142)**

  Each contrast type is saved as TIFF files in organized directories:
  results/
  ├── absorption/*.tif
  ├── scattering/*.tif
  ├── phase/*.tif
  └── phasemap/*.tif

  5. Flat Field Processing & Corrections (shi.py:227-246)

  When flat field images are available:

  1. Flat Field SHI Processing: Applies same SHI analysis to flat field images
  2. Flat Averaging (directories.py:299-385): Creates averaged flat field references for each harmonic
  3. Flat Correction (corrections.py:122-173): Subtracts flat field artifacts from sample measurements

  6. Post-Processing Organization (shi.py:247-271)

  Based on the selected mode:

  2D Mode (--all-2d):
  - Averaging (directories.py:220-296): Combines multiple acquisitions into average images
  - Creates bidirectional contrast images combining horizontal and vertical components

  3D/CT Mode (--all-3d):
  - Directory Organization (directories.py:144-190): Sorts images by orientation for tomographic reconstruction
  - Organizes files into subdirectories by harmonic type (horizontal_positive, vertical_negative, etc.)

  Export (--export flag):
  - Copies final averaged results to a consolidated results/ folder
  - Enables easy access to processed images

  7. Key Mathematical Operations

  The core SHI algorithm relies on:

  1. Fourier Analysis: Decomposes grating-based interferometry patterns into spatial harmonics
  2. Phase Unwrapping: Resolves 2π phase ambiguities using various algorithms (skimage, branch_cut, least_squares, quality_guided)
  3. Differential Operators: Computes phase gradients using Sobel filters or numerical gradients
  4. Multi-contrast Retrieval: Extracts complementary information (absorption, scattering, phase) from the same measurement

  8. Data Flow Summary

  Raw Images → Corrections → FFT → Harmonic Extraction → Contrast Retrieval → Organization → Export
       ↓              ↓         ↓            ↓                    ↓              ↓          ↓
  [.tif files] → [crop+dark] → [freq domain] → [0th,±1st harmonics] → [abs,scat,phase] → [organized dirs] → [results/]

  The entire process transforms raw interferometric images into quantitative multi-contrast X-ray images suitable for materials characterization, with each step preserving mathematical rigor while maintaining user accessibility through the CLI
  interface.
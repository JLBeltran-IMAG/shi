import numpy as np
from skimage.restoration import unwrap_phase
from skimage.filters import sobel_h, sobel_v

import pickle
from pathlib import Path

from . import directories
from tifffile import imread
from . import unwrapping_phase as uphase


# Define the directory for temporary files using pathlib
tmp_files = Path(__file__).resolve().parents[1] / "tmp"
tmp_files.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe


def squared_fast_fourier_transform_linear_and_logarithmic(image, grid_period_projected, logarithmic_spectrum=False):
    """
    Computes the squared Fast Fourier Transform (FFT) of an image and returns either the linear or logarithmic spectrum.

    Parameters
    ----------
    image : np.ndarray
        2D array representing the input image.
    grid_period_projected : float
        The projected grid period used for computing the FFT frequencies.
    logarithmic_spectrum : bool, optional
        If True, the function returns the logarithmic spectrum of the FFT.
        If False, it returns the linear spectrum. Default is False.

    Returns
    -------
    wavevector_kx : np.ndarray
        1D array of wavevector components in the x-direction.
    wavevector_ky : np.ndarray
        1D array of wavevector components in the y-direction.
    fourier_transform_of_image : np.ndarray
        2D array of the FFT of the image. If `logarithmic_spectrum` is True,
        it returns the logarithm of the FFT amplitude.

    Notes
    -----
    This function performs the FFT shift to center the zero frequency component.
    The logarithmic spectrum is computed as `log10(1 + abs(FFT))` to prevent log(0).
    """
    # Image height and width
    image_height, image_width = image.shape

    # Spatial frequencies (Fourier space) for limiting the selected harmonics later
    wavevector_kx = np.fft.fftfreq(image_width, d=1 / grid_period_projected)
    wavevector_ky = np.fft.fftfreq(image_height, d=1 / grid_period_projected)

    # Calculate Fourier transform
    fourier_transform_of_image = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))

    # You have two options:
    #  - Log spectrum for better visualizing of Image Fourier Space
    if not logarithmic_spectrum:
        return wavevector_kx, wavevector_ky, fourier_transform_of_image

    #  - Linear spectrum for future calculations
    else:
        return wavevector_kx, wavevector_ky, np.log10(1 + np.abs(fourier_transform_of_image))


def zero_fft_region(array2d, top, bottom, left, right):
    """
    Sets a specific rectangular region of a 2D complex array to zero.

    This function is useful for filtering out certain frequency components
    in the Fourier domain of an image.

    Parameters:
    -----------
    array2d : np.ndarray
        A 2D NumPy array representing the Fourier transform of an image.
        It must be a complex-valued array.
    top : int
        The starting row index of the region to be zeroed.
    bottom : int
        The ending row index of the region to be zeroed (exclusive).
    left : int
        The starting column index of the region to be zeroed.
    right : int
        The ending column index of the region to be zeroed (exclusive).

    Returns:
    --------
    np.ndarray
        The modified 2D array with the specified region set to zero.
    """
    array2d[top:bottom, left:right] = np.complex128(0)
    return array2d


def extracting_harmonic(fourier_transform, ky_band_limit, kx_band_limit):
    """
    Extracts a rectangular region around the maximum harmonic component in a Fourier transform.

    This function locates the point with the highest magnitude in the Fourier transform and
    defines a rectangular region centered on that point, using the provided vertical and
    horizontal band limits.

    Parameters:
    -----------
    fourier_transform : np.ndarray
        A 2D NumPy array representing the Fourier transform of an image (complex values).
    ky_band_limit : int
        The vertical band limit (number of rows) to extract around the maximum component.
    kx_band_limit : int
        The horizontal band limit (number of columns) to extract around the maximum component.

    Returns:
    --------
    tuple
        A tuple containing:
            top_limit (int): The top boundary of the extracted region.
            bottom_limit (int): The bottom boundary of the extracted region.
            left_limit (int): The left boundary of the extracted region.
            right_limit (int): The right boundary of the extracted region.
            max_row_index (int): The row index of the maximum magnitude component.
            max_col_index (int): The column index of the maximum magnitude component.
    """
    # Find the index of the maximum magnitude in the Fourier transform
    max_row_index, max_col_index = np.unravel_index(np.argmax(np.abs(fourier_transform)), fourier_transform.shape)

    # Calculate boundaries and ensure they stay within the array dimensions
    top_limit = max(0, max_row_index - ky_band_limit)
    bottom_limit = min(fourier_transform.shape[0], max_row_index + ky_band_limit)
    left_limit = max(0, max_col_index - kx_band_limit)
    right_limit = min(fourier_transform.shape[1], max_col_index + kx_band_limit)

    return top_limit, bottom_limit, left_limit, right_limit, max_row_index, max_col_index


def identifying_harmonics_x1y1_higher_orders(x, y):
    """
    Identifies the harmonic diagonal based on the signs of x and y.

    Parameters:
    -----------
    x : numeric
        The x-coordinate.
    y : numeric
        The y-coordinate.

    Returns:
    --------
    str
        A string representing the harmonic diagonal:
          - "harmonic_diagonal_p1_p1" if x > 0 and y > 0.
          - "harmonic_diagonal_n1_p1" if x < 0 and y > 0.
          - "harmonic_diagonal_n1_n1" if x < 0 and y < 0.
          - "harmonic_diagonal_p1_n1" if x > 0 and y < 0.

    Raises:
    -------
    ValueError:
        If either x or y is zero, as the harmonic diagonal is undefined in that case.
    """
    if x == 0 or y == 0:
        raise ValueError("Invalid input: x and y must be non-zero to determine a harmonic diagonal.")

    if x > 0 and y > 0:
        return "harmonic_diagonal_p1_p1"
    elif x < 0 and y > 0:
        return "harmonic_diagonal_n1_p1"
    elif x < 0 and y < 0:
        return "harmonic_diagonal_n1_n1"
    elif x > 0 and y < 0:
        return "harmonic_diagonal_p1_n1"


def identifying_harmonic(main_harmonic_height, main_harmonic_width, harmonic_height, harmonic_width, angle_threshold=15):
    """
    Identifies the type of harmonic based on its position relative to a main harmonic.

    This function determines whether a harmonic peak is vertical, horizontal, or of a higher
    order by comparing its position to a main harmonic's position and analyzing the deviation angle.
    The angle_threshold parameter sets the threshold (in degrees) to determine if the deviation
    is predominantly vertical or horizontal.

    Parameters
    ----------
    main_harmonic_height : float
        The y-coordinate of the main harmonic peak.
    main_harmonic_width : float
        The x-coordinate of the main harmonic peak.
    harmonic_height : float
        The y-coordinate of the harmonic peak being analyzed.
    harmonic_width : float
        The x-coordinate of the harmonic peak being analyzed.
    angle_threshold : float, optional
        The threshold angle (in degrees) used to decide if a harmonic is primarily vertical or horizontal.
        Default is 15.

    Returns
    -------
    str
        The type of harmonic identified:
          - "harmonic_vertical_positive": Vertical harmonic above the main peak.
          - "harmonic_vertical_negative": Vertical harmonic below the main peak.
          - "harmonic_horizontal_positive": Horizontal harmonic to the right of the main peak.
          - "harmonic_horizontal_negative": Horizontal harmonic to the left of the main peak.
          - In other cases, the result of identifying_harmonics_x1y1_higher_orders(dx, dy).

    Notes
    -----
    If the deviation angle exceeds the angle_threshold, the function delegates the analysis to
    identifying_harmonics_x1y1_higher_orders(), which is assumed to handle higher order cases.
    """
    # Calculate differences between the harmonic and the main harmonic coordinates.
    dy = harmonic_height - main_harmonic_height
    dx = harmonic_width - main_harmonic_width
    abs_dy = abs(dy)
    abs_dx = abs(dx)

    # Case: Dominant vertical deviation.
    if abs_dy > abs_dx:
        # Calculate the deviation angle with respect to the vertical axis.
        deviation_angle = np.angle(complex(abs_dy, abs_dx), deg=True)
        if deviation_angle < angle_threshold:
            return "harmonic_vertical_positive" if dy > 0 else "harmonic_vertical_negative"
        else:
            return identifying_harmonics_x1y1_higher_orders(dx, dy)
    # Case: Dominant horizontal deviation.
    elif abs_dx > abs_dy:
        # Calculate the deviation angle with respect to the horizontal axis.
        deviation_angle = np.angle(complex(abs_dx, abs_dy), deg=True)
        if deviation_angle < angle_threshold:
            return "harmonic_horizontal_positive" if dx > 0 else "harmonic_horizontal_negative"
        else:
            return identifying_harmonics_x1y1_higher_orders(dx, dy)
    # Case: When vertical and horizontal deviations are equal.
    else:
        return identifying_harmonics_x1y1_higher_orders(dx, dy)


def spatial_harmonics_of_fourier_spectrum(fourier_transform, wavevector_ky, wavevector_kx, flat, limit_band=0.5):
    """
    Extracts spatial harmonics from a Fourier transform and either saves (if flat=True)
    or loads (if flat=False) the extraction limits to/from a pickle file.

    Parameters
    ----------
    fourier_transform : np.ndarray
        2D Fourier transform of an image.
    wavevector_ky : np.ndarray
        Array representing the y-component of the wavevector.
    wavevector_kx : np.ndarray
        Array representing the x-component of the wavevector.
    flat : bool
        If True, perform extraction and save the limits; if False, load the harmonic limits from file.
    limit_band : float, optional
        Band limit parameter to define the region for harmonics (default is 0.5).

    Returns
    -------
    tuple
        A tuple (harmonics, labels) where:
            harmonics : list of np.ndarray
                The extracted harmonic regions.
            labels : list of str
                The labels associated with each harmonic.
    """
    # Create a copy of the Fourier transform to avoid modifying the original.
    copy_of_fourier_transform = np.copy(fourier_transform)

    if flat:
        # Identify the main maximum harmonic (assumed to be near the center)
        max_index = np.argmax(np.abs(fourier_transform))
        main_max_h, main_max_w = np.unravel_index(max_index, fourier_transform.shape)

        # Determine band limits based on the wavevector arrays.
        ky_band_limit = np.argmin(np.abs(wavevector_ky - limit_band))
        kx_band_limit = np.argmin(np.abs(wavevector_kx - limit_band))

        harmonics = []
        labels = []
        export = {}

        # Extract the 0-order harmonic.
        top = main_max_h - ky_band_limit
        bottom = main_max_h + ky_band_limit
        left = main_max_w - kx_band_limit
        right = main_max_w + kx_band_limit

        harmonics.append(fourier_transform[top:bottom, left:right])
        label = "harmonic_00"
        labels.append(label)
        export[label] = [top, bottom, left, right, main_max_h, main_max_w]

        # Zero out the extracted region in the copy to avoid re-detection.
        zero_fft_region(copy_of_fourier_transform, top, bottom, left, right)

        # Extract higher-order harmonics (by default, 4 additional harmonics).
        for i in range(8):
            top, bottom, left, right, harmonic_h, harmonic_w = extracting_harmonic(
                copy_of_fourier_transform, ky_band_limit, kx_band_limit
            )
            harmonics.append(fourier_transform[top:bottom, left:right])
            label = identifying_harmonic(main_max_h, main_max_w, harmonic_h, harmonic_w)
            labels.append(label)
            export[label] = [top, bottom, left, right, harmonic_h, harmonic_w]
            zero_fft_region(copy_of_fourier_transform, top, bottom, left, right)

        # Save the harmonic extraction limits to a pickle file.
        pickle_path = tmp_files / "harmonics.pkl"
        try:
            with open(pickle_path, "wb") as harmonics_file:
                pickle.dump(export, harmonics_file)
        except Exception as e:
            raise IOError(f"Error writing harmonic limits to pickle file: {e}")

        return harmonics, labels

    else:
        harmonics = []
        labels = []
        pickle_path = tmp_files / "harmonics.pkl"
        try:
            with open(pickle_path, "rb") as harmonics_file:
                data = pickle.load(harmonics_file)
        except Exception as e:
            raise IOError(f"Error reading harmonic limits from pickle file: {e}")

        # Reconstruct harmonic regions using the stored limits.
        for label, limits in data.items():
            top, bottom, left, right, _, _ = limits
            harmonics.append(fourier_transform[top:bottom, left:right])
            labels.append(label)

        return harmonics, labels


def unwrapping_phase_gradient_operator(ratio, label, unwrap_algorithm="skimage"):
    """
    Unwraps the phase gradient of an image using either a scikit-image algorithm or NumPy's unwrap.

    The function first computes the phase map from the complex-valued 'ratio' using np.angle.
    It then determines the gradient axis based on the provided label ('horizontal' or 'vertical')
    and calculates the gradient of the phase map along that axis. The gradient is then wrapped into
    the principal interval using the arctan(tan(x)) operation. Finally, an unwrapping algorithm is
    applied to the wrapped phase gradient.

    Parameters:
        ratio (np.ndarray): Complex-valued input array from which the phase is computed.
        label (str): A string that should contain either "horizontal" or "vertical" to indicate
                     the direction for computing the gradient.
        unwrap_algorithm (str): The algorithm to use for phase unwrapping. Defaults to "skimage".
                                If not "skimage", NumPy's unwrap (applied twice) will be used.

    Returns:
        np.ndarray: The unwrapped phase gradient.
    """
    # Compute the phase map from the complex input
    phase_map = np.angle(ratio)

    # Determine the gradient axis based on the label
    if "horizontal" in label:
        gradient_axis = 0
    elif "vertical" in label:
        gradient_axis = 1
    else:
        raise ValueError("Label must contain either 'horizontal' or 'vertical'")

    # Compute the gradient of the phase map along the chosen axis
    phase_map_gradient = np.gradient(phase_map, axis=gradient_axis)

    # Wrap the gradient to the principal interval using arctan(tan(x))
    wrapped_phase_map_gradient = np.arctan(np.tan(phase_map_gradient))
    
    # Set the top-left element to zero as a specific correction
    wrapped_phase_map_gradient[0, 0] = 0

    # Unwrap the phase using the selected algorithm
    if unwrap_algorithm == "skimage":
        return unwrap_phase(wrapped_phase_map_gradient)
    else:
        # Apply NumPy's unwrap function along axis=1, then along axis=0
        return np.unwrap(np.unwrap(wrapped_phase_map_gradient, axis=1), axis=0)


def compute_phase_map(
    inverse_fourier_transform: np.ndarray,
    main_harmonic: np.ndarray,
    unwrap: str | None = None,
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    Computes the unwrapped phase map from the inverse Fourier transform and the main harmonic.

    Parameters:
    -----------
    inverse_fourier_transform : np.ndarray
        Array containing the inverse Fourier transform of the data.
    main_harmonic : np.ndarray
        Array containing the main harmonic in the Fourier domain.
    epsilon : float, optional
        Small value added to the denominator to avoid division by zero (default is 1e-12).

    Returns:
    --------
    unwrapped_phase_map : np.ndarray
        The unwrapped phase map.
    """
    # Compute the inverse Fourier transform of the main harmonic
    main_harmonic_ifft = np.fft.ifft2(main_harmonic)

    # Compute the ratio, avoiding division by zero by adding a small epsilon
    ratio = inverse_fourier_transform / (main_harmonic_ifft + epsilon)
    wrapped_phase = np.angle(ratio)

    # Unwrap the phase using the skimage algorithm
    if unwrap is None:
        unwrapped_phase_map = unwrap_phase(wrapped_phase, wrap_around=True)

    elif unwrap == "branch_cut":
        unwrapped_phase_map = uphase.goldstein_branch_cut_unwrap(wrapped_phase)

    elif unwrap == "least_squares":
        unwrapped_phase_map = uphase.ls_unwrap_phase(wrapped_phase)

    elif unwrap == "quality_guided":
        unwrapped_phase_map = uphase.quality_guided_unwrap(wrapped_phase)

    # elif unwrap == "min_lp":
    #     unwrapped_phase_map = uphase.min_lp_unwrap(wrapped_phase)

    else:
        raise ValueError("Unknown phase unwrapping algorithm")

    return unwrapped_phase_map


def compute_scattering(
    inverse_fourier_transform: np.ndarray,
    main_harmonic: np.ndarray,
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    Computes the scattering value from the inverse Fourier transform and the main harmonic.

    Parameters:
    -----------
    inverse_fourier_transform : np.ndarray
        Array containing the inverse Fourier transform of the data.
    main_harmonic : np.ndarray
        Array containing the main harmonic in the Fourier domain.
    epsilon : float, optional
        Small value added to the denominator to avoid division by zero (default is 1e-12).

    Returns:
    --------
    scattering_value : np.ndarray
        The computed scattering value.
    """
    # Compute the inverse Fourier transform of the main harmonic
    main_harmonic_ifft = np.fft.ifft2(main_harmonic)

    # Compute the ratio and avoid division by zero by adding epsilon
    ratio = inverse_fourier_transform / (main_harmonic_ifft + epsilon)

    # Get the absolute value of the ratio
    abs_ratio = np.abs(ratio)

    # Clip the absolute ratio to avoid taking the logarithm of values too close to zero
    abs_ratio = np.clip(abs_ratio, epsilon, None)

    # Compute the scattering as the natural logarithm of (1 / abs_ratio)
    scattering_value = np.log(1 / abs_ratio)

    return scattering_value


def differential_phase_contrast(image_main_harmonic, label, diff_operator="sobel"):
    """
    Computes the differential phase contrast of the main harmonic image using the specified operator.

    Parameters
    ----------
    image_main_harmonic : np.ndarray
        The input image representing the main harmonic.
    label : str
        The direction for the differential phase contrast.
        Expected values are "horizontal" or "vertical".
    diff_operator : str, optional
        The differential operator to use. Default is "sobel".
        Other accepted value is "gradient".

    Returns
    -------
    np.ndarray
        The computed differential phase contrast image.

    Raises
    ------
    ValueError
        If an unrecognized label or differential operator is provided.
    """
    # Convert label and operator to lowercase to ensure case-insensitive comparisons.
    label = label.lower()
    diff_operator = diff_operator.lower()

    if label == "horizontal":
        if diff_operator == "sobel":
            # Apply the horizontal Sobel operator.
            return sobel_h(image_main_harmonic)
        elif diff_operator == "gradient":
            # Use numpy's gradient along axis 0 for horizontal differentiation.
            return np.gradient(image_main_harmonic, axis=0)
        else:
            raise ValueError(f"Unknown differential operator: {diff_operator}")
    elif label == "vertical":
        if diff_operator == "sobel":
            # Apply the vertical Sobel operator.
            return sobel_v(image_main_harmonic)
        elif diff_operator == "gradient":
            # Use numpy's gradient along axis 1 for vertical differentiation.
            return np.gradient(image_main_harmonic, axis=1)
        else:
            raise ValueError(f"Unknown differential operator: {diff_operator}")
    else:
        raise ValueError(f"Unknown label for differential phase contrast: {label}")


def contrast_retrieval_individual_members(
    harmonic: np.ndarray,
    type_of_contrast: str,
    main_harmonic: np.ndarray = np.array([None, None, None, None]),
    unwrap: str | None = None,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Retrieves individual contrast members from a harmonic component.

    This function processes harmonic components to retrieve different types of contrast
    (absorption, scattering, or phase map) from the inverse Fourier transform of the input.

    Parameters
    ----------
    harmonic : ndarray
        The harmonic component in Fourier space to be processed.
    type_of_contrast : str
        The type of contrast to retrieve. Must be one of:
        - 'absorption': Computes absorption contrast
        - 'scattering': Computes scattering contrast
        - 'phasemap': Computes phase map contrast
    label : optional
        Label used for phase unwrapping when type_of_contrast is 'phasemap'.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.
    main_harmonic : any, optional
        Reserved for potential future use with main harmonic reference.

    Returns
    -------
    ndarray
        The retrieved contrast map according to the specified type_of_contrast.

    Raises
    ------
    ValueError
        If type_of_contrast is not one of 'absorption', 'scattering', or 'phasemap'.
    """
    # Compute the inverse Fourier transform of the harmonic component.
    ifft_harmonic = np.fft.ifft2(harmonic)

    # Avoid division by zero by adding a small constant to the magnitude.
    abs_ifft = np.abs(ifft_harmonic) + eps

    if type_of_contrast == "absorption":
        return np.log(1 / abs_ifft)

    elif type_of_contrast == "scattering":
        return compute_scattering(ifft_harmonic, main_harmonic)

    elif type_of_contrast == "phasemap":
        return compute_phase_map(ifft_harmonic, main_harmonic, unwrap)

    else:
        # Raise an error if the provided type_of_contrast is not recognized.
        raise ValueError(f"Unknown type_of_contrast: {type_of_contrast}")


def execute_SHI(path_to_images, path_to_result, mask_period, unwrap, flat):
    """
    Execute spatial harmonics analysis on a set of images.

    This function performs spatial harmonics analysis on a set of images and exports the results to the specified directory.

    Parameters
    ----------
    path_to_images : list of str or str
        A list of paths to the images for analysis or a directory path.
    path_to_result : str
        The path to the directory where the results will be exported.
    mask_period : int
        The period of the mask used in the analysis.

    """
    # Convert to Path objects if they're not already
    if isinstance(path_to_images, (str, Path)):
        path_to_images = Path(path_to_images)
    if isinstance(path_to_result, (str, Path)):
        path_to_result = Path(path_to_result)
    
    # If path_to_images is a directory, get all .tif files
    if isinstance(path_to_images, Path) and path_to_images.is_dir():
        image_paths = list(path_to_images.glob("*.tif"))
    elif isinstance(path_to_images, list):
        # If it's a list of paths already
        image_paths = path_to_images
    else:
        return
    
    for path in image_paths:
        try:
            img = imread(path).astype(np.float32)

            wavevector_kx, wavevector_ky, fft_img = squared_fast_fourier_transform_linear_and_logarithmic(
                img, mask_period
            )
            harmonics, labels = spatial_harmonics_of_fourier_spectrum(
                fft_img, wavevector_ky, wavevector_kx, flat
            )

            main_harmonic = harmonics[0]

            absorption = contrast_retrieval_individual_members(main_harmonic, type_of_contrast="absorption")
            directories.export_result_to(absorption, path.stem, path_to_result, "absorption")

            differential_phase_horizontal = differential_phase_contrast(absorption, label="horizontal")
            differential_phase_vertical = differential_phase_contrast(absorption, label="vertical")

            directories.export_result_to(differential_phase_horizontal, path.stem + "_" + "horizontal", path_to_result, "phase")
            directories.export_result_to(differential_phase_vertical, path.stem + "_" + "vertical", path_to_result, "phase")

            for idx in range(1, len(labels)):
                scattering = contrast_retrieval_individual_members(
                    harmonics[idx], type_of_contrast="scattering", main_harmonic=main_harmonic
                )
                directories.export_result_to(scattering, path.stem + "_" + labels[idx], path_to_result, "scattering")

                phasemap = contrast_retrieval_individual_members(
                    harmonics[idx], type_of_contrast="phasemap", main_harmonic=main_harmonic, unwrap=unwrap
                )
                directories.export_result_to(phasemap, path.stem + "_" + labels[idx], path_to_result, "phasemap")
        except Exception as e:
            # Keep this print as it's an error message
            print(f"Error processing {path}: {e}")

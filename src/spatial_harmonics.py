import numpy as np
from skimage.restoration import unwrap_phase
from skimage.filters import sobel_h, sobel_v

import pickle
from pathlib import Path

tmp_files = Path(__file__).resolve().parents[1].joinpath("tmp/")


def squared_fast_fourier_transform_linear_and_logarithmic(image, grid_period_projected, logarithmic_spectrum = False):
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

    Examples
    --------
    >>> image = np.random.rand(256, 256)
    >>> grid_period_projected = 1.0
    >>> wavevector_kx, wavevector_ky, fft_image = squared_fast_fourier_transform_linear_and_logarithmic(image, grid_period_projected)
    >>> wavevector_kx_log, wavevector_ky_log, fft_image_log = squared_fast_fourier_transform_linear_and_logarithmic(image, grid_period_projected, True)
    """
    image_height, image_width = image.shape

    wavevector_kx = np.fft.fftfreq(image_width, d = 1 / grid_period_projected)
    wavevector_ky = np.fft.fftfreq(image_height, d = 1 / grid_period_projected)

    fourier_transform_of_image = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))

    if not logarithmic_spectrum: return wavevector_kx, wavevector_ky, fourier_transform_of_image
    else: return wavevector_kx, wavevector_ky, np.log10(1 + np.abs(fourier_transform_of_image))

def making_zero_region_in_2darray_representing_fft_of_image(array2d, top_limit, botton_limit, left_limit, right_limit):
    array2d[top_limit : botton_limit, left_limit : right_limit] = complex(0, 0)
    return array2d

def extracting_harmonic(fourier_transform, ky_band_limit_of_harmonics, kx_band_limit_of_harmonics):
    next_index_of_maximun_height, next_index_of_maximun_width = np.unravel_index(np.argmax(np.abs(fourier_transform)), shape = fourier_transform.shape)
    top_limit = next_index_of_maximun_height - ky_band_limit_of_harmonics
    bottom_limit = next_index_of_maximun_height + ky_band_limit_of_harmonics
    left_limit = next_index_of_maximun_width - kx_band_limit_of_harmonics
    right_limit = next_index_of_maximun_width + kx_band_limit_of_harmonics
    return top_limit, bottom_limit, left_limit, right_limit, next_index_of_maximun_height, next_index_of_maximun_width

def identifying_harmonics_X1Y1_higher_orders(x, y):
    if x > 0 and y > 0: return "harmonic_diagonal_p1_p1"
    elif x < 0 and y > 0: return "harmonic_diagonal_n1_p1"
    elif x < 0 and y < 0: return "harmonic_diagonal_n1_n1"
    elif x > 0 and y < 0: return "harmonic_diagonal_p1_n1"
    else: return "Something wrong"

def identifying_harmonic(main_harmonic_height, main_harmonic_width, harmonic_height, harmonic_width):
    if abs(harmonic_height - main_harmonic_height) > abs(harmonic_width - main_harmonic_width):
        deviation_angle_of_peak = np.angle(complex(np.abs(harmonic_height - main_harmonic_height), np.abs(harmonic_width - main_harmonic_width)), deg = True)
        if deviation_angle_of_peak < 15:
            if harmonic_height - main_harmonic_height > 0:
                return "harmonic_vertical_positive"
            else:
                return "harmonic_vertical_negative"
        else: return identifying_harmonics_X1Y1_higher_orders(harmonic_width - main_harmonic_width, harmonic_height - main_harmonic_height)
    elif abs(harmonic_width - main_harmonic_width) > abs(harmonic_height - main_harmonic_height):
        deviation_angle_of_peak = np.angle(complex(np.abs(harmonic_width - main_harmonic_width), np.abs(harmonic_height - main_harmonic_height)), deg = True)
        if deviation_angle_of_peak < 15:
            if harmonic_width - main_harmonic_width > 0:
                return "harmonic_horizontal_positive"
            else:
                return "harmonic_horizontal_negative"
        else: return identifying_harmonics_X1Y1_higher_orders(harmonic_width - main_harmonic_width, harmonic_height - main_harmonic_height)
    else: return identifying_harmonics_X1Y1_higher_orders(harmonic_width - main_harmonic_width, harmonic_height - main_harmonic_height)

def identifying_harmonic_angle(main_harmonic_height,main_harmonic_width, harmonic_height, harmonic_width, tolerance = 25):
    deviation_angle_of_peak = np.angle(complex(harmonic_width - main_harmonic_width, harmonic_height - main_harmonic_height), deg = True)
    if deviation_angle_of_peak < tolerance and deviation_angle_of_peak > -tolerance:
        return "harmonic_horizontal_positive"
    elif deviation_angle_of_peak < 180 + tolerance and deviation_angle_of_peak > 180 - tolerance:
        return "harmonic_horizontal_negative"
    elif deviation_angle_of_peak < 90 + tolerance and deviation_angle_of_peak > 90 - tolerance:
        return "harmonic_vertical_positive"
    elif deviation_angle_of_peak > -(90 + tolerance) and deviation_angle_of_peak < -(90 - tolerance):
        return "harmonic_vertical_negative"
    else:
        return identifying_harmonics_X1Y1_higher_orders(harmonic_width - main_harmonic_width, harmonic_height - main_harmonic_height)


def spatial_harmonics_of_fourier_spectrum(fourier_transform, wavevector_ky, wavevector_kx, flat, limit_band = 0.5,):
    # copy of fourier transform to not change the original fourier transform
    copy_of_fourier_transform = np.copy(fourier_transform)

    if flat:
        # for selecting the maximun (is in the midle, logic) asign by code the maximun indexes
        index_of_main_maximun_height, index_of_main_maximun_width = np.unravel_index(np.argmax(np.abs(fourier_transform)), shape = fourier_transform.shape)

        # find the minimun indexes where harmonics of fourier space are limited by limited_band parameter, default = 0.5
        ky_band_limit_of_harmonics = np.argmin(np.abs(wavevector_ky - limit_band))
        kx_band_limit_of_harmonics = np.argmin(np.abs(wavevector_kx - limit_band))

        # create labels to each harmonic
        harmonics = list()
        labels = list()

        # create a list for [top_limit, bottom_limit, left_limit, right_limit] parameters to be read by images no flat
        export = dict()

        # Extracting 0-order harmonic
        top_limit = index_of_main_maximun_height - ky_band_limit_of_harmonics
        bottom_limit = index_of_main_maximun_height + ky_band_limit_of_harmonics
        left_limit = index_of_main_maximun_width - kx_band_limit_of_harmonics
        right_limit = index_of_main_maximun_width + kx_band_limit_of_harmonics

        # adding harmonic and label
        harmonics.append(fourier_transform[top_limit : bottom_limit, left_limit : right_limit])

        tmp_label = "harmonic_00"
        labels.append(tmp_label)

        # add the parameters [top_limit, bottom_limit, left_limit, right_limit] to export using pickle
        tmp_harmonic = [top_limit, bottom_limit, left_limit, right_limit, index_of_main_maximun_height, index_of_main_maximun_width]
        export[tmp_label] = tmp_harmonic

        making_zero_region_in_2darray_representing_fft_of_image(copy_of_fourier_transform, top_limit, bottom_limit, left_limit, right_limit)

        # plt.imshow(np.log(1 + np.abs(copy_of_fourier_transform)), cmap = "gray")
        # plt.show()

        #Extracting higher-order harmonics (dafault 1-order)

        for i in range(4):
            top_limit, bottom_limit, left_limit, right_limit, index_of_harmonic_height, index_of_harmonic_width = extracting_harmonic(copy_of_fourier_transform, ky_band_limit_of_harmonics, kx_band_limit_of_harmonics)

            tmp_harmonic = [top_limit, bottom_limit, left_limit, right_limit, index_of_harmonic_height, index_of_harmonic_width]
            harmonics.append(fourier_transform[top_limit : bottom_limit, left_limit : right_limit])

            tmp_label = identifying_harmonic(index_of_main_maximun_height, index_of_main_maximun_width, index_of_harmonic_height, index_of_harmonic_width)
            labels.append(tmp_label)

            export[tmp_label] = tmp_harmonic

            making_zero_region_in_2darray_representing_fft_of_image(copy_of_fourier_transform, top_limit, bottom_limit, left_limit, right_limit)
            # plt.imshow(np.log(1 + np.abs(copy_of_fourier_transform)), cmap = "gray")
            # plt.show()

        # once finished, export results to pickle file "*.pkl"
        with open('{}/harmonics.pkl'.format(tmp_files), 'wb') as harmonics_file:
            pickle.dump(export, harmonics_file)

        return harmonics, labels

    else:
        # create labels to each harmonic and read them
        harmonics = list()
        labels = list()

        with open('{}/harmonics.pkl'.format(tmp_files), 'rb') as harmonics_file:
            data = pickle.load(harmonics_file)

            for label, harmonic_limits in data.items():
                # for selecting the maximun (is in the midle, logic) asign by reading file of flat
                top_limit, bottom_limit, left_limit, right_limit, index_of_harmonic_height, index_of_harmonic_width = harmonic_limits

                harmonics.append(fourier_transform[top_limit : bottom_limit, left_limit : right_limit])
                labels.append(label)

                # # plt.imshow(np.log(1 + np.abs(copy_of_fourier_transform)), cmap = "gray")
                # # plt.show()

        return harmonics, labels


def unwrapping_phase_gradient_operator(inverse_fourier_transform, label, unwrap_algorithm = "skimage"):
    phase_map = np.angle(inverse_fourier_transform)

    if "horizontal" in label:
        phase_map_gradient = np.gradient(f = phase_map, axis = 0)
        wrapped_phase_map_gradient = np.arctan(np.tan(phase_map_gradient))
        wrapped_phase_map_gradient[0, 0] = 0

        if unwrap_algorithm == "skimage":
            return unwrap_phase(wrapped_phase_map_gradient)
        else:
            return np.unwrap(np.unwrap(wrapped_phase_map_gradient, axis = 1), axis = 0)

    if "vertical" in label:
        phase_map_gradient = np.gradient(f = phase_map, axis = 1)
        wrapped_phase_map_gradient = np.arctan(np.tan(phase_map_gradient))
        wrapped_phase_map_gradient[0, 0] = 0

        if unwrap_algorithm == "skimage":
            return unwrap_phase(wrapped_phase_map_gradient)
        else:
            return np.unwrap(np.unwrap(wrapped_phase_map_gradient, axis = 1), axis = 0)


def differential_phase_contrast(image_main_harmonic, label, diff_operator = "sobel"):
    if label == "horizontal":
        if diff_operator == "sobel":
            return sobel_h(image_main_harmonic)
        
        else:
            return np.gradient(f = image_main_harmonic, axis = 0)

    elif label == "vertical":
        if diff_operator == "sobel":
            return sobel_v(image_main_harmonic)
        
        else:
            return np.gradient(f = image_main_harmonic, axis = 1)

    else:
        pass


def contrast_retrieval_individual_members(harmonic, type_of_contrast, label = None):
    inverse_fourier_transform_of_harmonic = np.fft.ifft2(harmonic)
    if type_of_contrast == "absorption":
        return np.log(1 / np.abs(inverse_fourier_transform_of_harmonic))
    elif type_of_contrast == "scattering":
        return np.log(1 / np.abs(inverse_fourier_transform_of_harmonic))
    elif type_of_contrast == "phasemap":
        return unwrapping_phase_gradient_operator(inverse_fourier_transform_of_harmonic, label)
    else: return 0



import numpy as np
import spatial_harmonics as spatial_harmonic


def next_two_power_for_dimension_padding(image):
    image_height, image_width = image.shape
    max_dimension = np.max([image_height, image_width])
    new_dimension_to_pad_image = np.power(2, np.ceil(np.log2(max_dimension)))
    return int(new_dimension_to_pad_image)


def squared_fft(image):
    image_height, image_width = image.shape

    new_dimension_to_pad_image = next_two_power_for_dimension_padding(image)
    amount_of_zeros_in_height = int(new_dimension_to_pad_image - image_height)
    amount_of_zeros_in_width = int(new_dimension_to_pad_image - image_width)
    new_image_padded_with_zeros = np.pad(image, ((0, amount_of_zeros_in_height), (0, amount_of_zeros_in_width)), mode = "constant", constant_values = 0)

    fourier_transform_of_image = np.fft.fftshift(np.fft.fft2(new_image_padded_with_zeros.astype(np.float32)))

    return fourier_transform_of_image


def extracting_coordinates_of_peaks(image):
    fourier_transform = squared_fft(image.astype(np.float32))
    copy_of_fourier_transform = np.copy(fourier_transform)

    index_of_main_maximun_height, index_of_main_maximun_width = np.unravel_index(np.argmax(np.abs(fourier_transform)), shape = fourier_transform.shape)
    ky_band_limit_of_harmonics = 500
    kx_band_limit_of_harmonics = 500

    coordenates = list()

    # Extracting 0-order harmonic
    top_limit = index_of_main_maximun_height - ky_band_limit_of_harmonics
    bottom_limit = index_of_main_maximun_height + ky_band_limit_of_harmonics
    left_limit = index_of_main_maximun_width - kx_band_limit_of_harmonics
    right_limit = index_of_main_maximun_width + kx_band_limit_of_harmonics

    coordenates.append([index_of_main_maximun_height, index_of_main_maximun_width])

    spatial_harmonic.zero_fft_region(copy_of_fourier_transform, top_limit, bottom_limit, left_limit, right_limit)

    # plt.imshow(np.log(1 + np.abs(fourier_transform)), cmap = "gray")
    # plt.show()

    #Extracting higher-order harmonics (dafault 1-order)
    for i in range(4):
        top_limit, bottom_limit, left_limit, right_limit, index_of_harmonic_height, index_of_harmonic_width = spatial_harmonic.extracting_harmonic(copy_of_fourier_transform, ky_band_limit_of_harmonics, kx_band_limit_of_harmonics)

        coordenates.append([index_of_harmonic_height, index_of_harmonic_width])
        spatial_harmonic.zero_fft_region(copy_of_fourier_transform, top_limit, bottom_limit, left_limit, right_limit)
        # plt.imshow(np.log(1 + np.abs(copy_of_fourier_transform)), cmap = "gray")
        # plt.show()

    return coordenates


def quadrant_loc_sign(y, h, x, w, axes):
    sign = 0

    if axes == "y":
        if x > w and y < h: sign = 1
        elif x < w and y < h: sign = -1
        elif x < w and y > h: sign = 1
        elif x > w and y > h: sign = -1
        else: sign = 0

    elif axes == "x":
        if x > w and y > h: sign = 1
        elif x > w and y < h: sign = -1
        elif x < w and y < h: sign = 1
        elif x < w and y > h: sign = -1
        else: sign = 0

    else: sign = 0

    return sign


def calculating_angles_of_peaks_average(coords):
    main_harmonic_height, main_harmonic_width = coords[0]
    angles = list()
    sign = list()

    for i in range(1, len(coords)):
        height = abs(coords[i][0] - main_harmonic_height)
        width = abs(coords[i][1] - main_harmonic_width)
        if height > width:
            sign.append(quadrant_loc_sign(coords[i][0], main_harmonic_height, coords[i][1], main_harmonic_width, axes = "y"))
            angles.append(np.rad2deg(np.arctan2(width, height)))

        elif height < width:
            sign.append(quadrant_loc_sign(coords[i][0], main_harmonic_height, coords[i][1], main_harmonic_width, axes = "x"))
            angles.append(np.rad2deg(np.arctan2(height, width)))

    return np.mean(np.array(angles) * np.array(sign))



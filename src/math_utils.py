import numpy as np
from tifffile import imread, imwrite


def average_flat_harmonics(path_to_flat, type_of_contrast):
    """
    Generate average images from flat field harmonics.

    This function calculates the average images from flat field harmonics based on the provided type of contrast.

    Parameters
    ----------
    path_to_flat : str
        The path to the directory containing flat field images.
    type_of_contrast : str
        The type of contrast for which average images will be generated. Supported values are "scattering", "phase", and "absorption".

    Returns
    -------
    str or path-like or file-like
        The path to the directory where the average images are saved.

    Notes
    -----
    This function assumes that flat field images are stored in a directory structure where each type of contrast has its own subdirectory within the main flat field directory.

    Raises
    ------
    FileNotFoundError
        If the specified path to flat field directory does not exist.
    ValueError
        If an unsupported type of contrast is provided.

    """

    path_to_export = path_to_flat.joinpath(type_of_contrast, "average")

    if not path_to_export.exists(): path_to_export.mkdir()

    if type_of_contrast == "scattering" or type_of_contrast == "phasemap":
        list_of_flat = list(path_to_flat.joinpath(type_of_contrast).glob("*tif"))

        harmonic_horizontal_positive, harmonic_horizontal_negative, harmonic_vertical_positive, harmonic_vertical_negative = [], [], [], []

        for flat_images in list_of_flat:
            if "harmonic_horizontal_positive" in flat_images.stem: harmonic_horizontal_positive.append(flat_images)
            if "harmonic_horizontal_negative" in flat_images.stem: harmonic_horizontal_negative.append(flat_images)
            if "harmonic_vertical_positive" in flat_images.stem: harmonic_vertical_positive.append(flat_images)
            if "harmonic_vertical_negative" in flat_images.stem: harmonic_vertical_negative.append(flat_images)

        # checking dimensionality in harmonic_horizontal_positive
        if len(harmonic_horizontal_positive) == 1:
            harmonic_horizontal_positive = np.mean(imread(harmonic_horizontal_positive)[None, :, :], axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_horizontal_positive.tif"), harmonic_horizontal_positive, imagej = True)

        else:
            harmonic_horizontal_positive = np.mean(imread(harmonic_horizontal_positive), axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_horizontal_positive.tif"), harmonic_horizontal_positive, imagej = True)

        # checking dimensionality in harmonic_horizontal_negative
        if len(harmonic_horizontal_negative) == 1:
            harmonic_horizontal_negative = np.mean(imread(harmonic_horizontal_negative)[None, :, :], axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_horizontal_negative.tif"), harmonic_horizontal_negative, imagej = True)

        else:
            harmonic_horizontal_negative = np.mean(imread(harmonic_horizontal_negative), axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_horizontal_negative.tif"), harmonic_horizontal_negative, imagej = True)

        # checking dimensionality in harmonic_vertical_positive
        if len(harmonic_vertical_positive) == 1:
            harmonic_vertical_positive = np.mean(imread(harmonic_vertical_positive)[None, :, :], axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_vertical_positive.tif"), harmonic_vertical_positive, imagej = True)

        else:
            harmonic_vertical_positive = np.mean(imread(harmonic_vertical_positive), axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_vertical_positive.tif"), harmonic_vertical_positive, imagej = True)

        # checking dimensionality in harmonic_vertical_negative
        if len(harmonic_vertical_negative) == 1:
            harmonic_vertical_negative = np.mean(imread(harmonic_vertical_negative)[None, :, :], axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_vertical_negative.tif"), harmonic_vertical_negative, imagej = True)

        else:
            harmonic_vertical_negative = np.mean(imread(harmonic_vertical_negative), axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "harmonic_vertical_negative.tif"), harmonic_vertical_negative, imagej = True)


    elif type_of_contrast == "phase":
        list_of_flat = list(path_to_flat.joinpath(type_of_contrast).glob("*tif"))

        horizontal, vertical = [], []

        for flat_images in list_of_flat:
            if "horizontal" in flat_images.stem: horizontal.append(flat_images)
            if "vertical" in flat_images.stem: vertical.append(flat_images)

        # checking dimensionality in horizontal
        if len(horizontal) == 1:
            horizontal = np.mean(imread(horizontal)[None, :, :], axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "horizontal.tif"), horizontal, imagej = True)

        else:
            horizontal = np.mean(imread(horizontal), axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "horizontal.tif"), horizontal, imagej = True)

        # checking dimensionality in vertical
        if len(vertical) == 1:
            vertical = np.mean(imread(vertical)[None, :, :], axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "vertical.tif"), vertical, imagej = True)

        else:
            vertical = np.mean(imread(vertical), axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "vertical.tif"), vertical, imagej = True)

    elif type_of_contrast == "absorption":
        absorption_export = list(path_to_flat.joinpath(type_of_contrast).glob("*.tif"))

        # checking dimensionality in absorption_export
        if len(absorption_export) == 1:
            absorption_average = np.mean(imread(absorption_export)[None, :, :], axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "absorption.tif"), absorption_average, imagej = True)

        else:
            absorption_average = np.mean(imread(absorption_export), axis = 0, dtype = np.float32)
            imwrite("{}/{}".format(path_to_export, "absorption.tif"), absorption_average, imagej = True)

    else:
        print("Algo anda mal")

    return path_to_export






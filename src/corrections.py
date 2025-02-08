import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
from itertools import chain

import utils
import angles_correction

from skimage.transform import rotate

# Necesito manejar los Raiserror


def crop_without_corrections(path_to_images, crop = None, allow_crop = False, angle = 0.0):
    images = list(Path(path_to_images).glob("*.tif"))
    path_to_cropped_images = Path(path_to_images).joinpath("crop_without_correction")

    if not path_to_cropped_images.exists():
        path_to_cropped_images.mkdir()

    if allow_crop:
        y0, y1, x0, x1 = crop
    else:
        y0, x0 = 0, 0
        y1, x1 = images[0].shape

    for imgs in images:
        corrected_images = imread(imgs)

        cords = angles_correction.extracting_coordinates_of_peaks(corrected_images)
        deg = angles_correction.calculating_angles_of_peaks_average(cords)

        imwrite(path_to_cropped_images.joinpath("{}".format(imgs.name)), rotate(corrected_images, angle)[y0 : y1, x0 : x1].astype(np.float32), imagej = True)


def correct_darkfield(path_to_dark, path_to_images, crop = None, allow_crop = False, angle = 0.0):
    """
    Correct images using dark field correction.

    This function corrects images using dark field correction based on provided dark field images.

    Parameters
    ----------
    path_to_dark : str
        The path to the directory containing dark field images.
    path_to_images : str
        The path to the directory containing the images to be corrected.
    crop : tuple, optional
        A tuple specifying the crop region as (y0, y1, x0, x1), by default None.
    allow_crop : bool, optional
        A boolean indicating whether to allow cropping, by default False.

    """
    images = list(Path(path_to_images).glob("*.tif"))
    path_to_corrected_images = Path(path_to_images).joinpath("corrected_images")

    if not path_to_corrected_images.exists():
        path_to_corrected_images.mkdir()

    if allow_crop:
        y0, y1, x0, x1 = crop
    else:
        y0, y1, x0, x1 = 0, -1, 0, -1

    dark = imread([dark for dark in Path(path_to_dark).glob("*.tif")])

    # checking if there is only one element in the list
    if dark.ndim == 2:
        dark = dark[None, :, :]

    dark_image_average = np.mean(dark, axis = 0)
    for imgs in images:
        corrected_images = imread(imgs) - dark_image_average
        imwrite(path_to_corrected_images.joinpath("{}".format(imgs.name)), rotate(corrected_images, angle)[y0 : y1, x0 : x1].astype(np.float32), imagej = True)


def correct_flatmask(path_to_images, path_to_flat, type_of_contrast):
    """
    Correct flat mask in images.

    This function corrects the flat mask in images based on the specified type of contrast.

    Parameters
    ----------
    path_to_images : Path
        The path to the directory containing the images.
    path_to_flat : Path
        The path to the directory containing the flat field images.
    type_of_contrast : str
        The type of contrast for the correction, which can be "absorption", "scattering", or "phase".

    """
    path_to_file = path_to_images.joinpath(type_of_contrast)

    if type_of_contrast == "absorption":
        path_to_file_00 = list(path_to_file.glob("*.tif"))
        path_to_flat_file_00 = list(path_to_flat.glob("*.tif"))

        for path_to_files in path_to_file_00:
            division = imread(path_to_files) - imread(path_to_flat_file_00)
            imwrite("{}/{}_flatcorrected.tif".format(path_to_file.joinpath("flat_corrections"), path_to_files.stem), division.astype(np.float32), imagej = True)

    elif type_of_contrast == "scattering":
        path_to_corrected_file_00 = imread(list(path_to_file.joinpath("..", "absorption", "flat_corrections").glob("*.tif")))

        if path_to_corrected_file_00.ndim == 2:
            path_to_corrected_file_00 = path_to_corrected_file_00[None, :, :]

        avg_path_to_corrected_file_00 = np.mean(path_to_corrected_file_00, axis = 0)

        img_file_horizontal_scatt, img_file_vertical_scatt = utils.separate_orientation_lists(list(path_to_images.joinpath(type_of_contrast).glob("*.tif")))
        flat_img_file_horizontal_scatt, flat_img_file_vertical_scatt = utils.separate_orientation_lists(list(path_to_flat.glob("*.tif")))

        doble_data_horizontal_scatt = utils.join_harmonics_with_flat(img_file_horizontal_scatt, flat_img_file_horizontal_scatt)
        doble_data_vertical_scatt = utils.join_harmonics_with_flat(img_file_vertical_scatt, flat_img_file_vertical_scatt)

        for path_to_files in chain(doble_data_horizontal_scatt, doble_data_vertical_scatt):
            division = imread(path_to_files[0]) - imread(path_to_files[1]) - avg_path_to_corrected_file_00
            imwrite("{}/{}_flatcorrected.tif".format(path_to_file.joinpath("flat_corrections"), path_to_files[0].stem), division.astype(np.float32), imagej = True)

    elif type_of_contrast == "phase":
        img_file_horizontal_phase, img_file_vertical_phase = utils.separate_orientation_lists(list(path_to_images.joinpath(type_of_contrast).glob("*.tif")))
        flat_img_file_horizontal_phase, flat_img_file_vertical_phase = utils.separate_orientation_lists(list(path_to_flat.glob("*.tif")))

        doble_data_horizontal_phase = utils.join_harmonics_with_flat(img_file_horizontal_phase, flat_img_file_horizontal_phase)
        doble_data_vertical_phase = utils.join_harmonics_with_flat(img_file_vertical_phase, flat_img_file_vertical_phase)

        for path_to_files in chain(doble_data_horizontal_phase, doble_data_vertical_phase):
            division = imread(path_to_files[0]) - imread(path_to_files[1])
            imwrite("{}/{}_flatcorrected.tif".format(path_to_file.joinpath("flat_corrections"), path_to_files[0].stem), division.astype(np.float32), imagej = True)

    elif type_of_contrast == "phasemap":
        img_file_horizontal_phasemap, img_file_vertical_phasemap = utils.separate_orientation_lists(list(path_to_images.joinpath(type_of_contrast).glob("*.tif")))
        flat_img_file_horizontal_phasemap, flat_img_file_vertical_phasemap = utils.separate_orientation_lists(list(path_to_flat.glob("*.tif")))

        doble_data_horizontal_phasemap = utils.join_harmonics_with_flat(img_file_horizontal_phasemap, flat_img_file_horizontal_phasemap)
        doble_data_vertical_phasemap = utils.join_harmonics_with_flat(img_file_vertical_phasemap, flat_img_file_vertical_phasemap)

        for path_to_files in chain(doble_data_horizontal_phasemap, doble_data_vertical_phasemap):
            division = imread(path_to_files[0]) - imread(path_to_files[1])
            imwrite("{}/{}_flatcorrected.tif".format(path_to_file.joinpath("flat_corrections"), path_to_files[0].stem), division.astype(np.float32), imagej = True)





def correct_brightfield(path_to_bright, path_to_images):
    """
    Correct bright field in images.

    This function corrects the bright field in images based on the provided bright field images.

    Parameters
    ----------
    path_to_bright : Path
        The path to the directory containing the bright field images.
    path_to_images : Path
        The path to the directory containing the images to be corrected.

    """
    images = list(Path(path_to_images).glob("*.tif"))
    path_to_corrected_images = Path(path_to_images).joinpath("corrected_images")

    if not path_to_corrected_images.exists():
        path_to_corrected_images.mkdir()

    bright_image_average = np.mean([imread(bright) for bright in Path(path_to_bright).glob("*.tif")], axis = 0)
    for imgs in images:
        corrected_images = imread(imgs) / bright_image_average * np.mean(bright_image_average)
        imwrite(path_to_corrected_images.joinpath("{}".format(imgs.name)), corrected_images.astype(np.float32), imagej = True)



# Manejo de excepciones ---> Probar mas adelante:
# try, except, else and finally
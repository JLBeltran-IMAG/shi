import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
from itertools import chain

import utils

from skimage.transform import rotate

# Necesito manejar los Raiserror

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
            imwrite(
                "{}/{}_flatcorrected.tif".format(
                    path_to_file.joinpath("flat_corrections"),
                    path_to_files.stem),
                division.astype(np.float32),
                imagej = True
                )

    elif type_of_contrast == "scattering":
        (
            img_file_horizontal_scatt,
            img_file_vertical_scatt
            ) = utils.separate_orientation_lists(
                list(path_to_images.joinpath(type_of_contrast).glob("*.tif"))
                )
        (
            flat_img_file_horizontal_scatt,
            flat_img_file_vertical_scatt
            ) = utils.separate_orientation_lists(
                list(path_to_flat.glob("*.tif"))
                )

        doble_data_horizontal_scatt = utils.join_harmonics_with_flat(
            img_file_horizontal_scatt,
            flat_img_file_horizontal_scatt
            )
        doble_data_vertical_scatt = utils.join_harmonics_with_flat(
            img_file_vertical_scatt,
            flat_img_file_vertical_scatt
            )

        for path_to_files in chain(doble_data_horizontal_scatt, doble_data_vertical_scatt):
            division = imread(path_to_files[0]) - imread(path_to_files[1])
            imwrite(
                "{}/{}_flatcorrected.tif".format(
                    path_to_file.joinpath("flat_corrections"),
                    path_to_files[0].stem),
                division.astype(np.float32),
                imagej = True
                )

        path_to_corrected_file_00 = list(
            path_to_file.joinpath(
                "..",
                "absorption",
                "flat_corrections"
                ).glob("*.tif")
            )

        path_to_flatcorrected_files_horizontal = [
            path_to_h for path_to_h in path_to_file.joinpath("flat_corrections").glob("*.tif")
            if "horizontal" in path_to_h.stem
            ]
        path_to_flatcorrected_files_vertical = [
            path_to_v for path_to_v in path_to_file.joinpath("flat_corrections").glob("*.tif")
            if "vertical" in path_to_v.stem
            ]

        doble_data_abs_scatt_h = utils.join_corresponding_absorption_with(
            path_to_corrected_file_00,
            path_to_flatcorrected_files_horizontal
            )
        doble_data_abs_scatt_v = utils.join_corresponding_absorption_with(
            path_to_corrected_file_00,
            path_to_flatcorrected_files_vertical
            )

        for path_to_files in doble_data_abs_scatt_h:
            division = imread(path_to_files[1]) - imread(path_to_files[0])
            imwrite(
                "{}/{}_corrected.tif".format(
                    path_to_file.joinpath("abs_corrections", "horizontal"),
                    path_to_files[1].stem
                    ),
                division.astype(np.float32),
                imagej = True
                )

        for path_to_files in doble_data_abs_scatt_v:
            division = imread(path_to_files[1]) - imread(path_to_files[0])
            imwrite(
                "{}/{}_corrected.tif".format(
                    path_to_file.joinpath("abs_corrections", "vertical"),
                    path_to_files[1].stem),
                division.astype(np.float32),
                imagej = True
                )


    elif type_of_contrast == "phase":
        (
            img_file_horizontal_phase,
            img_file_vertical_phase
            ) = utils.separate_orientation_lists(
                list(path_to_images.joinpath(type_of_contrast).glob("*.tif"))
                )
        (
            flat_img_file_horizontal_phase,
            flat_img_file_vertical_phase
            ) = utils.separate_orientation_lists(
                list(path_to_flat.glob("*.tif"))
                )

        doble_data_horizontal_phase = utils.join_harmonics_with_flat(
            img_file_horizontal_phase,
            flat_img_file_horizontal_phase
            )
        doble_data_vertical_phase = utils.join_harmonics_with_flat(
            img_file_vertical_phase,
            flat_img_file_vertical_phase
            )

        for path_to_files in doble_data_horizontal_phase:
            division = imread(path_to_files[0]) - imread(path_to_files[1])
            imwrite(
                "{}/{}_flatcorrected.tif".format(
                    path_to_file.joinpath("flat_corrections", "horizontal"),
                    path_to_files[0].stem),
                division.astype(np.float32),
                imagej = True
                )

        for path_to_files in doble_data_vertical_phase:
            division = imread(path_to_files[0]) - imread(path_to_files[1])
            imwrite(
                "{}/{}_flatcorrected.tif".format(
                    path_to_file.joinpath("flat_corrections", "vertical"),
                    path_to_files[0].stem),
                division.astype(np.float32),
                imagej = True
                )


    elif type_of_contrast == "phasemap":
        img_file_horizontal_phasemap, img_file_vertical_phasemap = utils.separate_orientation_lists(list(path_to_images.joinpath(type_of_contrast).glob("*.tif")))
        flat_img_file_horizontal_phasemap, flat_img_file_vertical_phasemap = utils.separate_orientation_lists(list(path_to_flat.glob("*.tif")))

        doble_data_horizontal_phasemap = utils.join_harmonics_with_flat(img_file_horizontal_phasemap, flat_img_file_horizontal_phasemap)
        doble_data_vertical_phasemap = utils.join_harmonics_with_flat(img_file_vertical_phasemap, flat_img_file_vertical_phasemap)

        for path_to_files in chain(doble_data_horizontal_phasemap, doble_data_vertical_phasemap):
            division = imread(path_to_files[0]) - imread(path_to_files[1])
            imwrite("{}/{}_flatcorrected.tif".format(path_to_file.joinpath("flat_corrections"), path_to_files[0].stem), division.astype(np.float32), imagej = True)

        # path_to_corrected_file_00 = list(path_to_file.joinpath("..", "absorption", "corrections").glob("*.tif"))
        # path_to_flatcorrected_files = list(path_to_file.joinpath("flat_corrections").glob("*.tif"))

        # doble_data_abs_phasemap = utils.join_corresponding_absorption_with(path_to_corrected_file_00, path_to_flatcorrected_files)

        # for path_to_files in doble_data_abs_phasemap:
        #     division = imread(path_to_files[1]) - imread(path_to_files[0])
        #     imwrite("{}/{}_corrected.tif".format(path_to_file.joinpath("corrections"), path_to_files[0].stem), division.astype(np.float32), imagej = True)



# Manejo de excepciones ---> Probar mas adelante:
# try, except, else and finally
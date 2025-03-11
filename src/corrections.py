import numpy as np
from skimage.transform import rotate
from tifffile import imread, imwrite
from pathlib import Path
from itertools import chain
import logging
import utils
import angles_correction


def crop_without_corrections(path_to_images, crop=None, allow_crop=False, angle=0.0):
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

        imwrite(
            path_to_cropped_images.joinpath("{}".format(imgs.name)),
            rotate(corrected_images, angle)[y0:y1, x0:x1].astype(np.float32),
            imagej=True,
        )


def correct_darkfield(path_to_dark, path_to_images, crop=None, allow_crop=False, angle=0.0):
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

    dark_image_average = np.mean(dark, axis=0)
    for imgs in images:
        corrected_images = imread(imgs) - dark_image_average
        imwrite(
            path_to_corrected_images.joinpath("{}".format(imgs.name)),
            rotate(corrected_images, angle)[y0:y1, x0:x1].astype(np.float32),
            imagej=True,
        )


def process_flat_correction(image_path, flat_path, output_dir):
    """
    Reads an image and its corresponding flat field image, performs flat correction by subtraction,
    and writes the corrected image to the output directory.

    Parameters:
    -----------
    image_path : Path
        Path to the image file.
    flat_path : Path
        Path to the flat field image file.
    output_dir : Path
        Directory where the corrected image will be saved.
    """
    try:
        image = imread(image_path)
        flat = imread(flat_path)
        corrected = image - flat

        # Ensure the output directory exists
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{image_path.stem}_flatcorrected.tif"

        imwrite(output_file, corrected.astype(np.float32), imagej=True)

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")


def correct_flatmask(path_to_images, path_to_flat, type_of_contrast) -> None:
    """
    Corrects the flat mask in images based on the specified contrast type.

    Parameters:
    -----------
    path_to_images : Path
        Directory containing the images.
    path_to_flat : Path
        Directory containing the flat field images.
    type_of_contrast : str
        The contrast type for correction. Supported values are:
        "absorption", "scattering", "phase", and "phasemap".

    For 'absorption' contrast, if the number of flat files differs from the number of image files,
    the flat files are applied in a cyclic manner (using modulo indexing).

    For 'scattering', 'phase', and 'phasemap', images are separated by orientation and paired with
    the corresponding flat images using utility functions.
    """
    images_dir = path_to_images / type_of_contrast
    output_dir = images_dir / "flat_corrections"

    if type_of_contrast == "absorption":
        image_files = list(images_dir.glob("*.tif"))
        flat_path = list(path_to_flat.glob("*.tif"))

        # Check if the flat_path list is empty
        if not flat_path:
            raise ValueError("No flat files found in the specified flat directory.")

        # If the number of flat files differs from the number of images,
        # use modulo indexing to assign a flat file to each image.
        for img_path in image_files:
            process_flat_correction(img_path, flat_path, output_dir)

    elif type_of_contrast in {"scattering", "phase", "phasemap"}:
        image_files = list(images_dir.glob("*.tif"))
        flat_files = list(path_to_flat.glob("*.tif"))

        # Pair images with corresponding flat images using a utility function
        paired = utils.join_harmonics_with_flat(image_files, flat_files)

        for pair in paired:
            # Each pair should contain at least two paths: (image_path, flat_path)
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                process_flat_correction(pair[0], pair[1], output_dir)
            else:
                raise ValueError(f"Invalid pair format: {pair}")
    else:
        raise ValueError(f"Unsupported contrast type: {type_of_contrast}")


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
    path_to_corrected_bright = Path(path_to_bright).joinpath("corrected_images")
    path_to_corrected_images = Path(path_to_images).joinpath("corrected_images")

    images = list(Path(path_to_corrected_images).glob("*.tif"))

    if not path_to_corrected_images.exists():
        path_to_corrected_images.mkdir()

    bright = imread([bright for bright in Path(path_to_corrected_bright).glob("*.tif")])

    # checking if there is only one element in the list
    if bright.ndim == 2:
        bright = bright[None, :, :]

    bright_image_average = np.mean(bright, axis=0)

    for imgs in images:
        img_array = imread(imgs).astype(np.float32)
        result_div = np.copy(img_array).astype(np.float32)

        corrected_images = np.divide(img_array, bright_image_average, out=result_div, where=bright_image_average != 0)

        imwrite(
            path_to_corrected_images.joinpath("{}".format(imgs.name)),
            corrected_images.astype(np.float32),
            imagej=True
        )


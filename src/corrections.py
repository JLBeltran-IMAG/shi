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

    if type_of_contrast.lower() == "absorption":
        image_files = sorted(images_dir.glob("*.tif"))
        flat_path = sorted(path_to_flat.glob("*.tif"))

        # Check if the flat_path list is empty
        if not flat_path:
            raise ValueError("No flat files found in the specified flat directory.")

        # If the number of flat files differs from the number of images,
        # use modulo indexing to assign a flat file to each image.
        for img_path in image_files:
            process_flat_correction(img_path, flat_path, output_dir)

    elif type_of_contrast in {"scattering", "phase", "phasemap"}:
        image_files = sorted(images_dir.glob("*.tif"))
        flat_files = sorted(path_to_flat.glob("*.tif"))

        # Separate images by orientation (assuming the utility returns two lists: horizontal and vertical)
        image_horizontal, image_vertical = utils.separate_orientation_lists(image_files)
        flat_horizontal, flat_vertical = utils.separate_orientation_lists(flat_files)

        # Pair images with corresponding flat images using a utility function
        paired_horizontal = utils.join_harmonics_with_flat(image_horizontal, flat_horizontal)
        paired_vertical = utils.join_harmonics_with_flat(image_vertical, flat_vertical)

        for pair in chain(paired_horizontal, paired_vertical):
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
    images = list(Path(path_to_images).glob("*.tif"))
    path_to_corrected_images = Path(path_to_images).joinpath("corrected_images")

    if not path_to_corrected_images.exists():
        path_to_corrected_images.mkdir()

    bright_image_average = np.mean([imread(bright) for bright in Path(path_to_bright).glob("*.tif")], axis=0)
    for imgs in images:
        corrected_images = imread(imgs) / bright_image_average * np.mean(bright_image_average)
        imwrite(path_to_corrected_images.joinpath("{}".format(imgs.name)), corrected_images.astype(np.float32), imagej=True)


# Manejo de excepciones ---> Probar mas adelante:
# try, except, else and finally

from tifffile import imread, imwrite
from pathlib import Path
from datetime import datetime
import platform


def create_filedate(path_to_file):
    """
    Get the creation date for a file.

    This function retrieves the creation date for a file specified by the provided path.

    Parameters
    ----------
    path_to_file : str or path-like or file-like
        The path to the file for which the creation date will be retrieved.

    Returns
    -------
    str
        The creation date of the file in the format "YYYY-MM-DD".

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    """
    if platform.system() == "Windows": return datetime.fromtimestamp(path_to_file.stat().st_ctime).strftime("%Y-%m-%d")
    else: return datetime.fromtimestamp(path_to_file.stat().st_mtime).strftime("%Y-%m-%d")


def separate_orientation_lists(harmonics_list):
    """
    Separate harmonics into two lists based on their orientation.

    This function separates a list of harmonics into two lists according to whether they are horizontally
    or vertically oriented.

    Parameters
    ----------
    harmonics_list : list
        A list of Path objects representing file locations. 
        Each Path object corresponds to a file that contains either "horizontal" or "vertical" in its name. 
        The file names can be obtained using the "stem" attribute of the Path object.
        (See more information: https://docs.python.org/3/library/pathlib.html)

    Returns
    -------
    tuple of two lists
        The first list contains all horizontal harmonics.
        The second list contains all vertical harmonics.

    Notes
    -----
    The function expects that the filenames of the harmonics indicate their orientation, i.e.,
    contain either "horizontal" or "vertical".

    """
    horizontal_data = [harmonics for harmonics in harmonics_list if "horizontal" in harmonics.stem]
    vertical_data = [harmonics for harmonics in harmonics_list if "vertical" in harmonics.stem]
    return horizontal_data, vertical_data


def create_corrections_folder(path, ct_folders):
    """
    Create a folder named "corrections" at the specified location.

    This function creates a folder named "corrections" at the specified location. If the folder already exists, nothing will happen.

    Parameters
    ----------
    path : Path
        Path object representing the location where the "corrections" folder will be created.

    Returns
    -------
    path_to_corrections : Path
        Path object representing the location of the "corrections" folder.

    Notes
    -----
    If the folder already exists, this function does nothing.

    """
    directory_names = [names for names in path.iterdir() if "flat" not in names.name]
    for correction_folders in directory_names:
        path_to_corrections = correction_folders.joinpath("flat_corrections")
        if not path_to_corrections.exists():
            path_to_corrections.mkdir(parents = True, exist_ok = True)
    
    # path_phasemap_ct y scattering correction folders
    if ct_folders:
        path_scattering_cth = path.joinpath("scattering", "abs_corrections", "horizontal")
        path_scattering_ctv = path.joinpath("scattering", "abs_corrections", "vertical")
        path_scattering_cth.mkdir(parents = True, exist_ok = True)
        path_scattering_ctv.mkdir(parents = True, exist_ok = True)

        path_phasemap_cth = path.joinpath("phasemap", "abs_corrections", "horizontal")
        path_phasemap_ctv = path.joinpath("phasemap", "abs_corrections", "vertical")
        path_phasemap_cth.mkdir(parents = True, exist_ok = True)
        path_phasemap_ctv.mkdir(parents = True, exist_ok = True)

        path_phase_cth = path.joinpath("phase", "flat_corrections", "horizontal")
        path_phase_ctv = path.joinpath("phase", "flat_corrections", "vertical")
        path_phase_cth.mkdir(parents = True, exist_ok = True)
        path_phase_ctv.mkdir(parents = True, exist_ok = True)

    return path


def export_average_scatt_phase_images(path_to_images, type_of_contrast):
    """
    Export averaged phase and scattering contrast images to the corrections folder.

    This function exports phase and scattering contrast images averaged into the corrections folder created with the function "creating_folder_for_correction".

    Parameters
    ----------
    path_to_images : Path
        Path object representing the location of the images.
    type_of_contrast : str
        Specifies the type of contrast and the folder where the averaged images will be exported.

    Notes
    -----
    This function reads the images horizontally and calculates the mean:
    (imgs_horizontal_positive + imgs_horizontal_negative) / 2
    (imgs_vertical_positive + imgs_vertical_negative) / 2

    """
    path_to_export = path_to_images.joinpath(type_of_contrast, "corrections")

    corrections_images = list(path_to_export.glob("*.tif"))
    horizontal, vertical = separate_orientation_lists(corrections_images)

    horizontal_images = [imread(images) for images in horizontal]
    vertical_images = [imread(images) for images in vertical]

    image1 = (horizontal_images[0] + horizontal_images[1]) / 2
    image2 = (vertical_images[0] + vertical_images[1]) / 2
    image3 = (image1 + image2) / 2

    imwrite(path_to_export.joinpath("{} y.tif".format(type_of_contrast)), image1)
    imwrite(path_to_export.joinpath("{} x.tif".format(type_of_contrast)), image2)
    imwrite(path_to_export.joinpath("bi-directional {}.tif".format(type_of_contrast)), image3)


def join_harmonics_with_flat(harmonics_list, flat_harmonics_list):
    """
    Join harmonics images with their corresponding flat images.

    This function merges two lists into a list of tuples, pairing each element from the harmonics list with its corresponding element from the flat harmonics list.

    Parameters
    ----------
    harmonics_list : list
        List of paths to the images of each harmonic.
    flat_harmonics_list : list
        List of paths to the images of each flat harmonic.

    Returns
    -------
    list of tuples
        Each tuple contains two elements:
        - The first element corresponds to an image harmonic.
        - The second element corresponds to the flat image harmonic.

    Notes
    -----
    The function assumes that each element in the harmonics list has the same filename as its corresponding element in the flat harmonics list.

    """
    doble_data = list()
    for images in harmonics_list:
        doble_data.extend([(images, flat) for flat in flat_harmonics_list if flat.stem in images.stem])
    return doble_data


def join_corresponding_absorption_with(abs_list, type_of_contrast_list):
    doble_data = list()
    for images in abs_list:
        doble_data.extend([(images, type_of) for type_of in type_of_contrast_list 
                           if alphanumeric_chars_in_filename(type_of.stem)
                           in alphanumeric_chars_in_filename(images.stem)])

    return doble_data


def alphanumeric_chars_in_filename(filename):
    return "".join(list(filter(str.isdigit, filename)))



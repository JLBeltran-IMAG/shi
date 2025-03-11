import numpy as np
from tifffile import imwrite
from pathlib import Path
import shutil

import utils


def create_result_directory(folders_for_result="", sample_folder_name=""):
    """
    Create a directory structure for exporting analysis results.

    This function creates a directory structure for exporting analysis results. 
    It creates a main directory based on the provided folder names and a sample folder name, 
    with subdirectories for "absorption", "scattering", "phase" and "phasemap" results.

    Parameters
    ----------
    folders_for_result : str, optional
        A string specifying the main directory for the results, by default "".
    sample_folder_name : str, optional
        A string specifying the sample folder name, by default "".

    Returns
    -------
    Path
        Path object representing the main directory for exporting results.

    """
    path_to_results = Path.home().joinpath("Documents/CXI/CXI-DATA-ANALYSIS/{}/{}".format(folders_for_result, sample_folder_name))
    path_to_results.mkdir(parents = True, exist_ok = True)
    path_to_results.joinpath("absorption").mkdir(parents = True, exist_ok = True)
    path_to_results.joinpath("scattering").mkdir(parents = True, exist_ok = True)
    path_to_results.joinpath("phase").mkdir(parents = True, exist_ok = True)
    path_to_results.joinpath("phasemap").mkdir(parents = True, exist_ok = True)
    return path_to_results


def create_result_subfolders(directory_for_files, folders_for_result="", sample_folder_name=""):
    """
    Read files from an experiment folder and create subfolders for results.

    This function reads files from a specified directory and creates subfolders for exporting results. 
    It can create a main directory based on the provided folder names and a sample folder name.

    Parameters
    ----------
    directory_for_files : str
        The path to the directory containing the experiment files.
    folders_for_result : str, optional
        A string specifying the main directory for the results, by default "".
    sample_folder_name : str, optional
        A string specifying the sample folder name, by default "".

    Returns
    -------
    tuple
        A tuple containing two elements:
        - A list of Path objects representing the files read from the directory.
        - A list of Path objects representing the main directory and subdirectories for exporting results.

    """

    path_to_files = [x for x in Path(directory_for_files).glob("*.tif") if x.is_file()]
    path_to_results = list()

    if folders_for_result:
        path_to_results = create_result_directory(folders_for_result = folders_for_result, sample_folder_name = sample_folder_name)

    else:
        for tif_files_in_path in path_to_files:
            tiffiles_names, tiffiles_date_creation = tif_files_in_path.stem, utils.create_filedate(tif_files_in_path)
            path_to_results.append(create_result_directory(folders_for_result = tiffiles_date_creation, sample_folder_name = tiffiles_names))
    
    return path_to_files, path_to_results


def export_result_to(image_to_save, filename, path, type_of_contrast):
    """
    Export TIFF files to a default directory.

    This function exports a TIFF image to a default directory specified by the provided path and type of contrast.

    Parameters
    ----------
    image_to_save : ndarray
        The image data to be saved.
    filename : str
        The name of the TIFF file to be saved.
    path : str
        The path to the directory where the TIFF file will be saved.
    type_of_contrast : str
        The type of contrast for the image.

    """
    path_to_save_images = Path(path)

    if not path_to_save_images.exists():
        path_to_save_images.mkdir()

    if type_of_contrast in ["absorption", "scattering", "phase", "phasemap"]:
        if filename:
            path_to_file = path_to_save_images.joinpath(type_of_contrast, "{}.tif".format(filename))
            imwrite(path_to_file, image_to_save.astype(np.float32), imagej = True)
        else: print("Missing filename")
    else: print("Please, define the right type of contrast")


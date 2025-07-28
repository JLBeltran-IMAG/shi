import numpy as np
import tifffile as ti
from pathlib import Path
import shutil
import itertools


def create_result_directory(
    result_folder: str = "",
    sample_folder: str = ""
) -> Path:
    """
    Create a directory structure for exporting analysis results.

    This function creates a directory structure for exporting analysis results. 
    It creates a main directory based on the provided folder names and a sample folder name, 
    with subdirectories for "absorption", "scattering", "phase" and "phasemap" results.

    Parameters
    ----------
    result_folder : str, optional
        A string specifying the main directory for the results, by default "".
    sample_folder : str, optional
        A string specifying the sample folder name, by default "".

    Returns
    -------
    Path
        Path object representing the main directory for exporting results.
    """
    base_path = Path.home() / "Documents" / "CXI" / "CXI-DATA-ANALYSIS"

    result_path = base_path / result_folder / sample_folder
    result_path.mkdir(parents=True, exist_ok=True)

    for subdir in ["absorption", "scattering", "phase", "phasemap"]:
        (result_path / subdir).mkdir(parents=True, exist_ok=True)

    return result_path


def create_result_subfolders(
    file_dir: str,
    result_folder: str = "",
    sample_folder: str = ""
) -> tuple[list[Path], Path]:
    """
    Read files from an experiment folder and create subfolders for results.

    This function reads files from a specified directory and creates subfolders for exporting results. 
    It can create a main directory based on the provided folder names and a sample folder name.

    Parameters
    ----------
    file_dir : str
        The path to the directory containing the experiment files.
    result_folder : str, optional
        A string specifying the main directory for the results, by default "".
    sample_folder : str, optional
        A string specifying the sample folder name, by default "".

    Returns
    -------
    tuple
        A tuple containing two elements:
        - A list of Path objects representing the files read from the directory.
        - A list of Path objects representing the main directory and subdirectories for exporting results.
    """
    path_to_files = [x for x in Path(file_dir).glob("*.tif") if x.is_file()]

    if result_folder:
        result_path = create_result_directory(result_folder, sample_folder)
    else:
        result_path = create_result_directory()

    return path_to_files, result_path


def create_corrections_folder(path: Path) -> Path:
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
    directory_names = [names for names in path.iterdir() if "flat" not in names.name and "results" not in names.name]
    for correction_folders in directory_names:
        path_to_corrections = correction_folders.joinpath("flat_corrections")
        if not path_to_corrections.exists():
            path_to_corrections.mkdir(parents=True, exist_ok=True)

    return path


def export_result_to(
    image_to_save: np.ndarray,
    filename: str,
    path: Path,
    type_of_contrast: str
) -> None:
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
            path_to_file = path_to_save_images / type_of_contrast / "{}.tif".format(filename)
            ti.imwrite(path_to_file, image_to_save.astype(np.float32), imagej = True)
        else: print("Missing filename")
    else: print("Please, define the right type of contrast")


def organize_dir(path_to_files: Path, type_of_contrast: str) -> None:
    if type_of_contrast == "absorption":
        pass

    elif type_of_contrast in {"scattering", "phasemap"}:
        output_dir_list = (
            "horizontal_positive",
            "vertical_positive",
            "horizontal_negative",
            "vertical_negative",
            "diagonal_p1_p1",
            "diagonal_p1_n1",
            "diagonal_n1_p1",
            "diagonal_n1_n1",
        )

        for output_dir in output_dir_list:
            outdir = path_to_files / output_dir
            outdir.mkdir(parents=True, exist_ok=True)

            list_of_orientation = [
                path_to for path_to in path_to_files.glob("*.tif")
                if output_dir in path_to.stem
            ]

            for orientation in list_of_orientation:
                shutil.move(orientation, outdir)

    elif type_of_contrast == "phase":
        output_dir_list = ("horizontal", "vertical")

        for output_dir in output_dir_list:
            outdir = path_to_files.joinpath(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)

            list_of_orientation = [
                path_to for path_to in path_to_files.glob("*.tif")
                if output_dir in path_to.stem
            ]

            for orientation in list_of_orientation:
                shutil.move(orientation, outdir)
        pass

    else:
        raise ValueError(f"Unknown type_of_contrast: {type_of_contrast}")


def export_results(path):
    """
    Export all TIFF files found in the 'absorption', 'scattering', 'phase', and 'phasemap'
    directories (within the 'flat_corrections/average' structure) to a 'results' directory
    located in the given base path.

    Parameters:
        path (Path): The base directory where the files are located and where the 'results'
                     directory will be created.
    """
    # Create the destination directory for the results if it doesn't exist
    destination_directory = path / "results"
    destination_directory.mkdir(parents=True, exist_ok=True)

    # Define the categories to search for TIFF files
    categories = ["absorption", "scattering", "phase", "phasemap"]

    # Create an iterator that chains together all .tif files found in each category's
    # 'flat_corrections/average' subdirectory
    file_paths = itertools.chain(
        *( (path / category / "flat_corrections" / "average").glob("*.tif") for category in categories )
    )

    # Copy each file to the destination directory
    for file_path in file_paths:
        shutil.copy2(file_path, destination_directory)


def averaging(
    path_to_files: Path,
    type_of_contrast: str
) -> None:
    path_to_export = path_to_files.joinpath("average")
    path_to_export.mkdir(parents = True, exist_ok = True)

    if type_of_contrast == "absorption":
        imgs = ti.imread(list(Path(path_to_files).glob("*.tif")))

        if imgs.ndim == 2:
            imgs = imgs[None, :, :]

        avg = np.mean(imgs, axis = 0)
        ti.imwrite("{}/absorption.tif".format(path_to_export), avg)

    elif type_of_contrast == "phase":
        horizontal = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "horizontal" in x.stem])
        if horizontal.ndim == 2:
            horizontal = horizontal[None, :, :]

        vertical = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "vertical" in x.stem])
        if vertical.ndim == 2:
            vertical = vertical[None, :, :]

        avg_horizontal_positive = np.mean(horizontal, axis = 0)
        avg_vertical_positive = np.mean(vertical, axis = 0)

        bidirectional_positive = (avg_horizontal_positive + avg_vertical_positive) / 2

        ti.imwrite("{}/horizontal_phase.tif".format(path_to_export), avg_horizontal_positive)
        ti.imwrite("{}/vertical_phase.tif".format(path_to_export), avg_vertical_positive)
        ti.imwrite("{}/bidirectional_phase.tif".format(path_to_export), bidirectional_positive)

    elif type_of_contrast == "scattering":
        imgs_horizontal = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "horizontal" in x.stem])
        avg_horizontal = np.mean(imgs_horizontal, axis = 0)
        imgs_vertical = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "vertical" in x.stem])
        avg_vertical = np.mean(imgs_vertical, axis = 0)
        avg_scatt = (avg_horizontal + avg_vertical) / 2

        ti.imwrite("{}/horizontal_scattering.tif".format(path_to_export), avg_horizontal)
        ti.imwrite("{}/vertical_scattering.tif".format(path_to_export), avg_vertical)
        ti.imwrite("{}/bidirectional_scattering.tif".format(path_to_export), avg_scatt)

    elif type_of_contrast == "phasemap":
        imgs_horizontal1 = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "horizontal_negative" in x.stem])
        if imgs_horizontal1.ndim == 2:
            imgs_horizontal1 = imgs_horizontal1[None, :, :]
        avg_horizontal1 = np.mean(imgs_horizontal1, axis = 0)

        imgs_horizontal2 = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "horizontal_positive" in x.stem])
        if imgs_horizontal2.ndim == 2:
            imgs_horizontal2 = imgs_horizontal2[None, :, :]
        avg_horizontal2 = np.mean(imgs_horizontal2, axis = 0)

        imgs_vertical1 = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "vertical_negative" in x.stem])
        if imgs_vertical1.ndim == 2:
            imgs_vertical1 = imgs_vertical1[None, :, :]
        avg_vertical1 = np.mean(imgs_vertical1, axis = 0)

        imgs_vertical2 = ti.imread([x for x in Path(path_to_files).glob("*.tif") if "vertical_positive" in x.stem])
        if imgs_vertical2.ndim == 2:
            imgs_vertical2 = imgs_vertical2[None, :, :]
        avg_vertical2 = np.mean(imgs_vertical2, axis = 0)

        avg_horizontal = (avg_horizontal1 - avg_horizontal2) / 2
        avg_vertical = (avg_vertical1 - avg_vertical2) / 2
        avg_phasemap = (avg_horizontal + avg_vertical) / 2

        ti.imwrite("{}/horizontal_phasemap.tif".format(path_to_export), avg_horizontal)
        ti.imwrite("{}/vertical_phasemap.tif".format(path_to_export), avg_vertical)
        ti.imwrite("{}/bidirectional_phasemap.tif".format(path_to_export), avg_phasemap)

    else:
        print("Write the right contrast")



def average_flat_harmonics(path_to_flat: Path, type_of_contrast: str) -> Path:
    """
    Generate average images from flat field harmonics.

    This function calculates the average images from flat field harmonics based on the provided type of contrast.
    Supported values for type_of_contrast are "scattering", "phasemap", "phase", and "absorption".

    Parameters
    ----------
    path_to_flat : Path
        The path to the directory containing flat field images.
    type_of_contrast : str
        The type of contrast for which average images will be generated.

    Returns
    -------
    Path
        The path to the directory where the average images are saved.
    """
    # Create export directory (e.g. <path_to_flat>/<type_of_contrast>/average)
    export_dir = path_to_flat / type_of_contrast / "average"
    export_dir.mkdir(parents=True, exist_ok=True)

    def average_images(file_list):
        """
        Helper function to compute the average image from a list of files.
        If only one file is provided, it returns its image content as float32.
        Otherwise, it computes the pixelwise mean across all images.
        """
        if len(file_list) == 1:
            return ti.imread(file_list[0]).astype(np.float32)
        else:
            images = [ti.imread(f) for f in file_list]
            return np.mean(images, axis=0, dtype=np.float32)

    if type_of_contrast in ("scattering", "phasemap"):
        flat_files = list((path_to_flat / type_of_contrast).glob("*tif"))
        # Initialize a dictionary to store files by channel (including diagonal channels)
        channels = {
            "harmonic_horizontal_positive": [],
            "harmonic_horizontal_negative": [],
            "harmonic_vertical_positive": [],
            "harmonic_vertical_negative": [],
            "harmonic_diagonal_p1_p1": [],
            "harmonic_diagonal_p1_n1": [],
            "harmonic_diagonal_n1_p1": [],
            "harmonic_diagonal_n1_n1": []
        }
        # Categorize each file into its corresponding channel based on its stem
        for file in flat_files:
            stem = file.stem
            for key in channels:
                if key in stem:
                    channels[key].append(file)
        
        # Process each channel found in the dictionary
        for key, file_list in channels.items():
            if file_list:
                avg_image = average_images(file_list)
                ti.imwrite(export_dir / f"{key}.tif", avg_image, imagej=True)

    elif type_of_contrast == "phase":
        flat_files = list((path_to_flat / type_of_contrast).glob("*tif"))
        channels = {"horizontal": [], "vertical": []}
        for file in flat_files:
            stem = file.stem
            if "horizontal" in stem:
                channels["horizontal"].append(file)
            if "vertical" in stem:
                channels["vertical"].append(file)
        for key, file_list in channels.items():
            if file_list:
                avg_image = average_images(file_list)
                ti.imwrite(export_dir / f"{key}.tif", avg_image, imagej=True)

    elif type_of_contrast == "absorption":
        absorption_files = list((path_to_flat / type_of_contrast).glob("*.tif"))
        if absorption_files:
            avg_image = average_images(absorption_files)
            ti.imwrite(export_dir / "absorption.tif", avg_image, imagej=True)
        else:
            raise FileNotFoundError("No absorption flat field files found.")

    else:
        print("Unsupported type of contrast provided.")

    return export_dir

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path


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
            return imread(file_list[0]).astype(np.float32)
        else:
            images = [imread(f) for f in file_list]
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
                imwrite(export_dir / f"{key}.tif", avg_image, imagej=True)

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
                imwrite(export_dir / f"{key}.tif", avg_image, imagej=True)

    elif type_of_contrast == "absorption":
        absorption_files = list((path_to_flat / type_of_contrast).glob("*.tif"))
        if absorption_files:
            avg_image = average_images(absorption_files)
            imwrite(export_dir / "absorption.tif", avg_image, imagej=True)
        else:
            raise FileNotFoundError("No absorption flat field files found.")

    else:
        print("Unsupported type of contrast provided.")

    return export_dir

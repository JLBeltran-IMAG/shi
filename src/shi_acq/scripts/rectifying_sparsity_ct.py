import numpy as np

from pathlib import Path
import tifffile as ti
import sys


path_to = "/media/beltran/Beltransito/CT new data/25022025_chicken_parts/sample/ct_wing_tip/"


def check_sparse(img):
    """
    Check if an image is sparse based on its pixel density.
    This function calculates the density of non-zero pixels in an image and determines
    if the image is sparse (density <= 0.75).
    Parameters
    ----------
    img : numpy.ndarray
        Input image array.
    Returns
    -------
    bool
        True if image is sparse (density <= 0.75), False otherwise.
    Notes
    -----
    The density is calculated as the ratio of non-zero pixels to total pixels.
    An image is considered sparse if 75% or more of its pixels are zero.
    """
    pixel_number = img.size
    pixel_nonzero_number = np.count_nonzero(img)
    density = pixel_nonzero_number / pixel_number

    if density <= 0.75:
        return True
    else:
        return False


def alphanumeric_chars_in_filename(filename):
    """
    Extract numeric characters from a filename.

    This function filters out all non-digit characters from the input filename and
    returns a string containing only the numeric characters.

    Parameters
    ----------
    filename : str
        The input filename to process

    Returns
    -------
    str
        A string containing only the numeric characters found in the filename

    Examples
    --------
    >>> alphanumeric_chars_in_filename("test123.txt")
    '123'
    >>> alphanumeric_chars_in_filename("abc")
    ''
    """
    return "".join(list(filter(str.isdigit, filename)))


pos_paths = [pos for pos in Path(path_to).glob("*.txt")]
img_paths = [img for img in Path(path_to).glob("*.tif")]

angles = list()
for pos_path in pos_paths:
    with open(pos_path, "r") as file:
        for i in file.readlines():
            angles.append(int(i.strip()))

angles = sorted(angles)
img_paths = sorted(img_paths)
angles_to_delete = list()


export_pos_path = path_to + "positions_raw.txt"
with open(export_pos_path, "w") as pos_file:
    pos_file.write("\n".join(map(str, angles)) + "\n")


if len(angles) == len(img_paths):
    # assert len(angles) == len(img_paths)
    for i, img in enumerate(img_paths):
        im = ti.imread(img)
        if check_sparse(im):
            print("Deleting sparse image: ", img)
            img.unlink()
            angles_to_delete.append(i)
            # print("Deleting sparse position: ", angles[i])
            # angles.pop(i)

    for i in angles_to_delete:
        angles.pop(i)

    export_pos_path = path_to + "positions.txt"
    with open(export_pos_path, "w") as pos_file:
        pos_file.write("\n".join(map(str, angles)) + "\n")
else:
    print("The number of images and positions files does not match")
    sys.exit(1)




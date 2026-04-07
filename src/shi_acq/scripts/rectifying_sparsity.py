import numpy as np
from pathlib import Path
import tifffile as ti


path_to = "/media/beltran/Beltransito/CT new data/meltenct/bright"


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


img_paths = [img for img in Path(path_to).glob("*.tif")]
img_paths = sorted(img_paths)


for img in img_paths:
    im = ti.imread(img)
    if check_sparse(im):
        print("Deleting sparse image: ", img)
        img.unlink()


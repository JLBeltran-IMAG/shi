import numpy as np
import matplotlib.pyplot as plt
import tifffile as ti
import skimage.io as io

from scipy.ndimage import zoom


filepath = "/home/beltran/Documents/CXI/CXI-DATA-ANALYSIS/ct_wallnut/tomo_absorption_manual__astra32"

path_to_tomo = f"{filepath}.tif"
path_to_tomo_resized = f"{filepath}_resized.tif"
tomo = ti.imread(path_to_tomo)

scale_factor = 0.5
resized_tomo = zoom(tomo, (scale_factor, scale_factor, scale_factor), order=1)

io.imsave(path_to_tomo_resized, resized_tomo[10:-10])

# print(tomo.shape)


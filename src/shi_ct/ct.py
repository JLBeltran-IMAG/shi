import numpy as np
import tifffile as ti
import tomopy
from skimage import io
from pathlib import Path

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# Para automatizar
# ----------------
contrastes = ["absorption", "scattering", "phase", "phasemap"]


# Defining the paths
# ----------------------------------------
main_path = "/home/beltran/Documents/CXI/CXI-DATA-ANALYSIS/ct_hazelnut"
contrast = "absorption"
adjustment = "manual"
orientation = ""

path_to_ctscan = f"{main_path}/ct/{contrast}/{orientation}/{adjustment}_adjusted"
path_to_center = f"{main_path}/ct/{contrast}/{orientation}"
path_to_angles = f"{main_path}"


# Reading the images-proj
# ----------------------------------------
images_path = sorted([filepath for filepath in Path(path_to_ctscan).glob("*.tif")])
images = np.array([ti.imread(imgpath) for imgpath in images_path])


# Reading the centers
# -------------------
centers = np.load(f"{path_to_center}/centers.npz")
all_centers = np.unique(np.concatenate([centers["arr_0"], centers["arr_1"], centers["arr_2"]]))


# Reading the angles
# ------------------
theta_list = list()
ndeg = 1e-9


with open(f"{path_to_angles}/positions.txt", "r") as file_positions:
    for i in file_positions.readlines():
        angle = ndeg * int(i.strip())
        theta_list.append(angle)

theta = np.deg2rad(np.array(theta_list, dtype=np.float32))


# Preprocessing
# -------------




import time


t0 = time.time()
# Recontructing tomography
# ------------------------
recon = tomopy.recon(
    images,
    theta,
    center=278.5,
    algorithm='sirt',
    num_iter=1,
    accelerated=True,
    device='gpu',
    ncore=4,
    pool_size=16
    )
t1 = time.time()
print(f"Time taken for reconstruction: {t1 - t0} seconds")

# ti.imwrite(f"{main_path}/tomo_{contrast}_{adjustment}_{orientation}_sirt.tif", recon, imagej=True)



# recon = tomopy.remove_ring(recon)
# recon = tomopy.circ_mask(recon, axis=0, ratio=0.9)
# recon = tomopy.remove_outlier3d(recon, dif=0.2)
# recon = tomopy.remove_neg(recon)



# tomopy.write_center(
#     tomo=images,
#     theta=theta,
#     cen_range=[330, 340, 0.2],
#     mask=True,
#     ratio=0.9,
#     algorithm = "sirt",
#     filter_name="none"
#     )



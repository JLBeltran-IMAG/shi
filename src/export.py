import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import tifffile as ti
from skimage import exposure

from pathlib import Path
import shutil
import itertools



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
        print(f"Copied: {file_path} -> {destination_directory}")


def graphing_absorption(path, colormap = "gray"):
    path_to_absorption = path.joinpath("absorption_avg.tif")

    img = exposure.rescale_intensity(ti.imread(path_to_absorption), out_range = (0, 255))
    fig = plt.figure(figsize = (10, 10))

    ax = fig.add_subplot(1, 1, 1)
    c = ax.imshow(img, cmap = colormap)
    ax.set_title("Absorption")
    ax.set_xticks([])
    ax.set_yticks([])

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(c, cax = cax, orientation = "vertical")

    fig.tight_layout()
    fig.savefig("{}.png".format("Absorption"))


def graphing_scatt_phase(path_to_imgs, type_of_contrast, colormap = "gray"):
    images = list(path_to_imgs.glob("*.tif"))

    if type_of_contrast == "phase":
        img1, label_img1 = ti.imread([img for img in images if "vertical" in img.stem]), "vertical phase"
        img2, label_img2 = ti.imread([img for img in images if "horizontal" in img.stem]), "horizontal phase"
        img3, label_img3 = ti.imread([img for img in images if "bidirectional" in img.stem]), "bidirectional phase"

    elif type_of_contrast == "scattering":
        img1, label_img1 = ti.imread([img for img in images if "vertical_scattering" in img.stem]), "vertical scattering"
        img2, label_img2 = ti.imread([img for img in images if "horizontal_scattering" in img.stem]), "horizontal scattering"
        img3, label_img3 = ti.imread([img for img in images if "bidirectional_scattering" in img.stem]),"bidirectional scattering"
    
    else:
        print("error")

    fig = plt.figure(figsize = (10, 10))

    ax1 = fig.add_subplot(1, 3, 1)
    c1 = ax1.imshow(img1, cmap = colormap)
    ax1.set_title("{} contrast - Y".format(type_of_contrast))
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1_divider = make_axes_locatable(ax1)
    cax1 = ax1_divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(c1, cax = cax1, orientation = "vertical")

    ax2 = fig.add_subplot(1, 3, 2)
    c2 = ax2.imshow(img2, cmap = colormap)
    ax2.set_title("{} contrast - X".format(type_of_contrast))
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2_divider = make_axes_locatable(ax2)
    cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(c2, cax = cax2, orientation = "vertical")

    ax3 = fig.add_subplot(1, 3, 3)
    c3 = ax3.imshow(img3, cmap = colormap)
    ax3.set_title("Bi-directional {} contrast".format(type_of_contrast))
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax3_divider = make_axes_locatable(ax3)
    cax3 = ax3_divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(c3, cax = cax3, orientation = "vertical")

    fig.tight_layout()

    fig_to_export = plt.figure(figsize=(10, 10))
    ax = fig_to_export.add_subplot(1, 1, 1)


    for img_to_export, label_to_export in zip([img1, img2, img3], [label_img1, label_img2, label_img3]):
        c = ax.imshow(img_to_export, cmap = colormap)
        ax.set_title("{} - contrast".format(label_to_export))
        ax.set_xticks([])
        ax.set_yticks([])

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="3%", pad="2%")
        fig.colorbar(c, cax = cax, orientation = "vertical")

        fig_to_export.savefig("average/{}.png".format(label_to_export))

    return 0

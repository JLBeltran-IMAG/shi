import numpy as np
# import corrections
import tifffile as ti
from PySide6.QtWidgets import QApplication, QFileDialog
from pathlib import Path


def delete_detector_stripes(image, stripes0, stripes1):
    image_without_stripes = np.delete(image, stripes0, axis = 0)
    image_without_stripes = np.delete(image_without_stripes, stripes1, axis = 1)
    return image_without_stripes


def correcting_stripes():
    app = QApplication([])
    default_folder = Path().home().joinpath("Documents/CXI/CXI-DATA-ACQUISITION").as_posix()
    folder = QFileDialog.getExistingDirectory(None, "Choose folder", dir = default_folder)
    app.quit()

    path_to_images = set([path.parent for path in Path(folder).rglob("*tif") if "no_stripe" not in path.parent.name])

    for dirs in path_to_images:
        path_to_nostripes = dirs.joinpath("no_stripe")
        path_to_nostripes.mkdir(parents = True, exist_ok = True)

        for file in dirs.glob("*.tif"):
            export_file = path_to_nostripes.joinpath(file.name)
            img = ti.imread(file)
            img_without_stripes = delete_detector_stripes(img, [2944, 2945], [295, 722, 1167, 1388, 1541, 2062, 2302, 2303])

            ti.imwrite(export_file, img_without_stripes, imagej = True)


if __name__ == "__main__":
    correcting_stripes()



import numpy as np
import tifffile as ti
import argparse
from pathlib import Path


def averaging_same_name(path_to_files, type_of_contrast):
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

        avg_phasemap_positive = (avg_horizontal2 + avg_vertical2) / 2
        avg_phasemap_negative = (avg_horizontal1 + avg_vertical1) / 2

        ti.imwrite("{}/horizontal_positive_phasemap.tif".format(path_to_export), avg_horizontal2)
        ti.imwrite("{}/vertical_positive_phasemap.tif".format(path_to_export), avg_vertical2)
        ti.imwrite("{}/bidirectional_positive_phasemap.tif".format(path_to_export), avg_phasemap_positive)
        ti.imwrite("{}/horizontal_negative_phasemap.tif".format(path_to_export), avg_horizontal1)
        ti.imwrite("{}/vertical_negative_phasemap.tif".format(path_to_export), avg_vertical1)
        ti.imwrite("{}/bidirectional_negative_phasemap.tif".format(path_to_export), avg_phasemap_negative)

    else:
        print("Write the right contrast")
    
    return path_to_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Averange - python script", description = "%(prog)s implements the average of different files if they have the same name")
    parser.add_argument("-t", "--type", type = str, required = True, help = "type of contrast")
    parser.add_argument("--path", type = str, required = True, help = "path containing type of contrast")
    # parser.add_argument("-i", "--input_path", type = str, required = False, help = "path to files which should be averaged")
    args = parser.parse_args()


    path_to = averaging_same_name(Path(args.path), args.type)




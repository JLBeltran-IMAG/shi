from pathlib import Path
import shutil
import argparse


def organize_dir(path_to_images, type_of_contrast):
    path_to_images = Path(path_to_images)
    path_to_files = path_to_images.joinpath(type_of_contrast, "flat_corrections")

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
            outdir = path_to_files.joinpath(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)

            list_of_orientation = [path_to for path_to in path_to_files.glob("*.tif") if output_dir in path_to.stem]

            for orientation in list_of_orientation:
                print("Moving", orientation, "to", outdir)
                shutil.move(orientation, outdir)

    elif type_of_contrast == "phase":
        output_dir_list = ("horizontal", "vertical")

        for output_dir in output_dir_list:
            outdir = path_to_files.joinpath(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)

            list_of_orientation = [path_to for path_to in path_to_files.glob("*.tif") if output_dir in path_to.stem]

            for orientation in list_of_orientation:
                print("Moving", orientation, "to", outdir)
                shutil.move(orientation, outdir)
        pass

    else:
        raise ValueError(f"Unknown type_of_contrast: {type_of_contrast}")


if __name__ == "__main__":
    type_of_contrast = ("absorption", "scattering", "phase", "phasemap")

    parser = argparse.ArgumentParser(description="Organize the directory for CT images")
    parser.add_argument("path", help="Path to the directory where the images are stored")
    args = parser.parse_args()
    path_to_move = args.path

    for contrast in type_of_contrast:
        organize_dir(path_to_move, contrast)



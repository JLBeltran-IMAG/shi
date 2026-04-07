from pathlib import Path
import argparse

# import grabImage


def my_parser_cli():
    """Creates and returns an ArgumentParser for the ACQ command line interface.

    This function sets up the command line interface for the ACQ software, which is used for
    automated image acquisition. It creates a main parser with subparsers for different
    functionalities.

    Returns:
        argparse.ArgumentParser: A configured argument parser with the following structure:
            - Main parser with program name "ACQ"
            - Subcommands:
                - "create": For creating project directories
                    Options:
                    - -n/--name: Required, name of directory to create
                    - -d/--delete: Optional, name of directory to delete
            - Mutually exclusive group for image types:
                - --dark
                - --flat  
                - --bright
                - --sample (takes 2 arguments)

    Example:
        parser = my_parser_cli()
        args = parser.parse_args()
    """
    main_parser = argparse.ArgumentParser(
        prog="ACQ",
        description="%(prog)s: This software is an automated implementation for taking images with order",
    )

    subparsers = main_parser.add_subparsers(dest="comando")

    # Defining subparsers for various functionalities
    parser_create = subparsers.add_parser("create", help="This subcommand create the directry for project.")
    # parser_snapsn = subparsers.add_parser("snapsn", help="This subcommand take n snaps for flat, dark, bright or sample.")
    group_snapsn = main_parser.add_mutually_exclusive_group()

    # ---------------------------------------- Subparser calculate ----------------------------------------------------
    # Option 1: ...
    parser_create.add_argument("-n", "--name", required=True, type=str, help="Name of the directory.")
    parser_create.add_argument("-d", "--delete", type=str, help="Delete the directory with the name specify by -d or --delete")

    # Option 2: ...
    group_snapsn.add_argument("--dark", help="...", )
    group_snapsn.add_argument("--flat", help="...", )
    group_snapsn.add_argument("--bright", help="...", )
    group_snapsn.add_argument("--sample", nargs=2, help="...", )
    # -----------------------------------------------------------------------------------------------------------------

    return main_parser


def create(dirname):
    """Create directory structure for an acquisition project.

    This function creates a project directory with subdirectories for different types of
    acquisition data: dark, flat, bright and sample images.

    Args:
        dirname (str): Name of the project directory to be created

    Returns:
        tuple: Contains Path objects for the following directories:
            - project_dir: Main project directory
            - dark_dir: Directory for dark field images
            - flat_dir: Directory for flat field images
            - bright_dir: Directory for bright field images
            - sample_dir: Directory for sample images

    The directories are created with parents=True and exist_ok=True, meaning:
    - All necessary parent directories will be created
    - No error is raised if directories already exist
    """
    current_dir = Path().cwd()

    project_dir = current_dir.joinpath("{}".format(dirname))
    dark_dir = project_dir.joinpath("dark")
    flat_dir = project_dir.joinpath("flat")
    bright_dir = project_dir.joinpath("bright")
    sample_dir = project_dir.joinpath("sample")

    # Creating the folder and subfolders
    project_dir.mkdir(parents = True, exist_ok = True)

    dark_dir.mkdir(parents = True, exist_ok = True)
    flat_dir.mkdir(parents = True, exist_ok = True)
    bright_dir.mkdir(parents = True, exist_ok = True)
    sample_dir.mkdir(parents = True, exist_ok = True)

    return project_dir, dark_dir, flat_dir, bright_dir, sample_dir




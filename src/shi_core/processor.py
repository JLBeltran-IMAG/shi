"""Main processor class for SHI operations."""
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import skimage.io as io

from shi_core.config import config
from shi_core.exceptions import ImageNotFoundError, ProcessingError
from shi_core.logging import logger

import src.spatial_harmonics as spatial_harmonics
import src.directories as directories
import src.corrections as corrections
import src.crop_tk as crop_tk
import src.angles_correction as angles_correction


class SHIProcessor:
    """Main class for handling SHI processing operations."""
    
    def __init__(self, mask_period: int, unwrap_method: Optional[str] = None):
        """
        Initialize SHI processor.
        
        Args:
            mask_period: Number of projected pixels in the mask
            unwrap_method: Phase unwrapping method to use
        """
        self.mask_period = mask_period
        if unwrap_method and not config.validate_unwrap_method(unwrap_method):
            raise ValueError(f"Invalid unwrap method: {unwrap_method}")
        self.unwrap_method = unwrap_method

    def process_directory(
        self,
        images_path: Union[str, Path],
        dark_path: Optional[Union[str, Path]] = None,
        flat_path: Optional[Union[str, Path]] = None,
        bright_path: Optional[Union[str, Path]] = None,
        mode: str = "2d",
        angle_after: bool = False,
        average: bool = False,
        export: bool = False
    ) -> None:
        """
        Process all .tif files in a directory.

        Args:
            images_path: Path to directory containing sample images
            dark_path: Optional path to dark images
            flat_path: Optional path to flat images
            bright_path: Optional path to bright images
            mode: Processing mode ('2d' or '3d')
            angle_after: Whether to apply angle correction after measurements
            average: Whether to apply averaging
            export: Whether to export results
        """
        images_path = Path(images_path)
        
        # Convert other paths to Path objects if they exist
        dark_path = Path(dark_path) if dark_path else None
        flat_path = Path(flat_path) if flat_path else None
        bright_path = Path(bright_path) if bright_path else None

        # Find all .tif files in the directory
        image_files = list(images_path.glob("*.tif"))
        if not image_files:
            raise ImageNotFoundError(f"No .tif files found in {images_path}")

        logger.info(f"Processing {len(image_files)} images in {images_path}")

        # Get angle correction if needed
        deg = self._get_angle_correction(
            image_files[0],
            flat_path
        ) if angle_after else np.float32(0)

        # Important: Only do crop ONCE per directory, not per image
        # Get crop from first image in the directory
        # and make sure to use an actual image file
        first_image = image_files[0]  # Use the first .tif file found
        logger.info(f"Using image {first_image} for crop reference")
        crop_from_tmptxt = crop_tk.cropImage(first_image)

        # Apply corrections based on the crop settings
        if dark_path:
            # Ensure paths are not None before passing to correction functions
            if flat_path is None:
                raise ValueError("Flat path is required when dark path is provided")

            self._apply_dark_bright_corrections(
                dark_path,
                flat_path,
                images_path,
                bright_path,
                crop_from_tmptxt,
                deg
            )
            foldername_to = "corrected_images"
        else:
            # Ensure flat path is provided for crop only
            if flat_path is None:
                raise ValueError("Flat path is required for crop only operation")

            self._apply_crop_only(
                images_path,
                flat_path,
                crop_from_tmptxt,
                deg
            )
            foldername_to = "crop_without_correction"

        # Create result directories and process
        corrected_dir = images_path / foldername_to
        if not corrected_dir.exists():
            corrected_dir.mkdir(parents=True, exist_ok=True)

        path_to_images, path_to_result = directories.create_result_subfolders(
            file_dir=str(corrected_dir),
            result_folder=images_path.name,
            sample_folder=""
        )

        # Execute SHI processing
        if not flat_path:
            spatial_harmonics.execute_SHI(
                path_to_images,
                path_to_result,
                self.mask_period,
                self.unwrap_method,
                False
            )
        else:
            # Process with reference field
            path_to_corrected_flat = flat_path / foldername_to

            if not path_to_corrected_flat.exists():
                path_to_corrected_flat.mkdir(parents=True, exist_ok=True)

            path_to_flat, path_to_flat_result = directories.create_result_subfolders(
                file_dir=str(path_to_corrected_flat),
                result_folder=images_path.name,
                sample_folder="flat",
            )

            # Execute SHI for flat and images
            spatial_harmonics.execute_SHI(
                path_to_flat,
                path_to_flat_result,
                self.mask_period,
                self.unwrap_method,
                True
            )
            spatial_harmonics.execute_SHI(
                str(corrected_dir),
                path_to_result,
                self.mask_period,
                self.unwrap_method,
                False
            )

            # Create corrections folder and apply flat corrections
            result_path = directories.create_corrections_folder(path_to_result)
            for contrast in config.CONTRAST_TYPES:
                path_to_average_flat = directories.average_flat_harmonics(
                    path_to_flat_result,
                    contrast
                )
                corrections.correct_flatmask(
                    result_path,
                    path_to_average_flat,
                    contrast
                )

            # Handle 2D/3D specific operations
            if mode == "2d" and average:
                self._handle_2d_averaging(result_path)
            elif mode == "3d":
                self._handle_3d_organization(result_path)
        
        # Export if requested
        if export:
            logger.info(f"Exporting results to {path_to_result}")
            directories.export_results(path_to_result)


    def process_single_image(
        self,
        image_path: Path,
        dark_path: Optional[Path],
        flat_path: Optional[Path],
        bright_path: Optional[Path],
        mode: str,
        angle_after: bool,
        average: bool,
        export: bool
    ) -> None:
        """
        Process a single image file (public method).
        """
        return self._process_single_image(
            image_path,
            dark_path,
            flat_path,
            bright_path,
            mode,
            angle_after,
            average,
            export
        )


    def _process_single_image(
        self,
        image_path: Path,
        dark_path: Optional[Path],
        flat_path: Optional[Path],
        bright_path: Optional[Path],
        mode: str,
        angle_after: bool,
        average: bool,
        export: bool
    ) -> None:
        """Process a single image file."""
        logger.info(f"Processing measurement: {image_path}")

        # Get angle correction if needed
        deg = self._get_angle_correction(
            image_path,
            flat_path
        ) if angle_after else np.float32(0)

        # Crop image
        crop_from_tmptxt = crop_tk.cropImage(image_path)

        # Apply corrections
        if dark_path:
            # Ensure paths are not None before passing to correction functions
            if flat_path is None:
                raise ValueError("Flat path is required when dark path is provided")

            self._apply_dark_bright_corrections(
                dark_path,
                flat_path,
                image_path,
                bright_path,
                crop_from_tmptxt,
                deg
            )
            foldername_to = "corrected_images"
        else:
            # Ensure flat path is provided for crop only
            if flat_path is None:
                raise ValueError("Flat path is required for crop only operation")

            self._apply_crop_only(
                image_path,
                flat_path,
                crop_from_tmptxt,
                deg
            )
            foldername_to = "crop_without_correction"

        # Create result directories and process. Instead of creating corrected_images
        # as a child of the .tif file (which is wrong), create it as a sibling
        # directory to the parent folder containing the .tif
        parent_dir = image_path.parent
        corrected_dir = parent_dir / foldername_to

        # Ensure the directory exists
        if not corrected_dir.exists():
            corrected_dir.mkdir(parents=True, exist_ok=True)

        # Use the image stem as a subfolder name within corrected_images
        path_to_images, path_to_result = directories.create_result_subfolders(
            file_dir=str(corrected_dir),
            result_folder=image_path.stem,
            sample_folder=""
        )

        # Execute SHI processing
        if not flat_path:
            spatial_harmonics.execute_SHI(
                path_to_images,
                path_to_result,
                self.mask_period,
                self.unwrap_method,
                False
            )
        else:
            self._process_with_flat(
                corrected_dir,
                flat_path,
                path_to_result,
                mode,
                average
            )

        # Export if requested
        if export:
            logger.info(f"Exporting results to {path_to_result}")
            directories.export_results(path_to_result)


    def _get_angle_correction(
        self,
        image_path: Path,
        flat_path: Optional[Path]
    ) -> np.float32:
        """Calculate angle correction."""
        path_to_ang = flat_path if flat_path else image_path
        tif_files = list(path_to_ang.glob("*.tif"))

        if not tif_files:
            logger.warning(f"No .tif files found for angle correction in {path_to_ang}")
            return np.float32(0)

        path_to_angle_correction = tif_files[0]
        image_angle = io.imread(str(path_to_angle_correction))
        cords = angles_correction.extracting_coordinates_of_peaks(image_angle)

        return angles_correction.calculating_angles_of_peaks_average(cords)


    def _apply_dark_bright_corrections(
        self,
        dark_path: Path,
        flat_path: Path,
        images_path: Path,
        bright_path: Optional[Path],
        crop: tuple,
        angle: np.float32
    ) -> None:
        """Apply dark and bright field corrections to all images in a directory."""
        # Apply corrections to the flat path
        corrections.correct_darkfield(
            path_to_dark=str(dark_path),
            path_to_images=str(flat_path),
            crop=crop,
            allow_crop=True,
            angle=angle
        )
        
        # Apply corrections to the sample images directory
        corrections.correct_darkfield(
            path_to_dark=str(dark_path),
            path_to_images=str(images_path),
            crop=crop,
            allow_crop=True,
            angle=angle
        )
        
        if bright_path:
            corrections.correct_darkfield(
                path_to_dark=str(dark_path),
                path_to_images=str(bright_path),
                crop=crop,
                allow_crop=True,
                angle=angle
            )
            corrections.correct_brightfield(
                path_to_bright=str(bright_path),
                path_to_images=str(flat_path)
            )
            corrections.correct_brightfield(
                path_to_bright=str(bright_path),
                path_to_images=str(images_path)
            )


    def _apply_crop_only(
        self,
        images_path: Path,
        flat_path: Path,
        crop: tuple,
        angle: np.float32
    ) -> None:
        """Apply cropping without corrections to all images in a directory."""
        corrections.crop_without_corrections(
            path_to_images=str(images_path),
            crop=crop,
            allow_crop=True,
            angle=angle
        )
        corrections.crop_without_corrections(
            path_to_images=str(flat_path),
            crop=crop,
            allow_crop=True,
            angle=angle
        )


    def _process_with_flat(
        self,
        corrected_path: Path,
        flat_path: Path,
        result_path: Path,
        mode: str,
        average: bool
    ) -> None:
        """Process with flat field correction."""
        # Make sure corrected_path exists and is a directory
        if not corrected_path.exists() or not corrected_path.is_dir():
            logger.error(f"Invalid corrected path: {corrected_path}")
            return
            
        # Create corrected_images directory in flat_path if it doesn't exist
        path_to_corrected_flat = flat_path / corrected_path.name
        if not path_to_corrected_flat.exists():
            path_to_corrected_flat.mkdir(parents=True, exist_ok=True)
        
        path_to_flat, path_to_flat_result = directories.create_result_subfolders(
            file_dir=str(path_to_corrected_flat),
            result_folder=result_path.name,
            sample_folder="flat",
        )
        
        # Execute SHI for flat and images
        spatial_harmonics.execute_SHI(
            path_to_flat,
            path_to_flat_result,
            self.mask_period,
            self.unwrap_method,
            True
        )
        spatial_harmonics.execute_SHI(
            str(corrected_path),
            result_path,
            self.mask_period,
            self.unwrap_method,
            False
        )

        # Create corrections folder and apply flat corrections
        result_path = directories.create_corrections_folder(result_path)
        for contrast in config.CONTRAST_TYPES:
            path_to_average_flat = directories.average_flat_harmonics(
                path_to_flat_result,
                contrast
            )
            corrections.correct_flatmask(
                result_path,
                path_to_average_flat,
                contrast
            )

        # Handle 2D/3D specific operations
        if mode == "2d" and average:
            self._handle_2d_averaging(result_path)
        elif mode == "3d":
            self._handle_3d_organization(result_path)


    def _handle_2d_averaging(self, result_path: Path) -> None:
        """Handle 2D averaging operations."""
        for contrast in config.CONTRAST_TYPES:
            path_avg = result_path / contrast / "flat_corrections"
            logger.info(f"Averaging contrast: {contrast}")
            directories.averaging(path_avg, contrast)


    def _handle_3d_organization(self, result_path: Path) -> None:
        """Handle 3D organization operations."""
        for contrast in config.CONTRAST_TYPES:
            ct_dir = result_path / contrast / "flat_corrections"
            logger.info(f"Organizing contrast: {contrast}")
            directories.organize_dir(ct_dir, contrast)



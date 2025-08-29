"""
Main orchestrator for SHI processing pipeline.

This module contains the SHIProcessingOrchestrator class which handles
the complete processing pipeline for spatial harmonic imaging without
any UI dependencies.
"""

import time
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from skimage.transform import rotate
from tifffile import imread

from processing_context import (
    SHIMeasurementContext, 
    SHIBatchContext,
    MeasurementResult,
    SHIProcessingResults
)
from processing_interfaces import ProcessingLogger, ConsoleLogger
from processing_exceptions import (
    SHIProcessingError,
    ImageNotFoundError,
    CorrectionError,
    ContrastRetrievalError
)

# Import existing modules
import spatial_harmonics
import directories
import corrections
import utils


class SHIProcessingOrchestrator:
    """Orchestrates the complete SHI processing pipeline.
    
    This class handles all processing steps without any UI dependencies,
    making it suitable for use in different application types (CLI, web, API).
    """
    
    def __init__(self, logger: Optional[ProcessingLogger] = None):
        """Initialize the orchestrator.
        
        Args:
            logger: Logger implementation to use (defaults to ConsoleLogger)
        """
        self.logger = logger or ConsoleLogger()
    
    def process_batch(self, batch_context: SHIBatchContext) -> SHIProcessingResults:
        """Process a batch of measurements.
        
        Args:
            batch_context: Context containing all measurements and configuration
            
        Returns:
            SHIProcessingResults with details of all processed measurements
        """
        start_time = time.time()
        results = []
        
        self.logger.info("Starting batch processing of %d measurements", len(batch_context.measurements))
        
        for measurement in batch_context.measurements:
            self.logger.info("Processing measurement: %s", measurement.measurement_name)
            
            try:
                result = self._process_single_measurement(
                    measurement, 
                    batch_context
                )
                results.append(result)
                
                if result.success:
                    self.logger.info("Successfully processed: %s", measurement.measurement_name)
                else:
                    self.logger.error("Failed to process %s: %s", 
                                    measurement.measurement_name, result.error)
                    
            except Exception as e:
                self.logger.error("Unexpected error processing %s: %s", 
                                measurement.measurement_name, str(e))
                result = MeasurementResult(
                    measurement_name=measurement.measurement_name,
                    success=False,
                    output_paths={},
                    processing_time=0,
                    error=str(e)
                )
                results.append(result)
        
        total_time = time.time() - start_time
        processing_results = SHIProcessingResults(
            measurements=results,
            total_processing_time=total_time
        )
        
        self.logger.info("Batch processing completed: %s", processing_results.summary())
        
        return processing_results
    
    def _process_single_measurement(
        self, 
        context: SHIMeasurementContext,
        batch_context: SHIBatchContext
    ) -> MeasurementResult:
        """Process a single measurement.
        
        Args:
            context: Context for the specific measurement
            batch_context: Global batch configuration
            
        Returns:
            MeasurementResult with processing details
        """
        start_time = time.time()
        
        try:
            # Validate input images exist
            tif_files = list(context.images_path.glob("*.tif"))
            if not tif_files:
                raise ImageNotFoundError(
                    context.images_path, 
                    "sample", 
                    context.measurement_name
                )
            
            # Apply corrections based on available calibration data
            if context.dark_path is not None:
                foldername_to = self._apply_corrections(context, batch_context)
            else:
                foldername_to = self._apply_crop_only(context, batch_context)
            
            # Setup result directories
            path_to_corrected_images = context.images_path / foldername_to
            path_to_images, path_to_result = self._create_result_directories(
                path_to_corrected_images,
                context.measurement_name,
                batch_context.output_base_path
            )
            
            # Apply flat field corrections if available
            # This must happen BEFORE sample processing to create harmonics.pkl
            if context.flat_path is not None:
                self._apply_flat_corrections(
                    context,
                    path_to_result,
                    foldername_to,
                    batch_context
                )
            else:
                # Execute spatial harmonics analysis without flat correction
                self._execute_spatial_harmonics(
                    context,
                    path_to_images,
                    path_to_result,
                    batch_context.temp_directory,
                    is_flat=False
                )
            
            # Post-processing based on mode
            self._apply_post_processing(
                path_to_result,
                batch_context
            )
            
            # Export results if requested
            if batch_context.export_results:
                self._export_results(path_to_result)
            
            processing_time = time.time() - start_time
            
            return MeasurementResult(
                measurement_name=context.measurement_name,
                success=True,
                output_paths={
                    'main': path_to_result,
                    'absorption': path_to_result / 'absorption',
                    'scattering': path_to_result / 'scattering',
                    'phase': path_to_result / 'phase',
                    'phasemap': path_to_result / 'phasemap'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error("Error processing measurement %s: %s", 
                            context.measurement_name, str(e))
            
            return MeasurementResult(
                measurement_name=context.measurement_name,
                success=False,
                output_paths={},
                processing_time=processing_time,
                error=str(e)
            )
    
    def _apply_corrections(
        self, 
        context: SHIMeasurementContext,
        batch_context: SHIBatchContext
    ) -> str:
        """Apply dark and bright field corrections.
        
        Args:
            context: Measurement context
            batch_context: Batch configuration
            
        Returns:
            Name of the folder containing corrected images
        """
        self.logger.info("Applying dark field corrections")
        
        # Apply dark field correction to flat
        if context.flat_path:
            corrections.correct_darkfield(
                path_to_dark=context.dark_path,
                path_to_images=context.flat_path,
                crop=context.crop_region,
                allow_crop=True,
                angle=context.rotation_angle
            )
        
        # Apply dark field correction to sample
        corrections.correct_darkfield(
            path_to_dark=context.dark_path,
            path_to_images=context.images_path,
            crop=context.crop_region,
            allow_crop=True,
            angle=context.rotation_angle
        )
        
        # Apply bright field correction if available
        if context.bright_path is not None:
            self.logger.info("Applying bright field corrections")
            
            corrections.correct_darkfield(
                path_to_dark=context.dark_path,
                path_to_images=context.bright_path,
                crop=context.crop_region,
                allow_crop=True,
                angle=context.rotation_angle
            )
            
            if context.flat_path:
                corrections.correct_brightfield(
                    path_to_bright=context.bright_path,
                    path_to_images=context.flat_path
                )
            
            corrections.correct_brightfield(
                path_to_bright=context.bright_path,
                path_to_images=context.images_path
            )
        
        return "corrected_images"
    
    def _apply_crop_only(
        self,
        context: SHIMeasurementContext,
        batch_context: SHIBatchContext
    ) -> str:
        """Apply cropping without corrections.
        
        Args:
            context: Measurement context
            batch_context: Batch configuration
            
        Returns:
            Name of the folder containing cropped images
        """
        self.logger.info("Applying crop without corrections")
        
        corrections.crop_without_corrections(
            path_to_images=context.images_path,
            crop=context.crop_region,
            allow_crop=True,
            angle=context.rotation_angle
        )
        
        if context.flat_path:
            corrections.crop_without_corrections(
                path_to_images=context.flat_path,
                crop=context.crop_region,
                allow_crop=True,
                angle=context.rotation_angle
            )
        
        return "crop_without_correction"
    
    def _create_result_directories(
        self,
        corrected_path: Path,
        measurement_name: str,
        output_base_path: Path
    ) -> Tuple[List[Path], Path]:
        """Create result directories with configurable base path.
        
        Args:
            corrected_path: Path to corrected images
            measurement_name: Name of the measurement
            output_base_path: Base path for output
            
        Returns:
            Tuple of (image_paths, result_path)
        """
        # Get list of corrected images
        path_to_files = list(corrected_path.glob("*.tif"))
        
        # Create result directory structure
        result_path = output_base_path / measurement_name
        result_path.mkdir(parents=True, exist_ok=True)
        
        for subdir in ["absorption", "scattering", "phase", "phasemap"]:
            (result_path / subdir).mkdir(parents=True, exist_ok=True)
        
        return path_to_files, result_path
    
    def _execute_spatial_harmonics(
        self,
        context: SHIMeasurementContext,
        path_to_images: List[Path],
        path_to_result: Path,
        temp_directory: Path,
        is_flat: bool = False
    ) -> None:
        """Execute spatial harmonics analysis.
        
        Args:
            context: Measurement context
            path_to_images: List of image paths
            path_to_result: Output directory
            temp_directory: Temporary directory for processing
            is_flat: Whether processing flat field images
        """
        self.logger.info("Executing spatial harmonics analysis (flat=%s)", is_flat)
        
        # Pass temp_directory to spatial harmonics
        spatial_harmonics.execute_SHI(
            path_to_images,
            path_to_result,
            context.mask_period,
            context.unwrap_phase,
            is_flat,
            temp_dir=temp_directory
        )
    
    def _apply_flat_corrections(
        self,
        context: SHIMeasurementContext,
        path_to_result: Path,
        foldername_to: str,
        batch_context: SHIBatchContext
    ) -> None:
        """Apply flat field corrections.
        
        Args:
            context: Measurement context
            path_to_result: Result directory
            foldername_to: Name of corrected folder
            batch_context: Batch configuration
        """
        self.logger.info("Applying flat field corrections")
        
        path_to_corrected_flat = context.flat_path / foldername_to
        
        # Create flat result directories
        path_to_flat = list(path_to_corrected_flat.glob("*.tif"))
        path_to_flat_result = batch_context.output_base_path / path_to_result.name / "flat"
        path_to_flat_result.mkdir(parents=True, exist_ok=True)
        
        for subdir in ["absorption", "scattering", "phase", "phasemap"]:
            (path_to_flat_result / subdir).mkdir(parents=True, exist_ok=True)
        
        # Process flat field FIRST (creates harmonics.pkl)
        spatial_harmonics.execute_SHI(
            path_to_flat,
            path_to_flat_result,
            context.mask_period,
            context.unwrap_phase,
            True,  # is_flat parameter - creates harmonics.pkl
            temp_dir=batch_context.temp_directory
        )
        
        # Now process sample images (reads harmonics.pkl)
        path_to_images = list((context.images_path / foldername_to).glob("*.tif"))
        spatial_harmonics.execute_SHI(
            path_to_images,
            path_to_result,
            context.mask_period,
            context.unwrap_phase,
            False,  # is_flat parameter - reads harmonics.pkl
            temp_dir=batch_context.temp_directory
        )
        
        # Create corrections folder
        path_to_result = directories.create_corrections_folder(path_to_result)
        
        # Apply flat mask corrections for each contrast type
        type_of_contrast = ("absorption", "scattering", "phase", "phasemap")
        for contrast in type_of_contrast:
            path_to_average_flat = directories.average_flat_harmonics(
                path_to_flat_result,
                type_of_contrast=contrast
            )
            corrections.correct_flatmask(
                path_to_result,
                path_to_average_flat,
                type_of_contrast=contrast
            )
    
    def _apply_post_processing(
        self,
        path_to_result: Path,
        batch_context: SHIBatchContext
    ) -> None:
        """Apply post-processing based on processing mode.
        
        Args:
            path_to_result: Result directory
            batch_context: Batch configuration
        """
        type_of_contrast = ("absorption", "scattering", "phase", "phasemap")
        
        if batch_context.processing_mode == "2d":
            if batch_context.apply_averaging:
                self.logger.info("Applying averaging for 2D mode")
                for contrast in type_of_contrast:
                    path_avg = path_to_result / contrast / "flat_corrections"
                    if path_avg.exists():
                        directories.averaging(path_avg, contrast)
        
        elif batch_context.processing_mode == "3d":
            self.logger.info("Organizing directories for 3D/CT mode")
            for contrast in type_of_contrast:
                ct_dir = path_to_result / contrast / "flat_corrections"
                if ct_dir.exists():
                    directories.organize_dir(ct_dir, contrast)
    
    def _export_results(self, path_to_result: Path) -> None:
        """Export results to consolidated directory.
        
        Args:
            path_to_result: Result directory
        """
        self.logger.info("Exporting results to: %s", path_to_result)
        directories.export_results(path_to_result)
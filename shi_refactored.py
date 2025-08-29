#!/usr/bin/env python3
"""
Refactored SHI CLI using the new orchestrator architecture.

This script handles CLI argument parsing and interactive operations,
then delegates processing to the SHIProcessingOrchestrator.
"""

import numpy as np
import skimage.io as io
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import time

# Path to source
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.joinpath("src")

# Adding source directory to sys.path
sys.path.append(str(src_dir))

# Import processing modules
from processing_context import (
    SHIMeasurementContext,
    SHIBatchContext,
    SHIProcessingResults
)
from processing_orchestrator import SHIProcessingOrchestrator
from processing_interfaces import ConsoleLogger

# Import existing modules for interactive operations
import crop_tk
import angles_correction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_parser():
    """Create and configure the argument parser."""
    main_parser = argparse.ArgumentParser(
        prog="SHI",
        description="%(prog)s: Automated implementation of Spatial Harmonic Imaging",
    )

    subparsers = main_parser.add_subparsers(dest="command", required=True)

    # Calculate subcommand
    parser_shi = subparsers.add_parser("calculate", help="Execute the SHI method.")
    parser_shi.add_argument("-m", "--mask_period", required=True, type=str, 
                           help="Number of projected pixels in the mask.")
    
    # Input images options
    parser_shi.add_argument("-i", "--images", type=str, help="Path to sample image(s)")
    parser_shi.add_argument("-f", "--flat", type=str, help="Path to flat image(s)")
    parser_shi.add_argument("-d", "--dark", type=str, help="Path to dark image(s)")
    parser_shi.add_argument("-b", "--bright", type=str, help="Path to bright image(s)")
    
    # Automatic detection options
    parser_shi.add_argument("--all-2d", action="store_true", 
                           help="Execute SHI-2D method automatically")
    parser_shi.add_argument("--all-3d", action="store_true", 
                           help="Execute SHI-CT method automatically")
    
    # Processing options
    parser_shi.add_argument("--average", action="store_true", help="Apply averaging")
    parser_shi.add_argument("--export", action="store_true", help="Export results")
    parser_shi.add_argument("--angle-after", action="store_true", 
                           help="Apply angle correction after measurements")
    parser_shi.add_argument("--unwrap-phase", type=str, default=None, 
                           help="Phase unwrapping method")
    
    # Output options
    parser_shi.add_argument("--output-dir", type=str, 
                           help="Custom output directory (default: ~/Documents/CXI/CXI-DATA-ANALYSIS)")
    parser_shi.add_argument("--temp-dir", type=str, 
                           help="Custom temporary directory (default: ./tmp)")
    
    
    return main_parser


def determine_paths(args) -> Tuple[str, List[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Determine image paths based on arguments.
    
    Returns:
        Tuple of (processing_mode, images_paths, dark_path, flat_path, bright_path)
    """
    if args.all_2d or args.all_3d:
        processing_mode = "2d" if args.all_2d else "3d"
        logger.info("Executing SHI processing in %s mode", processing_mode.upper())
        
        measurement_directory = Path.cwd()
        images_paths = list((measurement_directory / "sample").iterdir())
        dark_path = measurement_directory / "dark"
        flat_path = measurement_directory / "flat"
        bright_path = measurement_directory / "bright"
        
        logger.info("Automatic image detection in directory: %s", measurement_directory)
    else:
        # Default to 2D mode for manual specification
        processing_mode = "2d"
        images_paths = list(Path(args.images).iterdir()) if args.images else []
        dark_path = Path(args.dark) if args.dark else None
        flat_path = Path(args.flat) if args.flat else None
        bright_path = Path(args.bright) if args.bright else None
        
        logger.info("Using user-specified image paths")
    
    # Validate paths exist
    dark_path = dark_path if dark_path and dark_path.exists() else None
    flat_path = flat_path if flat_path and flat_path.exists() else None
    bright_path = bright_path if bright_path and bright_path.exists() else None
    
    return processing_mode, images_paths, dark_path, flat_path, bright_path


def calculate_angle_correction(
    args, 
    flat_path: Optional[Path], 
    image_path: Path
) -> float:
    """Calculate angle correction if requested.
    
    Args:
        args: Command line arguments
        flat_path: Path to flat field images
        image_path: Path to sample images
        
    Returns:
        Rotation angle in degrees
    """
    if not args.angle_after:
        return 0.0
    
    path_to_ang = flat_path if flat_path is not None else image_path
    tif_ang_files = list(Path(path_to_ang).glob("*.tif"))
    
    if not tif_ang_files:
        logger.warning("No .tif files found for angle correction in %s", path_to_ang)
        return 0.0
    
    path_to_angle_correction = tif_ang_files[0]
    image_angle = io.imread(path_to_angle_correction)
    cords = angles_correction.extracting_coordinates_of_peaks(image_angle)
    deg = angles_correction.calculating_angles_of_peaks_average(cords)
    
    logger.info("Calculated rotation angle: %.2f degrees", deg)
    return float(deg)


def prepare_measurement_contexts(args) -> Tuple[List[SHIMeasurementContext], str]:
    """Phase 1: Handle all interactive operations and prepare contexts.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (measurement_contexts, processing_mode)
    """
    contexts = []
    
    # Determine processing mode and paths
    processing_mode, images_paths, dark_path, flat_path, bright_path = determine_paths(args)
    
    # Process each measurement
    for image_path in images_paths:
        logger.info("Preparing measurement: %s", image_path)
        
        # Validate TIFF files exist
        tif_files = list(Path(image_path).glob("*.tif"))
        if not tif_files:
            logger.error("No .tif files found in %s, skipping", image_path)
            continue
        
        # INTERACTIVE: Get ROI coordinates
        logger.info("Please select ROI for measurement: %s", image_path.name)
        crop_region = crop_tk.cropImage(tif_files[0])
        
        # COMPUTATIONAL: Calculate angle correction
        rotation_angle = calculate_angle_correction(args, flat_path, image_path)
        
        # Create context for this measurement
        context = SHIMeasurementContext(
            measurement_name=image_path.stem,
            images_path=image_path,
            dark_path=dark_path,
            flat_path=flat_path,
            bright_path=bright_path,
            crop_region=crop_region,
            rotation_angle=rotation_angle,
            mask_period=int(args.mask_period),
            unwrap_phase=args.unwrap_phase
        )
        contexts.append(context)
        
        logger.info("Prepared context for %s", image_path.stem)
    
    return contexts, processing_mode


def main():
    """Main entry point for the refactored SHI CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "calculate":
        try:
            # Phase 1: Interactive preparation
            logger.info("Phase 1: Preparing measurements (interactive)")
            measurement_contexts, processing_mode = prepare_measurement_contexts(args)
            
            if not measurement_contexts:
                logger.error("No valid measurements found")
                sys.exit(1)
            
            # Determine output and temp directories
            output_base_path = Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else \
                              Path.home() / "Documents" / "CXI" / "CXI-DATA-ANALYSIS"
            temp_directory = Path(args.temp_dir) if hasattr(args, 'temp_dir') and args.temp_dir else \
                            current_dir / "tmp"
            
            # Create batch context
            batch_context = SHIBatchContext(
                measurements=measurement_contexts,
                processing_mode=processing_mode,
                apply_averaging=args.average,
                export_results=args.export,
                output_base_path=output_base_path,
                temp_directory=temp_directory
            )
            
            # Phase 2: Non-interactive batch processing
            logger.info("Phase 2: Processing measurements (non-interactive)")
            orchestrator = SHIProcessingOrchestrator(logger=ConsoleLogger())
            results = orchestrator.process_batch(batch_context)
            
            # Display results
            print("\n" + "="*60)
            print(results.summary())
            print("="*60)
            
            # Exit with appropriate code
            sys.exit(0 if results.all_successful else 1)
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error("Fatal error: %s", str(e))
            sys.exit(1)
    
    elif args.command == "clean":
        logger.info("Clean command not yet implemented in refactored version")
        sys.exit(0)
    
    elif args.command == "morphostructural":
        logger.info("Morphostructural command not yet implemented in refactored version")
        sys.exit(0)
    
    elif args.command == "preprocessing":
        logger.info("Preprocessing command not yet implemented in refactored version")
        sys.exit(0)
    
    else:
        logger.error("Unknown command: %s", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
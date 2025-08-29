#!/usr/bin/env python3
"""
Test script for the SHI orchestrator with test data.
Uses predefined ROI coordinates to avoid interactive GUI.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add src to path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

from processing_context import SHIMeasurementContext, SHIBatchContext
from processing_orchestrator import SHIProcessingOrchestrator
from processing_interfaces import ConsoleLogger

def test_orchestrator_with_testdata():
    """Test the orchestrator with the testdata/measurements folder."""
    
    print("=" * 60)
    print("Testing SHI Orchestrator with testdata/measurements")
    print("=" * 60)
    
    # Set up paths
    test_dir = current_dir / "testdata" / "measurements"
    sample_dir = test_dir / "sample"
    dark_path = test_dir / "dark"
    flat_path = test_dir / "flat"
    bright_path = test_dir / "bright"
    
    # Get measurement directories
    measurement_dirs = [d for d in sample_dir.iterdir() if d.is_dir()]
    
    if not measurement_dirs:
        print("No measurement directories found in sample folder")
        return
    
    print(f"Found {len(measurement_dirs)} measurement(s):")
    for d in measurement_dirs:
        print(f"  - {d.name}")
    
    # Create measurement contexts with predefined ROI (no cropping)
    contexts = []
    for measurement_dir in measurement_dirs:
        print(f"\nPreparing context for: {measurement_dir.name}")
        
        # Check if TIFF files exist
        tif_files = list(measurement_dir.glob("*.tif"))
        if not tif_files:
            print(f"  No TIFF files found, skipping")
            continue
        
        print(f"  Found {len(tif_files)} TIFF files")
        
        # Create context with no cropping (full image)
        context = SHIMeasurementContext(
            measurement_name=measurement_dir.name,
            images_path=measurement_dir,
            dark_path=dark_path,
            flat_path=flat_path,
            bright_path=bright_path,
            crop_region=(0, -1, 0, -1),  # No cropping
            rotation_angle=0.0,  # No rotation
            mask_period=10,  # Default mask period
            unwrap_phase=None  # Default unwrapping
        )
        contexts.append(context)
    
    if not contexts:
        print("No valid measurements to process")
        return
    
    # Create batch context
    output_dir = current_dir / "test_output" / f"test_{int(time.time())}"
    temp_dir = current_dir / "test_tmp"
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Temp directory: {temp_dir}")
    
    batch_context = SHIBatchContext(
        measurements=contexts,
        processing_mode="2d",
        apply_averaging=False,
        export_results=True,
        output_base_path=output_dir,
        temp_directory=temp_dir
    )
    
    # Process with orchestrator
    print("\n" + "=" * 60)
    print("Starting processing with orchestrator...")
    print("=" * 60)
    
    orchestrator = SHIProcessingOrchestrator(logger=ConsoleLogger())
    
    try:
        start_time = time.time()
        results = orchestrator.process_batch(batch_context)
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("PROCESSING RESULTS")
        print("=" * 60)
        print(results.summary())
        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
        
        # List output files created
        if results.all_successful:
            print("\n" + "=" * 60)
            print("OUTPUT FILES CREATED")
            print("=" * 60)
            
            for measurement in results.get_successful_measurements():
                print(f"\n{measurement.measurement_name}:")
                for contrast_type, path in measurement.output_paths.items():
                    if path.exists():
                        file_count = len(list(path.glob("**/*.tif")))
                        print(f"  {contrast_type}: {file_count} files in {path}")
        
        return results.all_successful
        
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_orchestrator_with_testdata()
    sys.exit(0 if success else 1)
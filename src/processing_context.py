"""
Data classes for SHI processing context and results.

This module contains the data structures used to pass configuration
and results between the CLI interface and the processing orchestrator.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class SHIMeasurementContext:
    """Context for processing a single measurement.
    
    Contains all necessary data for processing one measurement directory,
    including pre-computed interactive elements like ROI and rotation angle.
    """
    measurement_name: str
    images_path: Path
    dark_path: Optional[Path]
    flat_path: Optional[Path] 
    bright_path: Optional[Path]
    crop_region: Tuple[int, int, int, int]  # (y0, y1, x0, x1) - pre-computed
    rotation_angle: float  # pre-computed angle in degrees
    mask_period: int
    unwrap_phase: Optional[str]
    
    def __post_init__(self):
        """Validate the context data."""
        if not self.images_path.exists():
            raise ValueError(f"Images path does not exist: {self.images_path}")
        
        if self.dark_path and not self.dark_path.exists():
            raise ValueError(f"Dark path does not exist: {self.dark_path}")
            
        if self.flat_path and not self.flat_path.exists():
            raise ValueError(f"Flat path does not exist: {self.flat_path}")
            
        if self.bright_path and not self.bright_path.exists():
            raise ValueError(f"Bright path does not exist: {self.bright_path}")
            
        if self.mask_period <= 0:
            raise ValueError(f"Mask period must be positive: {self.mask_period}")


@dataclass
class SHIBatchContext:
    """Context for batch processing multiple measurements.
    
    Contains all measurements to be processed and global configuration
    that applies to the entire batch.
    """
    measurements: List[SHIMeasurementContext]
    processing_mode: str  # "2d" or "3d"
    apply_averaging: bool = False
    export_results: bool = False
    output_base_path: Path = field(default_factory=lambda: Path.home() / "Documents" / "CXI" / "CXI-DATA-ANALYSIS")
    temp_directory: Path = field(default_factory=lambda: Path.cwd() / "tmp")
    
    def __post_init__(self):
        """Validate the batch context."""
        if not self.measurements:
            raise ValueError("At least one measurement is required")
            
        if self.processing_mode not in ("2d", "3d"):
            raise ValueError(f"Processing mode must be '2d' or '3d', got: {self.processing_mode}")
            
        # Create output and temp directories if they don't exist
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        self.temp_directory.mkdir(parents=True, exist_ok=True)


@dataclass
class MeasurementResult:
    """Result from processing a single measurement."""
    measurement_name: str
    success: bool
    output_paths: Dict[str, Path]  # Maps contrast type to output directory
    processing_time: float
    error: Optional[str] = None
    
    def __repr__(self):
        status = "SUCCESS" if self.success else f"FAILED: {self.error}"
        return f"MeasurementResult({self.measurement_name}: {status})"


@dataclass
class SHIProcessingResults:
    """Results from batch processing."""
    measurements: List[MeasurementResult]
    total_processing_time: float
    success_count: int = field(init=False)
    failure_count: int = field(init=False)
    
    def __post_init__(self):
        """Calculate success and failure counts."""
        self.success_count = sum(1 for m in self.measurements if m.success)
        self.failure_count = sum(1 for m in self.measurements if not m.success)
    
    @property
    def all_successful(self) -> bool:
        """Check if all measurements were processed successfully."""
        return self.failure_count == 0
    
    def get_failed_measurements(self) -> List[MeasurementResult]:
        """Get list of failed measurements."""
        return [m for m in self.measurements if not m.success]
    
    def get_successful_measurements(self) -> List[MeasurementResult]:
        """Get list of successful measurements."""
        return [m for m in self.measurements if m.success]
    
    def summary(self) -> str:
        """Generate a summary of the processing results."""
        lines = [
            f"Processing Results Summary:",
            f"  Total measurements: {len(self.measurements)}",
            f"  Successful: {self.success_count}",
            f"  Failed: {self.failure_count}",
            f"  Total time: {self.total_processing_time:.2f}s"
        ]
        
        if self.failure_count > 0:
            lines.append("\nFailed measurements:")
            for m in self.get_failed_measurements():
                lines.append(f"  - {m.measurement_name}: {m.error}")
        
        return "\n".join(lines)
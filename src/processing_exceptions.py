"""
Custom exceptions for SHI processing.

This module defines specific exception types for different failure
scenarios in the SHI processing pipeline.
"""

from typing import Optional
from pathlib import Path


class SHIProcessingError(Exception):
    """Base exception class for all SHI processing errors."""
    
    def __init__(self, message: str, measurement_name: Optional[str] = None):
        """Initialize the processing error.
        
        Args:
            message: Error message
            measurement_name: Name of the measurement that caused the error
        """
        self.measurement_name = measurement_name
        super().__init__(message)


class ImageNotFoundError(SHIProcessingError):
    """Raised when required image files are not found."""
    
    def __init__(self, path: Path, image_type: str, measurement_name: Optional[str] = None):
        """Initialize the error.
        
        Args:
            path: Path where images were expected
            image_type: Type of images (e.g., 'sample', 'dark', 'flat', 'bright')
            measurement_name: Name of the measurement
        """
        self.path = path
        self.image_type = image_type
        message = f"No {image_type} images found at: {path}"
        super().__init__(message, measurement_name)


class InvalidParameterError(SHIProcessingError):
    """Raised when invalid parameters are provided."""
    
    def __init__(self, parameter_name: str, value, reason: str):
        """Initialize the error.
        
        Args:
            parameter_name: Name of the invalid parameter
            value: The invalid value
            reason: Explanation of why the value is invalid
        """
        self.parameter_name = parameter_name
        self.value = value
        message = f"Invalid parameter '{parameter_name}' = {value}: {reason}"
        super().__init__(message)


class ROISelectionError(SHIProcessingError):
    """Raised when ROI selection fails or is cancelled."""
    
    def __init__(self, image_path: Path, reason: str = "ROI selection cancelled"):
        """Initialize the error.
        
        Args:
            image_path: Path to the image for ROI selection
            reason: Reason for the failure
        """
        self.image_path = image_path
        message = f"ROI selection failed for {image_path}: {reason}"
        super().__init__(message)


class CorrectionError(SHIProcessingError):
    """Raised when image correction operations fail."""
    
    def __init__(self, correction_type: str, reason: str, measurement_name: Optional[str] = None):
        """Initialize the error.
        
        Args:
            correction_type: Type of correction (e.g., 'darkfield', 'brightfield', 'flatfield')
            reason: Reason for the failure
            measurement_name: Name of the measurement
        """
        self.correction_type = correction_type
        message = f"{correction_type} correction failed: {reason}"
        super().__init__(message, measurement_name)


class HarmonicExtractionError(SHIProcessingError):
    """Raised when harmonic extraction from FFT fails."""
    
    def __init__(self, reason: str, measurement_name: Optional[str] = None):
        """Initialize the error.
        
        Args:
            reason: Reason for the failure
            measurement_name: Name of the measurement
        """
        message = f"Harmonic extraction failed: {reason}"
        super().__init__(message, measurement_name)


class ContrastRetrievalError(SHIProcessingError):
    """Raised when contrast retrieval fails."""
    
    def __init__(self, contrast_type: str, reason: str, measurement_name: Optional[str] = None):
        """Initialize the error.
        
        Args:
            contrast_type: Type of contrast (e.g., 'absorption', 'scattering', 'phase')
            reason: Reason for the failure
            measurement_name: Name of the measurement
        """
        self.contrast_type = contrast_type
        message = f"Failed to retrieve {contrast_type} contrast: {reason}"
        super().__init__(message, measurement_name)


class PhaseUnwrappingError(SHIProcessingError):
    """Raised when phase unwrapping fails."""
    
    def __init__(self, algorithm: str, reason: str, measurement_name: Optional[str] = None):
        """Initialize the error.
        
        Args:
            algorithm: Name of the unwrapping algorithm
            reason: Reason for the failure
            measurement_name: Name of the measurement
        """
        self.algorithm = algorithm
        message = f"Phase unwrapping failed with algorithm '{algorithm}': {reason}"
        super().__init__(message, measurement_name)


class DirectoryCreationError(SHIProcessingError):
    """Raised when directory creation fails."""
    
    def __init__(self, path: Path, reason: str):
        """Initialize the error.
        
        Args:
            path: Path that could not be created
            reason: Reason for the failure
        """
        self.path = path
        message = f"Failed to create directory {path}: {reason}"
        super().__init__(message)


class FileWriteError(SHIProcessingError):
    """Raised when file writing operations fail."""
    
    def __init__(self, file_path: Path, reason: str, measurement_name: Optional[str] = None):
        """Initialize the error.
        
        Args:
            file_path: Path to the file that could not be written
            reason: Reason for the failure
            measurement_name: Name of the measurement
        """
        self.file_path = file_path
        message = f"Failed to write file {file_path}: {reason}"
        super().__init__(message, measurement_name)


class ProcessingModeError(SHIProcessingError):
    """Raised when an invalid processing mode is specified."""
    
    def __init__(self, mode: str):
        """Initialize the error.
        
        Args:
            mode: The invalid processing mode
        """
        self.mode = mode
        message = f"Invalid processing mode '{mode}'. Must be '2d' or '3d'"
        super().__init__(message)


class BatchProcessingError(Exception):
    """Raised when batch processing encounters critical errors."""
    
    def __init__(self, message: str, failed_measurements: list):
        """Initialize the error.
        
        Args:
            message: Error message
            failed_measurements: List of measurement names that failed
        """
        self.failed_measurements = failed_measurements
        super().__init__(message)
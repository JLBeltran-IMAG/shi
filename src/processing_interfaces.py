"""
Abstract interfaces for SHI processing components.

This module defines abstract interfaces that allow the orchestrator
to work with different implementations of ROI selection, logging,
and other external dependencies.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import logging


class ROIProvider(ABC):
    """Abstract interface for ROI (Region of Interest) selection.
    
    Implementations can provide ROI through different mechanisms:
    - Interactive GUI (Tkinter, Qt, etc.)
    - Pre-configured values
    - Web-based selection
    - Automated detection
    """
    
    @abstractmethod
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Get ROI coordinates for an image.
        
        Args:
            image_path: Path to the image file for ROI selection
            
        Returns:
            Tuple of (y0, y1, x0, x1) coordinates defining the ROI
        """
        pass


class DefaultROIProvider(ROIProvider):
    """Default ROI provider that returns full image (no cropping)."""
    
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Return coordinates for full image (no cropping).
        
        Args:
            image_path: Path to the image file (unused)
            
        Returns:
            (0, -1, 0, -1) indicating full image should be used
        """
        return (0, -1, 0, -1)


class PredefinedROIProvider(ROIProvider):
    """ROI provider that uses predefined coordinates."""
    
    def __init__(self, roi_coords: Tuple[int, int, int, int]):
        """Initialize with predefined ROI coordinates.
        
        Args:
            roi_coords: Tuple of (y0, y1, x0, x1) coordinates
        """
        self.roi_coords = roi_coords
    
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Return predefined ROI coordinates.
        
        Args:
            image_path: Path to the image file (unused)
            
        Returns:
            The predefined ROI coordinates
        """
        return self.roi_coords


class ProcessingLogger(ABC):
    """Abstract interface for logging during processing.
    
    Allows different logging implementations:
    - Console logging
    - File logging
    - Web API logging
    - Database logging
    """
    
    @abstractmethod
    def info(self, message: str, *args, **kwargs) -> None:
        """Log an informational message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message."""
        pass
    
    @abstractmethod
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message."""
        pass


class ConsoleLogger(ProcessingLogger):
    """Logger implementation that outputs to console."""
    
    def __init__(self, name: str = "SHI"):
        """Initialize console logger.
        
        Args:
            name: Logger name for identification
        """
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message to console."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message to console."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message to console."""
        self.logger.error(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message to console."""
        self.logger.debug(message, *args, **kwargs)


class SilentLogger(ProcessingLogger):
    """Logger implementation that suppresses all output."""
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Suppress info message."""
        pass
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Suppress warning message."""
        pass
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Suppress error message."""
        pass
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Suppress debug message."""
        pass


class FileLogger(ProcessingLogger):
    """Logger implementation that outputs to a file."""
    
    def __init__(self, log_file: Path, name: str = "SHI"):
        """Initialize file logger.
        
        Args:
            log_file: Path to the log file
            name: Logger name for identification
        """
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message to file."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message to file."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message to file."""
        self.logger.error(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message to file."""
        self.logger.debug(message, *args, **kwargs)


class MessageCollector(ProcessingLogger):
    """Logger that collects messages for later retrieval.
    
    Useful for web applications or APIs that need to return
    processing logs to the client.
    """
    
    def __init__(self):
        """Initialize message collector."""
        self.messages = {
            'info': [],
            'warning': [],
            'error': [],
            'debug': []
        }
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Collect info message."""
        formatted = message % args if args else message
        self.messages['info'].append(formatted)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Collect warning message."""
        formatted = message % args if args else message
        self.messages['warning'].append(formatted)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Collect error message."""
        formatted = message % args if args else message
        self.messages['error'].append(formatted)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Collect debug message."""
        formatted = message % args if args else message
        self.messages['debug'].append(formatted)
    
    def get_messages(self, level: Optional[str] = None) -> list:
        """Get collected messages.
        
        Args:
            level: Specific level to retrieve, or None for all
            
        Returns:
            List of messages for the specified level or all messages
        """
        if level:
            return self.messages.get(level, [])
        
        all_messages = []
        for level_messages in self.messages.values():
            all_messages.extend(level_messages)
        return all_messages
    
    def clear(self) -> None:
        """Clear all collected messages."""
        for level in self.messages:
            self.messages[level].clear()
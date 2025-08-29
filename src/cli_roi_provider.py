"""
ROI provider implementation for CLI that uses Tkinter.

This module provides the interactive ROI selection functionality
for the CLI version of SHI, keeping UI dependencies separate from
the core processing logic.
"""

from pathlib import Path
from typing import Tuple
from processing_interfaces import ROIProvider
import crop_tk


class TkinterROIProvider(ROIProvider):
    """ROI provider that uses Tkinter for interactive selection."""
    
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Get ROI coordinates using Tkinter interface.
        
        Args:
            image_path: Path to the image file for ROI selection
            
        Returns:
            Tuple of (y0, y1, x0, x1) coordinates defining the ROI
        """
        return crop_tk.cropImage(image_path)


class CachedROIProvider(ROIProvider):
    """ROI provider that caches the first ROI selection and reuses it.
    
    Useful when you want to apply the same ROI to multiple measurements.
    """
    
    def __init__(self, base_provider: ROIProvider):
        """Initialize with a base provider.
        
        Args:
            base_provider: The underlying ROI provider to use for actual selection
        """
        self.base_provider = base_provider
        self.cached_roi = None
        self.cache_enabled = True
    
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Get ROI coordinates, using cache if available.
        
        Args:
            image_path: Path to the image file for ROI selection
            
        Returns:
            Tuple of (y0, y1, x0, x1) coordinates defining the ROI
        """
        if self.cached_roi is not None and self.cache_enabled:
            return self.cached_roi
        
        roi = self.base_provider.get_roi_coordinates(image_path)
        if self.cache_enabled:
            self.cached_roi = roi
        
        return roi
    
    def clear_cache(self):
        """Clear the cached ROI."""
        self.cached_roi = None
    
    def disable_cache(self):
        """Disable caching (each call will get fresh ROI)."""
        self.cache_enabled = False
        self.cached_roi = None
    
    def enable_cache(self):
        """Enable caching."""
        self.cache_enabled = True


class ConfigFileROIProvider(ROIProvider):
    """ROI provider that reads coordinates from a configuration file.
    
    Useful for batch processing with predefined ROIs.
    """
    
    def __init__(self, config_file: Path):
        """Initialize with a configuration file.
        
        Args:
            config_file: Path to the configuration file containing ROI definitions
        """
        self.config_file = config_file
        self.roi_mappings = self._load_config()
    
    def _load_config(self) -> dict:
        """Load ROI mappings from configuration file.
        
        Expected format (JSON):
        {
            "default": [0, -1, 0, -1],
            "measurement_01": [100, 500, 100, 500],
            "measurement_02": [150, 450, 150, 450]
        }
        
        Returns:
            Dictionary mapping measurement names to ROI coordinates
        """
        import json
        
        if not self.config_file.exists():
            return {"default": (0, -1, 0, -1)}
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        # Convert lists to tuples
        return {k: tuple(v) for k, v in config.items()}
    
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Get ROI coordinates from configuration.
        
        Args:
            image_path: Path to the image file (used to determine measurement name)
            
        Returns:
            Tuple of (y0, y1, x0, x1) coordinates defining the ROI
        """
        measurement_name = image_path.parent.stem
        
        # Try to find specific ROI for this measurement
        if measurement_name in self.roi_mappings:
            return self.roi_mappings[measurement_name]
        
        # Fall back to default
        return self.roi_mappings.get("default", (0, -1, 0, -1))
    
    def save_roi(self, measurement_name: str, roi: Tuple[int, int, int, int]):
        """Save an ROI definition to the configuration.
        
        Args:
            measurement_name: Name of the measurement
            roi: ROI coordinates to save
        """
        import json
        
        self.roi_mappings[measurement_name] = roi
        
        # Convert tuples to lists for JSON serialization
        config = {k: list(v) for k, v in self.roi_mappings.items()}
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
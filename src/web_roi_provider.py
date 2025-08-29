"""
ROI provider implementation for web-based interface.

This module provides ROI selection functionality that works with
the web interface, avoiding GUI dependencies like Tkinter.
"""

from pathlib import Path
from typing import Tuple, Optional
import json
import tempfile
import os
from processing_interfaces import ROIProvider


class WebROIProvider(ROIProvider):
    """ROI provider that uses web-based coordinate storage."""
    
    def __init__(self):
        """Initialize web ROI provider."""
        self.roi_storage_dir = Path(tempfile.gettempdir()) / "shi_web_roi"
        self.roi_storage_dir.mkdir(exist_ok=True)
    
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Get ROI coordinates from web storage.
        
        Args:
            image_path: Path to the image file for ROI selection
            
        Returns:
            Tuple of (y0, y1, x0, x1) coordinates defining the ROI
        """
        # Look for ROI file associated with this processing session
        roi_files = list(self.roi_storage_dir.glob("roi_*.json"))
        
        if roi_files:
            # Use most recent ROI file
            roi_file = max(roi_files, key=os.path.getmtime)
            with open(roi_file, 'r') as f:
                roi_data = json.load(f)
            return tuple(roi_data["coordinates"])
        
        # Default: use entire image (no cropping)
        return (0, -1, 0, -1)
    
    def save_roi_coordinates(self, job_id: str, coordinates: Tuple[int, int, int, int], image_info: dict = None) -> Path:
        """Save ROI coordinates for a specific job.
        
        Args:
            job_id: Unique identifier for the processing job
            coordinates: ROI coordinates (y0, y1, x0, x1)
            image_info: Optional metadata about the image
            
        Returns:
            Path to the saved ROI file
        """
        roi_data = {
            "job_id": job_id,
            "coordinates": coordinates,
            "timestamp": str(Path().resolve()),
            "image_info": image_info or {}
        }
        
        roi_file = self.roi_storage_dir / f"roi_{job_id}.json"
        with open(roi_file, 'w') as f:
            json.dump(roi_data, f, indent=2)
        
        return roi_file
    
    def get_roi_for_job(self, job_id: str) -> Optional[Tuple[int, int, int, int]]:
        """Get ROI coordinates for a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ROI coordinates if found, None otherwise
        """
        roi_file = self.roi_storage_dir / f"roi_{job_id}.json"
        
        if roi_file.exists():
            with open(roi_file, 'r') as f:
                roi_data = json.load(f)
            return tuple(roi_data["coordinates"])
        
        return None
    
    def clear_roi_for_job(self, job_id: str):
        """Clear ROI data for a specific job.
        
        Args:
            job_id: Job identifier
        """
        roi_file = self.roi_storage_dir / f"roi_{job_id}.json"
        if roi_file.exists():
            roi_file.unlink()
    
    def cleanup_old_rois(self, max_age_hours: int = 24):
        """Clean up old ROI files.
        
        Args:
            max_age_hours: Maximum age in hours before ROI files are deleted
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for roi_file in self.roi_storage_dir.glob("roi_*.json"):
            if current_time - roi_file.stat().st_mtime > max_age_seconds:
                roi_file.unlink()


class PresetROIProvider(ROIProvider):
    """ROI provider that uses preset coordinates."""
    
    def __init__(self, coordinates: Tuple[int, int, int, int]):
        """Initialize with preset coordinates.
        
        Args:
            coordinates: Fixed ROI coordinates (y0, y1, x0, x1)
        """
        self.coordinates = coordinates
    
    def get_roi_coordinates(self, image_path: Path) -> Tuple[int, int, int, int]:
        """Return preset ROI coordinates.
        
        Args:
            image_path: Path to the image file (unused)
            
        Returns:
            Preset ROI coordinates
        """
        return self.coordinates
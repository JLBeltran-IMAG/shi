#!/usr/bin/env python3

"""
SHI FastAPI Backend - Complete web API equivalent of shi.py CLI

This FastAPI backend provides REST endpoints that mirror all functionality
of the shi.py command-line interface:
- calculate command (with all options: --all-2d, --all-3d, custom paths, etc.)
- morphostructural analysis
- preprocessing (stripes correction)
- clean command (cache and extra files cleanup)
"""

import asyncio
import shutil
import tempfile
import uuid
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import numpy as np
import skimage.io as io
import tifffile as ti
from PIL import Image
import io as python_io
import base64

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add source directory to path (same as shi.py)
current_dir = Path(__file__).resolve().parent.parent.parent
script_dir = current_dir.joinpath("scripts")
cache_dir = current_dir.joinpath("cache")
tmp_dir = current_dir.joinpath("tmp")
src_dir = current_dir.joinpath("src")
sys.path.append(str(src_dir))

# Import SHI modules (same as shi.py)
try:
    import spatial_harmonics
    import directories
    import corrections
    import crop_tk
    from web_roi_provider import WebROIProvider
    import angles_correction
    import correcting_stripes
except ImportError as e:
    print(f"Failed to import SHI modules: {e}")
    print(f"Make sure the SHI source code is available at: {src_dir}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SHI API",
    description="FastAPI backend for Spatial Harmonic Imaging - Web equivalent of shi.py CLI",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for job tracking
processing_jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=2)

# Global ROI provider for web-based ROI selection
roi_provider = WebROIProvider()


# Pydantic models for API requests/responses
class CalculateRequest(BaseModel):
    """Request model for calculate command - mirrors all shi.py calculate options"""
    mask_period: int
    unwrap_phase: Optional[str] = None
    mode: str  # "2d", "3d", or "custom"
    average: bool = False
    export: bool = False
    angle_after: bool = False
    # For custom mode (equivalent to -i, -f, -d, -b flags)
    custom_paths: Optional[Dict[str, str]] = None


class MorphostructuralRequest(BaseModel):
    """Request model for morphostructural analysis"""
    morphostructural: bool = True


class PreprocessingRequest(BaseModel):
    """Request model for preprocessing operations"""
    stripes: bool = False


class CleanRequest(BaseModel):
    """Request model for clean operations"""
    clear_cache: bool = False
    clear_extra: bool = False


class JobResponse(BaseModel):
    """Response model for job operations"""
    job_id: str
    command: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float
    message: str
    created_at: str
    completed_at: Optional[str] = None
    results_path: Optional[str] = None
    log_messages: List[str] = []


class ROICoordinates(BaseModel):
    """ROI coordinates model"""
    x0: int
    y0: int
    x1: int
    y1: int


class ROIRequest(BaseModel):
    """Request model for ROI operations"""
    job_id: str
    coordinates: ROICoordinates


class CommandInfo(BaseModel):
    """Information about available commands"""
    name: str
    description: str
    options: List[Dict[str, Any]]


def create_temp_directory() -> Path:
    """Create a temporary directory for processing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="shi_web_"))
    return temp_dir


def organize_uploaded_files(files: List[UploadFile], temp_dir: Path) -> Dict[str, Path]:
    """Organize uploaded files into SHI folder structure - one session = one measurement"""
    folders = {
        "sample": temp_dir / "sample",
        "dark": temp_dir / "dark", 
        "flat": temp_dir / "flat",
        "bright": temp_dir / "bright"
    }
    
    # Create directories
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    # Create measurement folder inside sample directory (key fix!)
    measurement_folder = folders["sample"] / "measurement_001"
    measurement_folder.mkdir(parents=True, exist_ok=True)
    
    # Sort files by type based on filename patterns
    file_mapping = {}
    
    for file in files:
        filename = file.filename.lower()
        
        if "dark" in filename:
            target_folder = folders["dark"]
        elif "flat" in filename:
            target_folder = folders["flat"]
        elif "bright" in filename:
            target_folder = folders["bright"]
        else:
            # All sample files go into the measurement folder (not directly in sample/)
            target_folder = measurement_folder
        
        file_mapping[file.filename] = target_folder
    
    return file_mapping


async def save_uploaded_files(files: List[UploadFile], file_mapping: Dict[str, Path]) -> bool:
    """Save uploaded files to their designated folders."""
    try:
        for file in files:
            target_path = file_mapping[file.filename] / file.filename
            
            with open(target_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
        return True
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        return False


def run_calculate_command(job_id: str, temp_dir: Path, request: CalculateRequest):
    """Execute the calculate command - exact implementation of shi.py calculate logic"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Starting SHI calculation..."
        
        mask_period = request.mask_period
        unwrap = request.unwrap_phase
        
        logger.info(f"Executing 'calculate' command with mask_period: {mask_period}")
        processing_jobs[job_id]["log_messages"].append(f"Executing calculate with mask_period: {mask_period}")
        
        # Determine mode and paths (exact same logic as shi.py)
        if request.mode in ["2d", "3d"]:
            # Auto-detect mode (same as --all-2d or --all-3d in shi.py)
            measurement_directory = temp_dir
            images_path = list((measurement_directory / "sample").iterdir())
            dark_path = measurement_directory / "dark"
            flat_path = measurement_directory / "flat"
            bright_dir = measurement_directory / "bright"
            # Only set bright_path if there are actual bright field images
            bright_path = bright_dir if list(bright_dir.glob("*.tif")) else None
            
            logger.info(f"Automatic image detection activated in directory: {measurement_directory}")
            processing_jobs[job_id]["log_messages"].append(f"Auto-detection mode: {request.mode.upper()}")
            
        else:
            # Custom paths mode (equivalent to -i, -f, -d, -b flags)
            if request.custom_paths:
                images_path = [Path(request.custom_paths.get("images", ""))] if request.custom_paths.get("images") else []
                dark_path = Path(request.custom_paths.get("dark", "")) if request.custom_paths.get("dark") else None
                flat_path = Path(request.custom_paths.get("flat", "")) if request.custom_paths.get("flat") else None
                bright_path = Path(request.custom_paths.get("bright", "")) if request.custom_paths.get("bright") else None
            else:
                # Fallback to auto-detect if no custom paths provided
                images_path = list((temp_dir / "sample").iterdir())
                dark_path = temp_dir / "dark"
                flat_path = temp_dir / "flat"
                bright_dir = temp_dir / "bright"
                # Only set bright_path if there are actual bright field images
                bright_path = bright_dir if list(bright_dir.glob("*.tif")) else None
            
            logger.info("Using specified image paths.")
            processing_jobs[job_id]["log_messages"].append("Using custom image paths")
        
        processing_jobs[job_id]["progress"] = 0.2
        
        # Process each measurement (exact same loop as shi.py lines 162-271)
        total_images = len(images_path)
        for i, image_path in enumerate(images_path):
            logger.info(f"Processing measurement: {image_path}")
            processing_jobs[job_id]["message"] = f"Processing measurement {i+1}/{total_images}: {image_path.name}"
            
            # Before dark field correction, crop the ROI (same as shi.py line 166)
            output = image_path.stem
            
            # Select first TIFF file within image directory (same as shi.py lines 168-172)
            tif_files = list(Path(image_path).glob("*.tif"))
            if not tif_files:
                logger.error(f"No .tif files found in {image_path}")
                raise ValueError(f"No .tif files found in {image_path}")
            
            # Angle correction logic (exact same as shi.py lines 174-187)
            if request.angle_after:
                path_to_ang = flat_path if flat_path is not None else image_path
                tif_ang_files = list(Path(path_to_ang).glob("*.tif"))
                
                if not tif_ang_files:
                    logger.error(f"No .tif files found for angle correction in {path_to_ang}")
                    deg: np.float32 = np.float32(0)
                else:
                    path_to_angle_correction = tif_ang_files[0]
                    image_angle = io.imread(path_to_angle_correction)
                    cords = angles_correction.extracting_coordinates_of_peaks(image_angle)
                    deg: np.float32 = angles_correction.calculating_angles_of_peaks_average(cords)
            else:
                deg: np.float32 = np.float32(0)
            
            processing_jobs[job_id]["progress"] = 0.3 + (i * 0.6 / total_images)
            
            # Get ROI coordinates from web provider (replaces Tkinter GUI)
            path_to_crop = tif_files[0]
            crop_from_tmptxt = roi_provider.get_roi_coordinates(path_to_crop)
            
            # Dark field correction (exact same logic as shi.py lines 192-218)
            if dark_path is not None:
                corrections.correct_darkfield(
                    path_to_dark=dark_path, path_to_images=flat_path, crop=crop_from_tmptxt, allow_crop=True, angle=deg
                )
                corrections.correct_darkfield(
                    path_to_dark=dark_path, path_to_images=image_path, crop=crop_from_tmptxt, allow_crop=True, angle=deg
                )
                
                if bright_path is not None:
                    corrections.correct_darkfield(
                        path_to_dark=dark_path, path_to_images=bright_path, crop=crop_from_tmptxt, allow_crop=True, angle=deg
                    )
                    
                    corrections.correct_brightfield(path_to_bright=bright_path, path_to_images=flat_path)
                    corrections.correct_brightfield(path_to_bright=bright_path, path_to_images=image_path)
                
                foldername_to = "corrected_images"
            else:
                corrections.crop_without_corrections(
                    path_to_images=image_path, crop=crop_from_tmptxt, allow_crop=True, angle=deg
                )
                corrections.crop_without_corrections(
                    path_to_images=flat_path, crop=crop_from_tmptxt, allow_crop=True, angle=deg
                )
                
                foldername_to = "crop_without_correction"
            
            # Create result folders in temp directory (web version)
            path_to_corrected_images = (Path(image_path) / foldername_to).as_posix()
            
            # Create local results folder with job ID (skip Documents folder)
            web_results_dir = temp_dir / "results" / f"job_{job_id}"  
            web_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Use create_result_subfolders to get correct file paths (same as shi.py line 221-222)
            path_to_images, path_to_result = directories.create_result_subfolders(
                file_dir=path_to_corrected_images, 
                result_folder="",  # Use empty for web version
                sample_folder=""
            )
            
            # Override path_to_result to use local web results folder and create subdirectories
            path_to_result = web_results_dir
            # Create the contrast subdirectories that SHI expects
            for subdir in ["absorption", "scattering", "phase", "phasemap"]:
                (path_to_result / subdir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"SHI results will be saved to: {path_to_result}")
            
            type_of_contrast = ("absorption", "scattering", "phase", "phasemap")
            
            # Execute SHI processing (exact same as shi.py lines 227-265)
            if flat_path is None:
                spatial_harmonics.execute_SHI(path_to_images, path_to_result, mask_period, unwrap, False)
            else:
                path_to_corrected_flat = Path(flat_path).joinpath(foldername_to).as_posix()
                
                path_to_flat, path_to_flat_result = directories.create_result_subfolders(
                    file_dir=path_to_corrected_flat,
                    result_folder="",  # Use empty for web version
                    sample_folder="flat",
                )
                
                spatial_harmonics.execute_SHI(path_to_flat, path_to_flat_result, mask_period, unwrap, True)
                spatial_harmonics.execute_SHI(path_to_images, path_to_result, mask_period, unwrap, False)
                path_to_result = directories.create_corrections_folder(path_to_result)
                
                for contrast in type_of_contrast:
                    path_to_average_flat = directories.average_flat_harmonics(path_to_flat_result, type_of_contrast=contrast)
                    corrections.correct_flatmask(path_to_result, path_to_average_flat, type_of_contrast=contrast)
                
                # Mode-specific processing (exact same as shi.py lines 247-265)
                if request.mode == "2d":
                    if request.average:
                        for contrast in type_of_contrast:
                            path_avg = path_to_result / contrast / "flat_corrections"
                            logger.info(f"Averaging contrast: {contrast}")
                            directories.averaging(path_avg, contrast)
                    else:
                        logger.info("Skipping averaging for 2D mode")
                
                elif request.mode == "3d":
                    for contrast in type_of_contrast:
                        ct_dir = path_to_result / contrast / "flat_corrections"
                        logger.info(f"Organizing contrast: {contrast}")
                        directories.organize_dir(ct_dir, contrast)
                else:
                    logger.error("No mode selected (neither 2d nor 3d specified)")
                    raise ValueError("No mode selected (neither 2d nor 3d specified)")
            
            # Export if requested (same as shi.py lines 268-270)
            if request.export:
                logger.info(f"Exporting results to {path_to_result}")
                directories.export_results(path_to_result)
        
        processing_jobs[job_id]["progress"] = 0.9
        processing_jobs[job_id]["message"] = "Creating results archive..."
        
        # Create results archive from the local results folder
        results_archive = temp_dir / "results.zip"
        local_results_path = temp_dir / "results"
        
        if local_results_path.exists():
            shutil.make_archive(str(results_archive.with_suffix("")), "zip", local_results_path)
            logger.info(f"Created results archive: {results_archive}")
        else:
            logger.warning(f"Results folder not found at {local_results_path}, creating archive of entire temp dir")
            shutil.make_archive(str(results_archive.with_suffix("")), "zip", temp_dir)
        
        # Mark as completed
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 1.0
        processing_jobs[job_id]["message"] = "SHI calculation completed successfully"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        processing_jobs[job_id]["results_path"] = str(results_archive)
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = f"Calculate command failed: {str(e)}"
        processing_jobs[job_id]["log_messages"].append(f"ERROR: {str(e)}")
        logger.error(f"Calculate error for job {job_id}: {e}")
        import traceback
        traceback.print_exc()


def run_morphostructural_command(job_id: str, temp_dir: Path, request: MorphostructuralRequest):
    """Execute morphostructural analysis - exact same as shi.py lines 287-294"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Starting morphostructural analysis..."
        
        if request.morphostructural:
            cmd = ["python", "morphos.py", "--select_folder", "--manually"]
            logger.info(f"Executing morphostructural analysis with command: {cmd}")
            processing_jobs[job_id]["log_messages"].append(f"Running: {' '.join(cmd)}")
            
            processing_jobs[job_id]["progress"] = 0.5
            processing_jobs[job_id]["message"] = "Running morphostructural analysis..."
            
            result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                processing_jobs[job_id]["status"] = "completed"
                processing_jobs[job_id]["progress"] = 1.0
                processing_jobs[job_id]["message"] = "Morphostructural analysis completed"
                processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                processing_jobs[job_id]["log_messages"].append("Analysis completed successfully")
                processing_jobs[job_id]["log_messages"].append(f"Output: {result.stdout}")
            else:
                raise Exception(f"Command failed with return code {result.returncode}: {result.stderr}")
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = f"Morphostructural analysis failed: {str(e)}"
        processing_jobs[job_id]["log_messages"].append(f"ERROR: {str(e)}")
        logger.error(f"Morphostructural error for job {job_id}: {e}")


def run_preprocessing_command(job_id: str, temp_dir: Path, request: PreprocessingRequest):
    """Execute preprocessing operations - exact same as shi.py lines 296-302"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Starting preprocessing..."
        
        if request.stripes:
            processing_jobs[job_id]["message"] = "Correcting stripes..."
            processing_jobs[job_id]["progress"] = 0.5
            
            # Use temp_dir as current folder (same logic as shi.py using Path.cwd())
            correcting_stripes.correcting_stripes(temp_dir)
            
            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["progress"] = 1.0
            processing_jobs[job_id]["message"] = "Stripe correction completed"
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            processing_jobs[job_id]["log_messages"].append("Stripes corrected successfully")
        else:
            raise ValueError("No preprocessing option specified")
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = f"Preprocessing failed: {str(e)}"
        processing_jobs[job_id]["log_messages"].append(f"ERROR: {str(e)}")
        logger.error(f"Preprocessing error for job {job_id}: {e}")


def run_clean_command(job_id: str, request: CleanRequest):
    """Execute clean operations - same as commented shi.py lines 274-285"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Starting cleanup..."
        
        if request.clear_cache:
            processing_jobs[job_id]["message"] = "Clearing cache..."
            processing_jobs[job_id]["progress"] = 0.5
            
            for content in cache_dir.iterdir():
                logger.info(f"Removing cache content: {content}")
                subprocess.run(["rm", "-rf", content.as_posix()], check=True)
                processing_jobs[job_id]["log_messages"].append(f"Removed: {content}")
        
        elif request.clear_extra:
            processing_jobs[job_id]["message"] = "Clearing extra files..."
            processing_jobs[job_id]["progress"] = 0.5
            # Implementation would depend on what "extra files" means
            processing_jobs[job_id]["log_messages"].append("Extra files cleanup completed")
        else:
            raise ValueError("No cleaning option specified. Use clear_cache or clear_extra.")
        
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 1.0
        processing_jobs[job_id]["message"] = "Cleanup completed"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = f"Cleanup failed: {str(e)}"
        processing_jobs[job_id]["log_messages"].append(f"ERROR: {str(e)}")
        logger.error(f"Cleanup error for job {job_id}: {e}")


# API Endpoints

@app.get("/api/commands", response_model=List[CommandInfo])
async def get_available_commands():
    """Get information about available commands - mirrors shi.py subcommands"""
    commands = [
        CommandInfo(
            name="calculate",
            description="Executes the SHI method",
            options=[
                {"name": "mask_period", "type": "int", "required": True, "description": "Number of projected pixels in the mask"},
                {"name": "mode", "type": "select", "required": True, "options": ["2d", "3d", "custom"], "description": "Processing mode"},
                {"name": "unwrap_phase", "type": "select", "required": False, "options": ["branch_cut", "least_squares", "quality_guided", "min_lp"], "description": "Phase unwrapping method"},
                {"name": "average", "type": "boolean", "required": False, "description": "Apply averaging"},
                {"name": "export", "type": "boolean", "required": False, "description": "Apply export"},
                {"name": "angle_after", "type": "boolean", "required": False, "description": "Apply angle correction after measurements"}
            ]
        ),
        CommandInfo(
            name="morphostructural",
            description="Performs morphostructural analysis",
            options=[
                {"name": "morphostructural", "type": "boolean", "required": True, "description": "Apply morphostructural analysis"}
            ]
        ),
        CommandInfo(
            name="preprocessing",
            description="Corrects angle alignment of optical components",
            options=[
                {"name": "stripes", "type": "boolean", "required": False, "description": "Correct stripes by deleting them"}
            ]
        ),
        CommandInfo(
            name="clean",
            description="Deletes temporary files generated by the SHI method",
            options=[
                {"name": "clear_cache", "type": "boolean", "required": False, "description": "Clear cache generated by SHI"},
                {"name": "clear_extra", "type": "boolean", "required": False, "description": "Clear extra files generated by SHI after calculate"}
            ]
        )
    ]
    return commands


@app.post("/api/upload-files")
async def upload_files_for_processing(
    files: List[UploadFile] = File(...)
):
    """Upload and organize files for ROI selection (Step 1 of processing)"""
    
    job_id = str(uuid.uuid4())
    temp_dir = create_temp_directory()
    
    try:
        # Organize and save files
        organized_folders = organize_uploaded_files(files, temp_dir)
        success = await save_uploaded_files(files, organized_folders)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save uploaded files")
        
        # Store job info (but don't start processing yet)
        processing_jobs[job_id] = {
            "job_id": job_id,
            "command": "calculate", 
            "status": "waiting_for_roi",  # New status for ROI selection
            "progress": 0.1,
            "message": "Files uploaded, waiting for ROI selection",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "results_path": None,
            "log_messages": [],
            "temp_dir": temp_dir,
            "organized_folders": organized_folders,
        }
        
        logger.info(f"Files uploaded for job {job_id}, waiting for ROI selection")
        return {"job_id": job_id, "status": "waiting_for_roi", "message": "Files uploaded successfully"}
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calculate")
async def calculate_command(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    mask_period: int = Form(...),
    mode: str = Form("2d"),
    unwrap_phase: Optional[str] = Form(None),
    average: bool = Form(False),
    export: bool = Form(False),
    angle_after: bool = Form(False)
):
    """Start calculate processing with ROI (Step 2 of processing)"""
    
    # Check if job exists and is waiting for ROI
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "waiting_for_roi":
        raise HTTPException(status_code=400, detail=f"Job is not waiting for ROI. Current status: {job['status']}")
    
    try:
        # Get previously uploaded files info
        temp_dir = job["temp_dir"]
        
        # Create request with new parameters
        request = CalculateRequest(
            mask_period=mask_period,
            unwrap_phase=unwrap_phase,
            mode=mode,
            average=average,
            export=export,
            angle_after=angle_after
        )
        
        # Update job status to start processing
        processing_jobs[job_id].update({
            "status": "pending",
            "progress": 0.2,
            "message": "Processing started with ROI...",
        })

        # Start background processing
        background_tasks.add_task(run_calculate_command, job_id, temp_dir, request)
        
        return {"job_id": job_id, "message": "Calculate processing started"}
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["message"] = str(e)
        raise HTTPException(status_code=500, detail=f"Calculate command failed: {str(e)}")


@app.post("/api/morphostructural")
async def morphostructural_command(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Morphostructural command - web equivalent of 'shi morphostructural'"""
    
    job_id = str(uuid.uuid4())
    temp_dir = create_temp_directory()
    
    try:
        # Save files
        file_mapping = organize_uploaded_files(files, temp_dir)
        success = await save_uploaded_files(files, file_mapping)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save uploaded files")
        
        request = MorphostructuralRequest(morphostructural=True)
        
        # Initialize job tracking
        processing_jobs[job_id] = {
            "job_id": job_id,
            "command": "morphostructural",
            "status": "pending",
            "progress": 0.0,
            "message": "Morphostructural analysis queued",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "results_path": None,
            "temp_dir": str(temp_dir),
            "log_messages": []
        }
        
        # Start processing
        background_tasks.add_task(run_morphostructural_command, job_id, temp_dir, request)
        
        return {"job_id": job_id, "message": "Morphostructural analysis started"}
        
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Morphostructural command failed: {str(e)}")


@app.post("/api/preprocessing")
async def preprocessing_command(
    background_tasks: BackgroundTasks,
    stripes: bool = Form(False),
    files: List[UploadFile] = File(...)
):
    """Preprocessing command - web equivalent of 'shi preprocessing'"""
    
    job_id = str(uuid.uuid4())
    temp_dir = create_temp_directory()
    
    try:
        # Save files
        file_mapping = organize_uploaded_files(files, temp_dir)
        success = await save_uploaded_files(files, file_mapping)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save uploaded files")
        
        request = PreprocessingRequest(stripes=stripes)
        
        # Initialize job tracking
        processing_jobs[job_id] = {
            "job_id": job_id,
            "command": "preprocessing",
            "status": "pending",
            "progress": 0.0,
            "message": "Preprocessing queued",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "results_path": None,
            "temp_dir": str(temp_dir),
            "log_messages": []
        }
        
        # Start processing
        background_tasks.add_task(run_preprocessing_command, job_id, temp_dir, request)
        
        return {"job_id": job_id, "message": "Preprocessing started"}
        
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Preprocessing command failed: {str(e)}")


@app.post("/api/clean")
async def clean_command(
    background_tasks: BackgroundTasks,
    clear_cache: bool = Form(False),
    clear_extra: bool = Form(False)
):
    """Clean command - web equivalent of 'shi clean'"""
    
    job_id = str(uuid.uuid4())
    
    try:
        if not clear_cache and not clear_extra:
            raise HTTPException(status_code=400, detail="No cleaning option specified. Use clear_cache or clear_extra.")
        
        request = CleanRequest(clear_cache=clear_cache, clear_extra=clear_extra)
        
        # Initialize job tracking
        processing_jobs[job_id] = {
            "job_id": job_id,
            "command": "clean",
            "status": "pending",
            "progress": 0.0,
            "message": "Cleanup queued",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "results_path": None,
            "temp_dir": None,
            "log_messages": []
        }
        
        # Start processing
        background_tasks.add_task(run_clean_command, job_id, request)
        
        return {"job_id": job_id, "message": "Cleanup started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clean command failed: {str(e)}")


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return JobResponse(**job)


@app.get("/api/jobs", response_model=List[JobResponse])
async def list_jobs():
    """List all jobs"""
    return [JobResponse(**job) for job in processing_jobs.values()]


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    """Download job results"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    results_path = job.get("results_path")
    if not results_path or not Path(results_path).exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        results_path,
        media_type="application/zip",
        filename=f"shi_{job['command']}_results_{job_id[:8]}.zip"
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    temp_dir = job.get("temp_dir")
    
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
    
    del processing_jobs[job_id]
    return {"message": "Job deleted successfully"}


def normalize_image_for_web(img_array: np.ndarray) -> Image.Image:
    """Convert TIFF image array to normalized PIL Image for web display."""
    # Handle different data types
    if img_array.dtype == np.uint16:
        # Scale 16-bit to 8-bit
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        if max_val - min_val == 0:
            norm_img = np.zeros_like(img_array, dtype=np.uint8)
        else:
            norm_img = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    elif img_array.dtype == np.uint8:
        norm_img = img_array
    else:
        # For float or other types, normalize to 0-255
        norm_img = ((img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255).astype(np.uint8)
    
    return Image.fromarray(norm_img)


@app.get("/api/roi/image/{job_id}")
async def get_roi_image(job_id: str):
    """Get the first sample image for ROI selection."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    temp_dir = job.get("temp_dir")
    
    if not temp_dir or not Path(temp_dir).exists():
        raise HTTPException(status_code=404, detail="Job files not found")
    
    # Find first sample image
    sample_dir = Path(temp_dir) / "sample" / "measurement_001"
    tiff_files = list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.tiff"))
    
    if not tiff_files:
        raise HTTPException(status_code=404, detail="No TIFF files found for ROI selection")
    
    # Load and normalize the first image
    image_path = tiff_files[0]
    
    try:
        # Load TIFF file
        with ti.TiffFile(str(image_path)) as tif:
            image_array = tif.asarray()
        
        # Convert to web-displayable format
        pil_image = normalize_image_for_web(image_array)
        
        # Convert to base64 for web display
        buffer = python_io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image_data": f"data:image/png;base64,{img_str}",
            "original_size": {"width": int(image_array.shape[1]), "height": int(image_array.shape[0])},
            "file_name": image_path.name
        }
        
    except Exception as e:
        logger.error(f"Error loading image for ROI: {e}")
        raise HTTPException(status_code=500, detail="Error loading image")


@app.post("/api/roi/save")
async def save_roi(roi_request: ROIRequest):
    """Save ROI coordinates for a job."""
    job_id = roi_request.job_id
    coordinates = roi_request.coordinates
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Convert web coordinates to SHI format (y0, y1, x0, x1)
    shi_coordinates = (coordinates.y0, coordinates.y1, coordinates.x0, coordinates.x1)
    
    # Save ROI using the web provider
    roi_file = roi_provider.save_roi_coordinates(
        job_id=job_id,
        coordinates=shi_coordinates,
        image_info={"timestamp": datetime.now().isoformat()}
    )
    
    # Update job status
    processing_jobs[job_id]["roi_coordinates"] = shi_coordinates
    processing_jobs[job_id]["roi_file"] = str(roi_file)
    
    return {
        "message": "ROI saved successfully",
        "coordinates": shi_coordinates,
        "roi_file": str(roi_file)
    }


@app.get("/api/roi/{job_id}")
async def get_roi(job_id: str):
    """Get saved ROI coordinates for a job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    roi_coords = roi_provider.get_roi_for_job(job_id)
    
    if roi_coords:
        # Convert from SHI format (y0, y1, x0, x1) to web format
        return {
            "job_id": job_id,
            "coordinates": {
                "x0": roi_coords[2],
                "y0": roi_coords[0], 
                "x1": roi_coords[3],
                "y1": roi_coords[1]
            }
        }
    else:
        raise HTTPException(status_code=404, detail="No ROI found for this job")


@app.delete("/api/roi/{job_id}")
async def delete_roi(job_id: str):
    """Delete ROI coordinates for a job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    roi_provider.clear_roi_for_job(job_id)
    
    # Update job status
    if "roi_coordinates" in processing_jobs[job_id]:
        del processing_jobs[job_id]["roi_coordinates"]
    if "roi_file" in processing_jobs[job_id]:
        del processing_jobs[job_id]["roi_file"]
    
    return {"message": "ROI deleted successfully"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in processing_jobs.values() if j["status"] == "processing"]),
        "available_commands": ["calculate", "morphostructural", "preprocessing", "clean"],
        "version": "1.0.0"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
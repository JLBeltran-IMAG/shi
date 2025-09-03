#!/usr/bin/env python3
"""
Morphostructural Analysis Tool
Main entry point for the morphostructural analysis functionality.
"""
import sys
from pathlib import Path
import os

# Configure OpenGL before importing Qt
os.environ.update({
    "QT_QUICK_BACKEND": "software",
    "QT_QPA_PLATFORM": "xcb",
    "QT_XCB_GL_INTEGRATION": "none"
})

# Add the src directory to Python path
src_dir = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_dir))

# Import the main function from morphostructural module
from src.post_shi.morphostructural import main

if __name__ == "__main__":
    sys.exit(main())

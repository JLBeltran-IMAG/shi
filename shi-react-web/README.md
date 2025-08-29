# SHI React Web Application

A complete React frontend with FastAPI backend that provides web access to all functionality of the `shi.py` CLI tool for Spatial Harmonic Imaging processing.

## Overview

This application creates a modern web interface that mirrors all commands and functionality of the original `shi.py` CLI tool:

- **calculate** - Executes the SHI method with all original options
- **morphostructural** - Performs morphostructural analysis  
- **preprocessing** - Corrects angle alignment and stripes
- **clean** - Removes temporary files and cache

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │────│   FastAPI Backend │────│ SHI Processing  │
│   (TypeScript)   │    │   (Python)       │    │   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  File Upload &    │
                        │  Job Management   │
                        └───────────────────┘
```

## Features

### ✅ Complete CLI Equivalence
- **All shi.py commands** implemented as web endpoints
- **Exact same logic** - uses the same source code and processing modules
- **All command options** available through web forms
- **Same file organization** - automatic sorting by filename patterns

### 🎯 User Interface
- **Modern React components** with TypeScript
- **Drag-and-drop file upload** with file categorization
- **Real-time progress monitoring** for all operations
- **Job management** - view, monitor, download, and delete jobs
- **Responsive design** - works on desktop and mobile

### 🔧 Backend API
- **FastAPI framework** for high-performance async processing
- **Complete REST API** for all shi.py functionality
- **Background job processing** with status tracking
- **File upload handling** with automatic organization
- **Results download** with ZIP archive creation

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.9+
- All SHI dependencies (numpy, scipy, scikit-image, etc.)

### 1. Start Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Backend runs on: http://localhost:8000

### 2. Start Frontend
```bash
cd frontend
npm install
npm start
```
Frontend runs on: http://localhost:3000

### 3. Use the Application
1. Open http://localhost:3000 in your browser
2. Select a command (calculate, morphostructural, preprocessing, clean)
3. Upload your TIFF files
4. Configure command options
5. Submit and monitor progress
6. Download results when complete

## Command Equivalence

### Calculate Command
**CLI:** `shi calculate -m 12 --all-2d --average --export`

**Web:** Select "calculate" → Set mask period to 12 → Choose "2D Processing" → Enable averaging and export → Upload files → Submit

### Morphostructural Analysis
**CLI:** `shi morphostructural --morphostructural`

**Web:** Select "morphostructural" → Upload analysis files → Submit

### Preprocessing
**CLI:** `shi preprocessing --stripes`

**Web:** Select "preprocessing" → Enable "Correct Stripes" → Upload raw data → Submit

### Clean
**CLI:** `shi clean --clear-cache`

**Web:** Select "clean" → Enable "Clear Cache" → Submit

## API Endpoints

The backend provides a complete REST API:

- `GET /api/commands` - Get available commands and options
- `POST /api/calculate` - Run calculate command
- `POST /api/morphostructural` - Run morphostructural analysis  
- `POST /api/preprocessing` - Run preprocessing operations
- `POST /api/clean` - Run cleanup operations
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/download/{job_id}` - Download job results
- `DELETE /api/jobs/{job_id}` - Delete job

## File Organization

Files are automatically organized based on filename patterns (same as CLI):

- **Sample images**: Default for unlabeled files
- **Dark images**: Contains "dark" in filename
- **Flat images**: Contains "flat" in filename  
- **Bright images**: Contains "bright" in filename

## Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python main.py --reload
```

### Frontend Development  
```bash
cd frontend
npm install
npm start
```

### Building for Production
```bash
cd frontend
npm run build
```

## Technology Stack

### Frontend
- **React 18** with TypeScript
- **Modern hooks** (useState, useEffect)
- **React Dropzone** for file upload
- **Axios** for API communication
- **Responsive CSS** with modern styling

### Backend
- **FastAPI** for high-performance API
- **Python multipart** for file uploads
- **Background tasks** for async processing
- **Job tracking** with in-memory storage
- **Direct integration** with existing SHI modules

## File Structure

```
shi-react-web/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── CommandSelector.tsx
│   │   │   ├── CalculateForm.tsx
│   │   │   ├── MorphostructuralForm.tsx
│   │   │   ├── PreprocessingForm.tsx
│   │   │   ├── CleanForm.tsx
│   │   │   └── JobMonitor.tsx
│   │   ├── services/
│   │   │   └── api.ts       # API service layer
│   │   ├── types.ts         # TypeScript types
│   │   ├── App.tsx          # Main application
│   │   └── App.css          # Application styles
│   ├── package.json         # Node.js dependencies
│   └── public/              # Static assets
└── README.md               # This file
```

## Benefits Over CLI

1. **User-Friendly Interface** - No command line knowledge required
2. **Visual File Management** - See uploaded files categorized automatically
3. **Progress Monitoring** - Real-time updates on processing status
4. **Job History** - Keep track of all processing operations
5. **Easy Result Access** - Download results with one click
6. **Error Handling** - Clear error messages and help text
7. **Cross-Platform** - Works on any device with a web browser

## Deployment

### Local Development
Both frontend and backend run locally for development and testing.

### Production Deployment
- Build React app: `npm run build`
- Serve static files through FastAPI
- Deploy to any Python hosting service
- Configure CORS for your domain

## Support

For issues or questions:
1. Check the original SHI documentation for processing-related questions
2. Review API documentation at http://localhost:8000/docs when backend is running
3. Check browser console for frontend errors
4. Review backend logs for processing errors

## License

This web application extends the SHI (Spatial Harmonic Imaging) software and follows the same licensing terms as the main project.
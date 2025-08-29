import React, { useState, useRef, useCallback, useEffect } from 'react';
import './ROISelector.css';

interface ROICoordinates {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface ROISelectorProps {
  imageUrl: string;
  onROISelect: (roi: ROICoordinates) => void;
  onCancel: () => void;
  initialROI?: ROICoordinates;
}

const ROISelector: React.FC<ROISelectorProps> = ({ 
  imageUrl, 
  onROISelect, 
  onCancel, 
  initialROI 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentROI, setCurrentROI] = useState<ROICoordinates | null>(initialROI || null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  
  // Image scaling factors
  const [scale, setScale] = useState({ x: 1, y: 1 });
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });

  const drawImage = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    const ctx = canvas?.getContext('2d');
    
    if (!canvas || !image || !ctx || !imageLoaded) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply brightness and contrast filters
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
    
    // Draw image
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    
    // Reset filter for ROI drawing
    ctx.filter = 'none';
    
    // Draw current ROI if exists
    if (currentROI) {
      drawROI(ctx, currentROI);
    }
  }, [imageLoaded, brightness, contrast, currentROI]);

  const drawROI = (ctx: CanvasRenderingContext2D, roi: ROICoordinates) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Convert image coordinates to canvas coordinates
    const canvasX0 = roi.x0 * scale.x;
    const canvasY0 = roi.y0 * scale.y;
    const canvasX1 = roi.x1 * scale.x;
    const canvasY1 = roi.y1 * scale.y;
    
    // Draw ROI rectangle
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(canvasX0, canvasY0, canvasX1 - canvasX0, canvasY1 - canvasY0);
    
    // Draw corner handles
    const handleSize = 8;
    ctx.fillStyle = '#ff0000';
    ctx.fillRect(canvasX0 - handleSize/2, canvasY0 - handleSize/2, handleSize, handleSize);
    ctx.fillRect(canvasX1 - handleSize/2, canvasY0 - handleSize/2, handleSize, handleSize);
    ctx.fillRect(canvasX0 - handleSize/2, canvasY1 - handleSize/2, handleSize, handleSize);
    ctx.fillRect(canvasX1 - handleSize/2, canvasY1 - handleSize/2, handleSize, handleSize);
    
    // Add ROI info text
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(canvasX0, canvasY0 - 25, 150, 20);
    ctx.fillStyle = '#000000';
    ctx.font = '12px Arial';
    ctx.fillText(`ROI: (${roi.x0},${roi.y0}) to (${roi.x1},${roi.y1})`, canvasX0 + 5, canvasY0 - 10);
  };

  useEffect(() => {
    drawImage();
  }, [drawImage]);

  const handleImageLoad = () => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    
    if (!image || !canvas) return;
    
    // Set original image dimensions
    setImageSize({ width: image.naturalWidth, height: image.naturalHeight });
    
    // Calculate display size (max 800px while maintaining aspect ratio)
    const maxSize = 800;
    const aspectRatio = image.naturalWidth / image.naturalHeight;
    
    let displayWidth, displayHeight;
    if (aspectRatio > 1) {
      displayWidth = Math.min(maxSize, image.naturalWidth);
      displayHeight = displayWidth / aspectRatio;
    } else {
      displayHeight = Math.min(maxSize, image.naturalHeight);
      displayWidth = displayHeight * aspectRatio;
    }
    
    setDisplaySize({ width: displayWidth, height: displayHeight });
    
    // Set canvas size
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    // Calculate scaling factors
    setScale({
      x: displayWidth / image.naturalWidth,
      y: displayHeight / image.naturalHeight
    });
    
    setImageLoaded(true);
  };

  const getMousePosition = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const canvasToImageCoordinates = (canvasX: number, canvasY: number) => {
    return {
      x: Math.round(canvasX / scale.x),
      y: Math.round(canvasY / scale.y)
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePosition(e);
    setStartPoint(pos);
    setIsDrawing(true);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint) return;
    
    const pos = getMousePosition(e);
    
    // Convert to image coordinates
    const start = canvasToImageCoordinates(startPoint.x, startPoint.y);
    const end = canvasToImageCoordinates(pos.x, pos.y);
    
    // Ensure proper ordering (top-left to bottom-right)
    const roi: ROICoordinates = {
      x0: Math.min(start.x, end.x),
      y0: Math.min(start.y, end.y),
      x1: Math.max(start.x, end.x),
      y1: Math.max(start.y, end.y)
    };
    
    setCurrentROI(roi);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    setStartPoint(null);
  };

  const handleConfirmROI = () => {
    if (currentROI) {
      onROISelect(currentROI);
    }
  };

  const handleResetROI = () => {
    setCurrentROI(null);
  };

  const handleUseFullImage = () => {
    const fullImageROI: ROICoordinates = {
      x0: 0,
      y0: 0,
      x1: imageSize.width,
      y1: imageSize.height
    };
    setCurrentROI(fullImageROI);
  };

  return (
    <div className="roi-selector-overlay">
      <div className="roi-selector-modal">
        <div className="roi-header">
          <h3>Select Region of Interest (ROI)</h3>
          <p>Click and drag on the image to select a region for processing</p>
        </div>
        
        <div className="roi-content">
          <div className="roi-image-container">
            {!imageLoaded && <div className="roi-loading">Loading image...</div>}
            
            <img
              ref={imageRef}
              src={imageUrl}
              alt="ROI Selection"
              onLoad={handleImageLoad}
              style={{ display: 'none' }}
            />
            
            <canvas
              ref={canvasRef}
              className="roi-canvas"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              style={{ 
                display: imageLoaded ? 'block' : 'none',
                cursor: isDrawing ? 'crosshair' : 'pointer'
              }}
            />
          </div>
          
          <div className="roi-controls">
            <div className="brightness-contrast-controls">
              <div className="control-group">
                <label htmlFor="brightness">Brightness: {brightness}%</label>
                <input
                  id="brightness"
                  type="range"
                  min="10"
                  max="300"
                  value={brightness}
                  onChange={(e) => setBrightness(parseInt(e.target.value))}
                />
              </div>
              
              <div className="control-group">
                <label htmlFor="contrast">Contrast: {contrast}%</label>
                <input
                  id="contrast"
                  type="range"
                  min="10"
                  max="300"
                  value={contrast}
                  onChange={(e) => setContrast(parseInt(e.target.value))}
                />
              </div>
            </div>
            
            <div className="roi-info">
              {currentROI ? (
                <div className="roi-coordinates">
                  <strong>Selected ROI:</strong><br />
                  Top-left: ({currentROI.x0}, {currentROI.y0})<br />
                  Bottom-right: ({currentROI.x1}, {currentROI.y1})<br />
                  Size: {currentROI.x1 - currentROI.x0} × {currentROI.y1 - currentROI.y0} pixels
                </div>
              ) : (
                <div className="no-roi">No ROI selected</div>
              )}
            </div>
          </div>
        </div>
        
        <div className="roi-actions">
          <button 
            type="button" 
            onClick={handleUseFullImage}
            className="roi-btn secondary"
          >
            Use Full Image
          </button>
          
          <button 
            type="button" 
            onClick={handleResetROI}
            className="roi-btn secondary"
            disabled={!currentROI}
          >
            Clear ROI
          </button>
          
          <button 
            type="button" 
            onClick={onCancel}
            className="roi-btn cancel"
          >
            Cancel
          </button>
          
          <button 
            type="button" 
            onClick={handleConfirmROI}
            className="roi-btn primary"
            disabled={!currentROI}
          >
            Confirm ROI
          </button>
        </div>
      </div>
    </div>
  );
};

export default ROISelector;
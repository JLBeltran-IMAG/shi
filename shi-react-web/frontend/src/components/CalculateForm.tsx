import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Job } from '../types';
import * as api from '../services/api';
import ROISelector from './ROISelector';
import './CalculateForm.css';

interface CalculateFormProps {
  onJobCreated: (job: Job) => void;
}

const CalculateForm: React.FC<CalculateFormProps> = ({ onJobCreated }) => {
  const [formData, setFormData] = useState({
    mask_period: '',
    mode: '2d',
    unwrap_phase: '',
    average: false,
    export: false,
    angle_after: false,
  });
  
  const [wizardStep, setWizardStep] = useState(0);
  const [fileCategories, setFileCategories] = useState({
    sample: [] as File[],
    dark: [] as File[],
    flat: [] as File[],
    bright: [] as File[]
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string>('');
  
  // ROI selection state
  const [showROISelector, setShowROISelector] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string>('');
  const [roiImage, setROIImage] = useState<string>('');
  const [roiCoordinates, setROICoordinates] = useState<api.ROICoordinates | null>(null);

  const wizardSteps = [
    {
      title: "Sample Images",
      category: "sample" as keyof typeof fileCategories,
      description: "Upload your main experimental TIFF images",
      required: true,
      help: "These are the primary images containing your experimental data that will be processed by the SHI method."
    },
    {
      title: "Dark Images",
      category: "dark" as keyof typeof fileCategories,
      description: "Upload dark field calibration images",
      required: false,
      help: "Dark field images are captured with the X-ray source off to measure detector noise and offset."
    },
    {
      title: "Flat Images",
      category: "flat" as keyof typeof fileCategories,
      description: "Upload flat field calibration images",
      required: false,
      help: "Flat field images are captured without a sample to measure beam intensity variations."
    },
    {
      title: "Bright Images",
      category: "bright" as keyof typeof fileCategories,
      description: "Upload bright field calibration images (optional)",
      required: false,
      help: "Bright field images provide additional calibration data for enhanced correction."
    }
  ];

  const currentStep = wizardSteps[wizardStep];

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/tiff': ['.tif', '.tiff'],
    },
    multiple: true,
    onDrop: (acceptedFiles) => {
      setFileCategories(prev => ({
        ...prev,
        [currentStep.category]: [...prev[currentStep.category], ...acceptedFiles]
      }));
      setError('');
    },
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : value,
    }));
  };

  const removeFile = (category: keyof typeof fileCategories, index: number) => {
    setFileCategories(prev => ({
      ...prev,
      [category]: prev[category].filter((_, i) => i !== index)
    }));
  };

  const nextStep = () => {
    if (wizardStep < wizardSteps.length - 1) {
      setWizardStep(wizardStep + 1);
    }
  };

  const prevStep = () => {
    if (wizardStep > 0) {
      setWizardStep(wizardStep - 1);
    }
  };

  const canProceed = () => {
    return !currentStep.required || fileCategories[currentStep.category].length > 0;
  };

  const getAllFiles = () => {
    return [
      ...fileCategories.sample,
      ...fileCategories.dark,
      ...fileCategories.flat,
      ...fileCategories.bright
    ];
  };


  const startROISelection = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.mask_period) {
      setError('Mask period is required');
      return;
    }
    
    const allFiles = getAllFiles();
    if (allFiles.length === 0) {
      setError('Please upload at least one TIFF file');
      return;
    }

    if (fileCategories.sample.length === 0) {
      setError('Sample images are required');
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      // Step 1: Upload files only (no processing yet)
      const uploadResponse = await api.uploadFilesForProcessing(allFiles);
      setCurrentJobId(uploadResponse.job_id);

      // Step 2: Get image for ROI selection
      try {
        const roiImageData = await api.getROIImage(uploadResponse.job_id);
        setROIImage(roiImageData.image_data);
        setShowROISelector(true);
      } catch (roiError) {
        console.error('Failed to get ROI image:', roiError);
        setError('Failed to load image for ROI selection. Processing will continue with full image.');
        // Continue without ROI selection
        completeJobSubmission(uploadResponse.job_id);
      }

    } catch (error: any) {
      setError(error.response?.data?.detail || 'Failed to start calculate command');
      setIsSubmitting(false);
    }
  };

  const completeJobSubmission = (jobId: string) => {
    // Create a job object to add to the list
    const newJob: Job = {
      job_id: jobId,
      command: 'calculate',
      status: 'pending',
      progress: 0,
      message: 'Calculate command processing...',
      created_at: new Date().toISOString(),
      log_messages: [],
    };

    onJobCreated(newJob);

    // Reset form
    setFormData({
      mask_period: '',
      mode: '2d',
      unwrap_phase: '',
      average: false,
      export: false,
      angle_after: false,
    });
    setFileCategories({
      sample: [],
      dark: [],
      flat: [],
      bright: []
    });
    setWizardStep(0);
    setIsSubmitting(false);
  };

  const handleROISelect = async (coordinates: api.ROICoordinates) => {
    try {
      // Save ROI coordinates
      await api.saveROI(currentJobId, coordinates);
      setROICoordinates(coordinates);
      setShowROISelector(false);
      
      // Start processing with ROI
      await startProcessingWithROI(currentJobId);
      
    } catch (error) {
      console.error('Failed to save ROI:', error);
      setError('Failed to save ROI. Processing will continue with full image.');
      setShowROISelector(false);
      // Start processing anyway with default ROI
      await startProcessingWithROI(currentJobId);
    }
  };

  const startProcessingWithROI = async (jobId: string) => {
    try {
      // Step 3: Start processing with saved ROI
      await api.runCalculateCommand({
        job_id: jobId,
        mask_period: parseInt(formData.mask_period),
        mode: formData.mode,
        unwrap_phase: formData.unwrap_phase || undefined,
        average: formData.average,
        export: formData.export,
        angle_after: formData.angle_after,
      });
      
      // Complete job submission
      completeJobSubmission(jobId);
      
    } catch (error) {
      console.error('Failed to start processing:', error);
      setError('Failed to start processing. Please try again.');
      setIsSubmitting(false);
    }
  };

  const handleROICancel = async () => {
    setShowROISelector(false);
    setIsSubmitting(false);
    
    // Clean up the job that was waiting for ROI
    if (currentJobId) {
      try {
        await api.deleteJob(currentJobId);
      } catch (error) {
        console.error('Failed to clean up cancelled job:', error);
      }
      setCurrentJobId('');
    }
  };


  return (
    <div className="calculate-form">
      <h3>Calculate Command</h3>
      <p className="form-description">
        Executes the SHI method. Equivalent to <code>shi calculate</code>
      </p>

      <form onSubmit={startROISelection}>
        {/* Mask Period - Required */}
        <div className="form-group">
          <label htmlFor="mask_period">
            Mask Period <span className="required">*</span>
          </label>
          <input
            type="number"
            id="mask_period"
            name="mask_period"
            value={formData.mask_period}
            onChange={handleInputChange}
            placeholder="e.g., 12"
            min="1"
            step="1"
            required
          />
          <div className="field-help">
            Number of projected pixels in the mask (equivalent to <code>-m</code> flag)
          </div>
        </div>

        {/* Processing Mode */}
        <div className="form-group">
          <label htmlFor="mode">Processing Mode</label>
          <select
            id="mode"
            name="mode"
            value={formData.mode}
            onChange={handleInputChange}
          >
            <option value="2d">2D Processing (--all-2d)</option>
            <option value="3d">3D/CT Processing (--all-3d)</option>
            <option value="custom">Custom Paths</option>
          </select>
          <div className="field-help">
            Processing mode: 2D for standard images, 3D for CT reconstruction
          </div>
        </div>

        {/* Phase Unwrapping Method */}
        <div className="form-group">
          <label htmlFor="unwrap_phase">Phase Unwrapping Method</label>
          <select
            id="unwrap_phase"
            name="unwrap_phase"
            value={formData.unwrap_phase}
            onChange={handleInputChange}
          >
            <option value="">Default (reliability-based)</option>
            <option value="branch_cut">Branch Cut (Goldstein's)</option>
            <option value="least_squares">Least Squares (FFT-based)</option>
            <option value="quality_guided">Quality Guided</option>
            <option value="min_lp">Minimum Lp-Norm</option>
          </select>
          <div className="field-help">
            Phase unwrapping algorithm (equivalent to <code>--unwrap-phase</code> flag)
          </div>
        </div>

        {/* Additional Options */}
        <div className="form-group">
          <div className="checkbox-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                name="average"
                checked={formData.average}
                onChange={handleInputChange}
              />
              Apply Averaging (--average)
            </label>
            
            <label className="checkbox-label">
              <input
                type="checkbox"
                name="export"
                checked={formData.export}
                onChange={handleInputChange}
              />
              Apply Export (--export)
            </label>
            
            <label className="checkbox-label">
              <input
                type="checkbox"
                name="angle_after"
                checked={formData.angle_after}
                onChange={handleInputChange}
              />
              Angle Correction After (--angle-after)
            </label>
          </div>
        </div>

        {/* File Upload Wizard */}
        <div className="form-group">
          <div className="wizard-container">
            <div className="wizard-header">
              <h4>Upload Files - Step {wizardStep + 1} of {wizardSteps.length}</h4>
              <div className="wizard-progress">
                {wizardSteps.map((_, index) => (
                  <div 
                    key={index} 
                    className={`progress-step ${
                      index <= wizardStep ? 'completed' : 'pending'
                    }`}
                  />
                ))}
              </div>
            </div>
            
            <div className="wizard-step">
              <h5>{currentStep.title} {currentStep.required && <span className="required">*</span>}</h5>
              <p className="step-description">{currentStep.description}</p>
              
              <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'active' : ''}`}
              >
                <input {...getInputProps()} />
                {isDragActive ? (
                  <p>Drop the TIFF files here...</p>
                ) : (
                  <p>
                    Drag & drop {currentStep.title.toLowerCase()} here, or click to select
                    <br />
                    <small>Accepts .tif and .tiff files</small>
                  </p>
                )}
              </div>
              
              <div className="field-help">{currentStep.help}</div>
              
              {/* Current step files */}
              {fileCategories[currentStep.category].length > 0 && (
                <div className="step-files">
                  <h6>Uploaded {currentStep.title} ({fileCategories[currentStep.category].length} files)</h6>
                  <div className="file-list">
                    {fileCategories[currentStep.category].map((file, index) => (
                      <div key={index} className="file-item">
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">
                          ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                        </span>
                        <button
                          type="button"
                          onClick={() => removeFile(currentStep.category, index)}
                          className="remove-file"
                          title="Remove file"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="wizard-navigation">
                <button
                  type="button"
                  onClick={prevStep}
                  disabled={wizardStep === 0}
                  className="wizard-btn prev"
                >
                  Previous
                </button>
                
                {wizardStep < wizardSteps.length - 1 ? (
                  <button
                    type="button"
                    onClick={nextStep}
                    disabled={!canProceed()}
                    className="wizard-btn next"
                  >
                    Next Step
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setWizardStep(0)}
                    className="wizard-btn review"
                  >
                    Review All Files
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* File Summary */}
        {getAllFiles().length > 0 && (
          <div className="file-summary">
            <h4>All Uploaded Files ({getAllFiles().length} total)</h4>
            
            {Object.entries(fileCategories).map(([category, categoryFiles]) => {
              if (categoryFiles.length === 0) return null;
              
              return (
                <div key={category} className="file-category">
                  <h5 className={`category-${category}`}>
                    {category.toUpperCase()} ({categoryFiles.length} files)
                  </h5>
                  <div className="file-list">
                    {categoryFiles.map((file, index) => (
                      <div key={index} className="file-item">
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">
                          ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                        </span>
                        <button
                          type="button"
                          onClick={() => removeFile(category as keyof typeof fileCategories, index)}
                          className="remove-file"
                          title="Remove file"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        <button
          type="submit"
          disabled={isSubmitting || !formData.mask_period || getAllFiles().length === 0 || fileCategories.sample.length === 0}
          className="submit-button"
        >
          {isSubmitting ? 'Preparing ROI Selection...' : 'Select ROI & Run Calculate'}
        </button>
      </form>
      
      {/* ROI Selector Modal */}
      {showROISelector && roiImage && (
        <ROISelector
          imageUrl={roiImage}
          onROISelect={handleROISelect}
          onCancel={handleROICancel}
          initialROI={roiCoordinates || undefined}
        />
      )}
    </div>
  );
};

export default CalculateForm;
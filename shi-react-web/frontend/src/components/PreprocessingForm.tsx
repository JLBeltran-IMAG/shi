import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Job } from '../types';
import * as api from '../services/api';
import './PreprocessingForm.css';

interface PreprocessingFormProps {
  onJobCreated: (job: Job) => void;
}

const PreprocessingForm: React.FC<PreprocessingFormProps> = ({ onJobCreated }) => {
  const [formData, setFormData] = useState({
    stripes: false,
  });
  
  const [files, setFiles] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string>('');

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/tiff': ['.tif', '.tiff'],
    },
    multiple: true,
    onDrop: (acceptedFiles) => {
      setFiles(prev => [...prev, ...acceptedFiles]);
      setError('');
    },
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: checked,
    }));
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.stripes) {
      setError('Please select at least one preprocessing option');
      return;
    }
    
    if (files.length === 0) {
      setError('Please upload at least one TIFF file');
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const response = await api.runPreprocessingCommand({
        stripes: formData.stripes,
        files,
      });

      // Create a job object to add to the list immediately
      const newJob: Job = {
        job_id: response.job_id,
        command: 'preprocessing',
        status: 'pending',
        progress: 0,
        message: 'Preprocessing queued',
        created_at: new Date().toISOString(),
        log_messages: [],
      };

      onJobCreated(newJob);

      // Reset form
      setFormData({ stripes: false });
      setFiles([]);

    } catch (error: any) {
      setError(error.response?.data?.detail || 'Failed to start preprocessing');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="preprocessing-form">
      <h3>Preprocessing</h3>
      <p className="form-description">
        Corrects angle alignment of optical components. Equivalent to <code>shi preprocessing</code>
      </p>

      <form onSubmit={handleSubmit}>
        {/* Preprocessing Options */}
        <div className="form-group">
          <label>Preprocessing Operations</label>
          <div className="checkbox-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                name="stripes"
                checked={formData.stripes}
                onChange={handleInputChange}
              />
              Correct Stripes (--stripes)
              <div className="option-help">
                Correct detector stripes that might introduce false features in final images
              </div>
            </label>
          </div>
        </div>

        {/* File Upload */}
        <div className="form-group">
          <label>Raw Experimental Data <span className="required">*</span></label>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <p>Drop the TIFF files here...</p>
            ) : (
              <p>
                Drag & drop raw experimental data files, or click to select files
                <br />
                <small>Accepts .tif and .tiff files</small>
              </p>
            )}
          </div>
          
          <div className="field-help">
            Upload the folder containing all raw experimental data (input images, dark images, and flat images).
            A subfolder named "no stripe" will be created in each subfolder of the processed directory.
          </div>
        </div>

        {/* File Preview */}
        {files.length > 0 && (
          <div className="file-preview">
            <h4>Selected Files ({files.length})</h4>
            <div className="file-list">
              {files.map((file, index) => (
                <div key={index} className="file-item">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">
                    ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                  </span>
                  <button
                    type="button"
                    onClick={() => removeFile(index)}
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

        {error && <div className="error-message">{error}</div>}

        <button
          type="submit"
          disabled={isSubmitting || !formData.stripes || files.length === 0}
          className="submit-button"
        >
          {isSubmitting ? 'Starting Preprocessing...' : 'Run Preprocessing'}
        </button>
      </form>

      <div className="command-info">
        <h4>About Preprocessing</h4>
        <p>
          Preprocessing operations help correct common issues in experimental data:
        </p>
        <ul>
          <li>
            <strong>Stripe Correction:</strong> Removes detector artifacts that create 
            false features in the final images. These stripes are typically caused by 
            detector inhomogeneities or dead pixels.
          </li>
        </ul>
        <p>
          <strong>Important:</strong> After preprocessing, update your configuration file 
          with the path to the new folder containing the corrected images.
        </p>
      </div>
    </div>
  );
};

export default PreprocessingForm;
import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Job } from '../types';
import * as api from '../services/api';
import './MorphostructuralForm.css';

interface MorphostructuralFormProps {
  onJobCreated: (job: Job) => void;
}

const MorphostructuralForm: React.FC<MorphostructuralFormProps> = ({ onJobCreated }) => {
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

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (files.length === 0) {
      setError('Please upload at least one TIFF file');
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const response = await api.runMorphostructuralCommand(files);

      // Create a job object to add to the list immediately
      const newJob: Job = {
        job_id: response.job_id,
        command: 'morphostructural',
        status: 'pending',
        progress: 0,
        message: 'Morphostructural analysis queued',
        created_at: new Date().toISOString(),
        log_messages: [],
      };

      onJobCreated(newJob);

      // Reset form
      setFiles([]);

    } catch (error: any) {
      setError(error.response?.data?.detail || 'Failed to start morphostructural analysis');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="morphostructural-form">
      <h3>Morphostructural Analysis</h3>
      <p className="form-description">
        Performs morphostructural analysis. Equivalent to <code>shi morphostructural --morphostructural</code>
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Analysis Files <span className="required">*</span></label>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <p>Drop the files here...</p>
            ) : (
              <p>
                Drag & drop files for morphostructural analysis, or click to select
                <br />
                <small>Accepts .tif and .tiff files</small>
              </p>
            )}
          </div>
          
          <div className="field-help">
            Upload absorption and scattering images for morphostructural analysis. 
            The analysis will examine structural characteristics based on scattering and absorption data.
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
          disabled={isSubmitting || files.length === 0}
          className="submit-button"
        >
          {isSubmitting ? 'Starting Analysis...' : 'Run Morphostructural Analysis'}
        </button>
      </form>

      <div className="command-info">
        <h4>About Morphostructural Analysis</h4>
        <p>
          This analysis tool combines scattering and absorption data to examine structural 
          characteristics of your samples. It will automatically detect and analyze patterns 
          in the uploaded images to provide insights into the morphological structure.
        </p>
        <p>
          <strong>Expected Input:</strong> Processed absorption and scattering images from SHI analysis.
        </p>
      </div>
    </div>
  );
};

export default MorphostructuralForm;
// API service for SHI React Web App
import axios from 'axios';
import { Command, Job, JobCreateResponse, HealthResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Commands API
export const getCommands = async (): Promise<Command[]> => {
  const response = await api.get('/commands');
  return response.data;
};

// Jobs API
export const getJobs = async (): Promise<Job[]> => {
  const response = await api.get('/jobs');
  return response.data;
};

export const getJob = async (jobId: string): Promise<Job> => {
  const response = await api.get(`/jobs/${jobId}`);
  return response.data;
};

export const deleteJob = async (jobId: string): Promise<void> => {
  await api.delete(`/jobs/${jobId}`);
};

// Calculate Command
// Step 1: Upload files only
export const uploadFilesForProcessing = async (files: File[]): Promise<JobCreateResponse> => {
  const formData = new FormData();
  
  files.forEach(file => {
    formData.append('files', file);
  });
  
  const response = await api.post('/upload-files', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Step 2: Start processing with parameters and ROI
export const runCalculateCommand = async (data: {
  job_id: string;
  mask_period: number;
  mode: string;
  unwrap_phase?: string;
  average?: boolean;
  export?: boolean;
  angle_after?: boolean;
}): Promise<JobCreateResponse> => {
  const formData = new FormData();
  
  formData.append('job_id', data.job_id);
  formData.append('mask_period', data.mask_period.toString());
  formData.append('mode', data.mode);
  if (data.unwrap_phase) formData.append('unwrap_phase', data.unwrap_phase);
  formData.append('average', data.average?.toString() || 'false');
  formData.append('export', data.export?.toString() || 'false');
  formData.append('angle_after', data.angle_after?.toString() || 'false');
  
  const response = await api.post('/calculate', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Morphostructural Command
export const runMorphostructuralCommand = async (files: File[]): Promise<JobCreateResponse> => {
  const formData = new FormData();
  
  files.forEach(file => {
    formData.append('files', file);
  });
  
  const response = await api.post('/morphostructural', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Preprocessing Command
export const runPreprocessingCommand = async (data: {
  stripes: boolean;
  files: File[];
}): Promise<JobCreateResponse> => {
  const formData = new FormData();
  
  formData.append('stripes', data.stripes.toString());
  
  data.files.forEach(file => {
    formData.append('files', file);
  });
  
  const response = await api.post('/preprocessing', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Clean Command
export const runCleanCommand = async (data: {
  clear_cache?: boolean;
  clear_extra?: boolean;
}): Promise<JobCreateResponse> => {
  const formData = new FormData();
  
  if (data.clear_cache) formData.append('clear_cache', 'true');
  if (data.clear_extra) formData.append('clear_extra', 'true');
  
  const response = await api.post('/clean', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Download Results
export const downloadResults = async (jobId: string): Promise<void> => {
  const response = await api.get(`/download/${jobId}`, {
    responseType: 'blob',
  });
  
  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  
  // Get filename from response headers or use default
  const contentDisposition = response.headers['content-disposition'];
  let filename = `shi_results_${jobId.substring(0, 8)}.zip`;
  
  if (contentDisposition) {
    const filenameMatch = contentDisposition.match(/filename="(.+)"/);
    if (filenameMatch) {
      filename = filenameMatch[1];
    }
  }
  
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
};

// ROI API
export interface ROICoordinates {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface ROIImage {
  image_data: string;
  original_size: { width: number; height: number };
  file_name: string;
}

export const getROIImage = async (jobId: string): Promise<ROIImage> => {
  const response = await api.get(`/roi/image/${jobId}`);
  return response.data;
};

export const saveROI = async (jobId: string, coordinates: ROICoordinates): Promise<void> => {
  await api.post('/roi/save', {
    job_id: jobId,
    coordinates
  });
};

export const getROI = async (jobId: string): Promise<{ job_id: string; coordinates: ROICoordinates }> => {
  const response = await api.get(`/roi/${jobId}`);
  return response.data;
};

export const deleteROI = async (jobId: string): Promise<void> => {
  await api.delete(`/roi/${jobId}`);
};

// Health Check
export const getHealth = async (): Promise<HealthResponse> => {
  const response = await api.get('/health');
  return response.data;
};
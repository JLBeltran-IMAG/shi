// TypeScript types for SHI React Web App

export interface Command {
  name: string;
  description: string;
  options: CommandOption[];
}

export interface CommandOption {
  name: string;
  type: 'int' | 'select' | 'boolean' | 'string';
  required: boolean;
  description: string;
  options?: string[];
}

export interface Job {
  job_id: string;
  command: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  created_at: string;
  completed_at?: string;
  results_path?: string;
  log_messages: string[];
}

// API Request types
export interface CalculateRequest {
  mask_period: number;
  mode: '2d' | '3d' | 'custom';
  unwrap_phase?: string;
  average?: boolean;
  export?: boolean;
  angle_after?: boolean;
  files: File[];
}

export interface MorphostructuralRequest {
  files: File[];
}

export interface PreprocessingRequest {
  stripes: boolean;
  files: File[];
}

export interface CleanRequest {
  clear_cache?: boolean;
  clear_extra?: boolean;
}

// API Response types
export interface JobCreateResponse {
  job_id: string;
  message: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  active_jobs: number;
  available_commands: string[];
  version: string;
}
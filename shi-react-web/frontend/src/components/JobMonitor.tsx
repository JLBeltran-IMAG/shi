import React, { useState } from 'react';
import { Job } from '../types';
import './JobMonitor.css';

interface JobMonitorProps {
  jobs: Job[];
  onJobDeleted: (jobId: string) => void;
  onDownload: (jobId: string) => void;
}

const JobMonitor: React.FC<JobMonitorProps> = ({ jobs, onJobDeleted, onDownload }) => {
  const [expandedJob, setExpandedJob] = useState<string | null>(null);

  const getStatusIcon = (status: Job['status']) => {
    switch (status) {
      case 'pending':
        return '⏳';
      case 'processing':
        return '⚙️';
      case 'completed':
        return '✅';
      case 'failed':
        return '❌';
      default:
        return '❓';
    }
  };

  const getStatusClass = (status: Job['status']) => {
    return `job-status status-${status}`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const formatDuration = (start: string, end?: string) => {
    const startTime = new Date(start).getTime();
    const endTime = end ? new Date(end).getTime() : Date.now();
    const duration = Math.floor((endTime - startTime) / 1000);
    
    if (duration < 60) return `${duration}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m ${duration % 60}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  const toggleJobExpansion = (jobId: string) => {
    setExpandedJob(expandedJob === jobId ? null : jobId);
  };

  if (jobs.length === 0) {
    return (
      <div className="job-monitor">
        <h2>Processing Jobs</h2>
        <div className="no-jobs">
          <div className="no-jobs-icon">📋</div>
          <p>No processing jobs yet.</p>
          <p>Submit a command above to see jobs here.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="job-monitor">
      <h2>Processing Jobs ({jobs.length})</h2>
      
      <div className="jobs-list">
        {jobs.map((job) => (
          <div key={job.job_id} className={`job-card ${job.status}`}>
            <div className="job-header" onClick={() => toggleJobExpansion(job.job_id)}>
              <div className="job-info">
                <div className="job-title">
                  <span className="job-command">{job.command}</span>
                  <span className="job-id">#{job.job_id.substring(0, 8)}</span>
                </div>
                <div className="job-meta">
                  <span className={getStatusClass(job.status)}>
                    {getStatusIcon(job.status)} {job.status.toUpperCase()}
                  </span>
                  <span className="job-time">
                    Started: {formatDate(job.created_at)}
                  </span>
                  {job.completed_at && (
                    <span className="job-duration">
                      Duration: {formatDuration(job.created_at, job.completed_at)}
                    </span>
                  )}
                </div>
              </div>
              
              <div className="job-progress">
                <div className="progress-bar">
                  <div 
                    className={`progress-fill status-${job.status}`}
                    style={{ width: `${job.progress * 100}%` }}
                  ></div>
                </div>
                <div className="progress-text">
                  {Math.round(job.progress * 100)}%
                </div>
              </div>
            </div>

            <div className="job-message">
              {job.message}
            </div>

            {/* Expanded Details */}
            {expandedJob === job.job_id && (
              <div className="job-details">
                <div className="job-actions">
                  {job.status === 'completed' && job.results_path && (
                    <button
                      className="action-button download"
                      onClick={() => onDownload(job.job_id)}
                    >
                      📥 Download Results
                    </button>
                  )}
                  
                  {(job.status === 'completed' || job.status === 'failed') && (
                    <button
                      className="action-button delete"
                      onClick={() => onJobDeleted(job.job_id)}
                    >
                      🗑️ Delete Job
                    </button>
                  )}
                </div>

                {/* Log Messages */}
                {job.log_messages.length > 0 && (
                  <div className="job-logs">
                    <h4>Log Messages</h4>
                    <div className="log-messages">
                      {job.log_messages.map((message, index) => (
                        <div key={index} className="log-message">
                          {message}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Job Details */}
                <div className="job-metadata">
                  <div className="metadata-row">
                    <strong>Job ID:</strong> {job.job_id}
                  </div>
                  <div className="metadata-row">
                    <strong>Command:</strong> {job.command}
                  </div>
                  <div className="metadata-row">
                    <strong>Created:</strong> {formatDate(job.created_at)}
                  </div>
                  {job.completed_at && (
                    <div className="metadata-row">
                      <strong>Completed:</strong> {formatDate(job.completed_at)}
                    </div>
                  )}
                  {job.status === 'processing' && (
                    <div className="metadata-row">
                      <strong>Running for:</strong> {formatDuration(job.created_at)}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default JobMonitor;
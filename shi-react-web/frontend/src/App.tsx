import React, { useState, useEffect } from 'react';
import './App.css';
import CommandSelector from './components/CommandSelector';
import CalculateForm from './components/CalculateForm';
import MorphostructuralForm from './components/MorphostructuralForm';
import PreprocessingForm from './components/PreprocessingForm';
import CleanForm from './components/CleanForm';
import JobMonitor from './components/JobMonitor';
import { Command, Job } from './types';
import * as api from './services/api';

function App() {
  const [selectedCommand, setSelectedCommand] = useState<string>('calculate');
  const [commands, setCommands] = useState<Command[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadCommands();
    loadJobs();
    // Set up periodic job refresh
    const interval = setInterval(loadJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadCommands = async () => {
    try {
      const commandsData = await api.getCommands();
      setCommands(commandsData);
    } catch (error) {
      console.error('Failed to load commands:', error);
    }
  };

  const loadJobs = async () => {
    try {
      const jobsData = await api.getJobs();
      setJobs(jobsData);
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to load jobs:', error);
      setIsLoading(false);
    }
  };

  const handleJobCreated = (job: Job) => {
    setJobs(prevJobs => [job, ...prevJobs]);
  };

  const handleJobDeleted = async (jobId: string) => {
    try {
      await api.deleteJob(jobId);
      setJobs(prevJobs => prevJobs.filter(job => job.job_id !== jobId));
    } catch (error) {
      console.error('Failed to delete job:', error);
    }
  };

  const handleDownload = async (jobId: string) => {
    try {
      await api.downloadResults(jobId);
    } catch (error) {
      console.error('Failed to download results:', error);
    }
  };

  const renderCommandForm = () => {
    switch (selectedCommand) {
      case 'calculate':
        return <CalculateForm onJobCreated={handleJobCreated} />;
      case 'morphostructural':
        return <MorphostructuralForm onJobCreated={handleJobCreated} />;
      case 'preprocessing':
        return <PreprocessingForm onJobCreated={handleJobCreated} />;
      case 'clean':
        return <CleanForm onJobCreated={handleJobCreated} />;
      default:
        return <div>Select a command to begin</div>;
    }
  };

  if (isLoading) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>Spatial Harmonic Imaging</h1>
          <p>Loading...</p>
        </header>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <div className="container">
          <div className="brand">
            <span className="brand-icon">🔬</span>
            <div>
              <h1>Spatial Harmonic Imaging</h1>
              <div className="subtitle">Multi-contrast X-ray analysis platform</div>
            </div>
          </div>
        </div>
      </header>

      <main className="App-main">
        <div className="main-layout">
          <div className="content-panel">
            <div className="section">
              <div className="section-header">
                <CommandSelector 
                  commands={commands}
                  selectedCommand={selectedCommand}
                  onCommandSelect={setSelectedCommand}
                />
              </div>
              
              <div className="section-content">
                {renderCommandForm()}
              </div>
            </div>
          </div>
          
          <div className="sidebar-panel">
            <JobMonitor 
              jobs={jobs}
              onJobDeleted={handleJobDeleted}
              onDownload={handleDownload}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

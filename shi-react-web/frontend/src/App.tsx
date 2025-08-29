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
          <h1>🔬 SHI - Spatial Harmonic Imaging</h1>
          <p>Loading...</p>
        </header>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔬 SHI - Spatial Harmonic Imaging</h1>
        <p>Web interface for spatial harmonic imaging processing</p>
      </header>

      <main className="App-main">
        <div className="command-section">
          <CommandSelector 
            commands={commands}
            selectedCommand={selectedCommand}
            onCommandSelect={setSelectedCommand}
          />
          
          <div className="command-form">
            {renderCommandForm()}
          </div>
        </div>

        <div className="jobs-section">
          <JobMonitor 
            jobs={jobs}
            onJobDeleted={handleJobDeleted}
            onDownload={handleDownload}
          />
        </div>
      </main>
    </div>
  );
}

export default App;

import React from 'react';
import { Command } from '../types';
import './CommandSelector.css';

interface CommandSelectorProps {
  commands: Command[];
  selectedCommand: string;
  onCommandSelect: (command: string) => void;
}

const CommandSelector: React.FC<CommandSelectorProps> = ({
  commands,
  selectedCommand,
  onCommandSelect,
}) => {
  return (
    <div className="command-selector">
      <div className="selector-header">
        <h2 className="section-title">Processing Commands</h2>
        <p className="section-subtitle">Select an operation to configure</p>
      </div>
      
      <div className="command-tabs">
        {commands.map((command) => (
          <button
            key={command.name}
            className={`command-tab ${selectedCommand === command.name ? 'active' : ''}`}
            onClick={() => onCommandSelect(command.name)}
          >
            <div className="tab-content">
              <div className="command-name">{command.name}</div>
              <div className="command-description">{command.description}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default CommandSelector;
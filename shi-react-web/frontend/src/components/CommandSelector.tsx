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
      <h2>Available Commands</h2>
      <div className="command-tabs">
        {commands.map((command) => (
          <button
            key={command.name}
            className={`command-tab ${selectedCommand === command.name ? 'active' : ''}`}
            onClick={() => onCommandSelect(command.name)}
          >
            <div className="command-name">{command.name}</div>
            <div className="command-description">{command.description}</div>
          </button>
        ))}
      </div>
      
      {selectedCommand && (
        <div className="command-help">
          <h3>Command: {selectedCommand}</h3>
          <p>
            {commands.find(cmd => cmd.name === selectedCommand)?.description}
          </p>
          
          <div className="cli-equivalent">
            <strong>CLI Equivalent:</strong>
            <code>shi {selectedCommand}</code>
          </div>
        </div>
      )}
    </div>
  );
};

export default CommandSelector;
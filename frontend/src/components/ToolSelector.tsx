import React from 'react';

interface Tool {
  name: string;
  description: string;
  requires_file: boolean;
}

interface ToolSelectorProps {
  tool: Tool;
  icon: React.ReactNode;
  isSelected: boolean;
  onSelect: () => void;
  messageCount?: number;
}

const ToolSelector: React.FC<ToolSelectorProps> = ({ tool, icon, isSelected, onSelect, messageCount = 0 }) => {
  return (
    <div
      className={`tool-selector cursor-pointer transition-all duration-200 ${
        isSelected ? 'active' : ''
      }`}
      onClick={onSelect}
    >
      <div className="flex items-start space-x-3">
        <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
          isSelected ? 'bg-spotify-green text-spotify-black' : 'bg-spotify-gray text-spotify-light-gray'
        }`}>
          {icon}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h3 className={`text-sm font-medium ${
              isSelected ? 'text-white' : 'text-spotify-light-gray'
            }`}>
              {tool.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </h3>
            {messageCount > 0 && (
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                isSelected ? 'bg-spotify-green text-spotify-black' : 'bg-spotify-gray text-spotify-light-gray'
              }`}>
                {messageCount}
              </span>
            )}
          </div>
          <p className="text-xs text-spotify-light-gray mt-1 line-clamp-2">
            {tool.description}
          </p>
          {tool.requires_file && (
            <div className="mt-1">
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-spotify-green text-spotify-black">
                File Upload
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ToolSelector; 
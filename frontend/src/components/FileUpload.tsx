import React, { useRef, useState } from 'react';
import { Upload, X } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (fileName: string, filePath?: string) => void;
  selectedTab?: string | null;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, selectedTab }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = async (file: File) => {
    if (file) {
      try {
        // Upload file to backend
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData,
        });
        
        if (response.ok) {
          const result = await response.json();
          onFileUpload(file.name, result.path);
        } else {
          console.error('Upload failed');
          onFileUpload(file.name);
        }
      } catch (error) {
        console.error('Upload error:', error);
        onFileUpload(file.name);
      }
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex-shrink-0">
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleChange}
        accept={
          selectedTab === 'songwriting' 
            ? '.pdf,.txt,.rdf'
            : selectedTab === 'production'
            ? '.mid,.midi'
            : '.mid,.midi,.wav,.mp3,.aiff,.flac,.json,.xml,.pdf,.txt'
        }
      />
      
      <button
        onClick={onButtonClick}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`btn-secondary flex items-center space-x-2 transition-all duration-200 ${
          dragActive ? 'border-spotify-green bg-spotify-gray' : ''
        }`}
      >
        <Upload className="w-4 h-4" />
        <span>
          {selectedTab === 'songwriting' 
            ? 'Upload Lyrics' 
            : selectedTab === 'production'
            ? 'Upload MIDI'
            : 'Upload'}
        </span>
      </button>
    </div>
  );
};

export default FileUpload; 
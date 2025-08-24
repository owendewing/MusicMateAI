import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, Music, Mic, Music2, Headphones, Layout, Zap, Monitor, Search } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import FileUpload from './components/FileUpload';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Tool {
  name: string;
  description: string;
  requires_file: boolean;
}

interface Tab {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

const App: React.FC = () => {
  const [chatSessions, setChatSessions] = useState<{ [tabId: string]: Message[] }>({});
  const [inputMessage, setInputMessage] = useState('');
  const [selectedTab, setSelectedTab] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Get current messages for selected tab
  const currentMessages = selectedTab ? (chatSessions[selectedTab] || []) : [];

  // Define the three main tabs
  const tabs: Tab[] = [
    {
      id: 'songwriting',
      name: 'Songwriting Assistant',
      description: 'Lyrics, melodies, and creative inspiration',
      icon: <Mic className="w-6 h-6" />
    },
    {
      id: 'production',
      name: 'Production Assistant', 
      description: 'Instruments, samples, arrangement, and energy',
      icon: <Music2 className="w-6 h-6" />
    },
    {
      id: 'daw_theory',
      name: 'DAW & Music Theory',
      description: 'DAW functionality, music theory, and technical advice',
      icon: <Monitor className="w-6 h-6" />
    }
  ];

  const toolIcons: { [key: string]: React.ReactNode } = {
    write_lyrics_tool: <Mic className="w-5 h-5" />,
    suggest_instruments_tool: <Music2 className="w-5 h-5" />,
    suggest_sample_packs_tool: <Headphones className="w-5 h-5" />,
    arrangement_advice_tool: <Layout className="w-5 h-5" />,
    song_energy_tool: <Zap className="w-5 h-5" />,
    daw_functionality_tool: <Monitor className="w-5 h-5" />,
    tavily_search_tool: <Search className="w-5 h-5" />,
  };

  // Fetch available tools on component mount
  useEffect(() => {
    fetchTools();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [currentMessages]);

  const fetchTools = async () => {
    try {
      const response = await fetch('/tools');
      const data = await response.json();
      setTools(data.tools);
    } catch (error) {
      console.error('Error fetching tools:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && !uploadedFile) return;
    if (!selectedTab) return;

    const userMessage: Message = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
    };

    // Add message to current tab's chat session
    setChatSessions(prev => ({
      ...prev,
      [selectedTab]: [...(prev[selectedTab] || []), userMessage]
    }));
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          history: currentMessages.map(msg => ({
            role: msg.role,
            content: msg.content,
          })),
          selected_tool: null, // Let AI choose tools automatically
          file_upload: uploadedFile,
          file_path: uploadedFilePath,
        }),
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
      };

      // Add assistant message to current tab's chat session
      setChatSessions(prev => ({
        ...prev,
        [selectedTab]: [...(prev[selectedTab] || []), assistantMessage]
      }));
    } catch (error) {
      console.error('Error sending message:', error);
      let errorContent = 'Sorry, I encountered an error. Please try again.';
      
      // Provide more specific error messages
      if (error instanceof Error) {
        if (error.message.includes('recursion_limit')) {
          errorContent = 'The AI is taking too long to process your request. Please try rephrasing your question or uploading a smaller file.';
        } else if (error.message.includes('network')) {
          errorContent = 'Network error. Please check your connection and try again.';
        }
      }
      
      const errorMessage: Message = {
        role: 'assistant',
        content: errorContent,
        timestamp: new Date(),
      };
      setChatSessions(prev => ({
        ...prev,
        [selectedTab]: [...(prev[selectedTab] || []), errorMessage]
      }));
    } finally {
      setIsLoading(false);
      // Don't clear uploaded file info - keep it for subsequent messages
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleTabSelect = (tabId: string) => {
    setSelectedTab(selectedTab === tabId ? null : tabId);
    // Clear input when switching tabs
    setInputMessage('');
    setUploadedFile(null);
    setUploadedFilePath(null);
  };

  const handleFileUpload = (fileName: string, filePath?: string) => {
    // Check file type based on selected tab
    const fileExtension = fileName.split('.').pop()?.toLowerCase();
    
    if (selectedTab === 'songwriting') {
      const allowedExtensions = ['pdf', 'txt', 'rdf'];
      if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
        alert('Error: Only PDF, TXT, and RDF files are allowed for the Songwriting Assistant.');
        return;
      }
    } else if (selectedTab === 'production') {
      if (fileExtension !== 'midi' && fileExtension !== 'mid') {
        alert('Error: Only MIDI files are allowed for the Production Assistant.');
        return;
      }
    } else if (selectedTab === 'daw_theory') {
      alert('Error: File upload is not available for the DAW & Music Theory Assistant.');
      return;
    }
    
    setUploadedFile(fileName);
    setUploadedFilePath(filePath || null);
    
    // Add confirmation message to chat
    if (selectedTab) {
      const confirmationMessage: Message = {
        role: 'assistant',
        content: `✅ File uploaded successfully: ${fileName}\n\nI can now analyze this file and help you with your music production needs. What would you like to know about it?`,
        timestamp: new Date(),
      };
      
      setChatSessions(prev => ({
        ...prev,
        [selectedTab]: [...(prev[selectedTab] || []), confirmationMessage]
      }));
    }
  };

  return (
    <div className="min-h-screen bg-spotify-black flex flex-col">
      {/* Header */}
      <header className="bg-spotify-dark-gray border-b border-spotify-gray p-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Music className="w-8 h-8 text-spotify-green" />
            <h1 className="text-2xl font-bold text-white">Music Mate AI</h1>
          </div>
          <div className="text-spotify-light-gray text-sm">
            Your creative companion for music production
          </div>
        </div>
      </header>

      <div className="flex-1 flex max-w-6xl mx-auto w-full">
        {/* Sidebar */}
        <div className="w-80 bg-spotify-dark-gray border-r border-spotify-gray p-4 overflow-y-auto">
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-white mb-4">Assistant Types</h2>
            
            {tabs.map((tab) => (
              <div
                key={tab.id}
                className={`tab-selector cursor-pointer transition-all duration-200 p-4 rounded-lg border ${
                  selectedTab === tab.id 
                    ? 'bg-spotify-green border-spotify-green text-spotify-black' 
                    : 'bg-spotify-gray border-spotify-gray text-spotify-light-gray hover:bg-spotify-dark-gray'
                }`}
                onClick={() => handleTabSelect(tab.id)}
              >
                <div className="flex items-start space-x-3">
                  <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
                    selectedTab === tab.id ? 'bg-spotify-black text-spotify-green' : 'bg-spotify-dark-gray text-spotify-light-gray'
                  }`}>
                    {tab.icon}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h3 className={`text-sm font-medium ${
                        selectedTab === tab.id ? 'text-spotify-black' : 'text-white'
                      }`}>
                        {tab.name}
                      </h3>
                      {chatSessions[tab.id]?.length > 0 && (
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          selectedTab === tab.id ? 'bg-spotify-black text-spotify-green' : 'bg-spotify-dark-gray text-spotify-light-gray'
                        }`}>
                          {chatSessions[tab.id].length}
                        </span>
                      )}
                    </div>
                    <p className={`text-xs mt-1 line-clamp-2 ${
                      selectedTab === tab.id ? 'text-spotify-black' : 'text-spotify-light-gray'
                    }`}>
                      {tab.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}


          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!selectedTab ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-spotify-light-gray">
                  <Music className="w-16 h-16 mx-auto mb-4 text-spotify-green" />
                  <h2 className="text-xl font-semibold mb-2">Welcome to Music Mate AI</h2>
                  <p className="text-sm">
                    Select an assistant type from the sidebar and start creating music!
                  </p>
                </div>
              </div>
            ) : currentMessages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-spotify-light-gray">
                  <div className="mb-4">
                    {tabs.find(tab => tab.id === selectedTab)?.icon || <Music className="w-16 h-16 mx-auto text-spotify-green" />}
                  </div>
                  <h2 className="text-xl font-semibold mb-2">Start chatting with {tabs.find(tab => tab.id === selectedTab)?.name}</h2>
                  <p className="text-sm">
                    Ask questions or upload files to get started! The AI will automatically choose the right tools for your request.
                  </p>
                </div>
              </div>
            ) : (
              currentMessages.map((message, index) => (
                <ChatMessage
                  key={index}
                  message={message}
                  isLoading={isLoading && index === currentMessages.length - 1}
                />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-spotify-gray p-4 bg-spotify-dark-gray">
            <div className="flex space-x-3">
              {selectedTab !== 'daw_theory' && (
                <FileUpload onFileUpload={handleFileUpload} selectedTab={selectedTab} />
              )}
              
              <div className="flex-1">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me about music production..."
                  className="input-field w-full resize-none"
                  rows={1}
                  disabled={isLoading}
                />
              </div>
              
              <button
                onClick={handleSendMessage}
                disabled={isLoading || (!inputMessage.trim() && !uploadedFile) || !selectedTab}
                className="btn-primary flex items-center space-x-2"
              >
                <Send className="w-5 h-5" />
                <span>Send</span>
              </button>
            </div>
            
            {selectedTab && (
              <div className="mt-2 text-sm text-spotify-light-gray">
                Assistant: <span className="text-spotify-green">{tabs.find(tab => tab.id === selectedTab)?.name}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App; 
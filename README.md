# Music Mate AI

A full-stack AI assistant for music producers, featuring a chat interface with specialized tools for lyrics generation, instrument suggestions, arrangement advice, and more. Built with FastAPI, React, and LangChain.

## Features

### 🎵 AI Tools
- **Lyrics Generation**: Create lyrics based on prompts, style, mood, or artist references
- **Instrument Suggestions**: Get recommendations for instruments and synth presets
- **Sample Pack Recommendations**: Find the perfect samples and loops for your project
- **Arrangement Advice**: Get song structure and arrangement suggestions
- **DAW Knowledge Base**: Advanced RAG system with music production manuals and textbooks
- **Web Search**: Search for music production information and tutorials
- **Advanced Retrieval**: Contextual compression and parent document retrieval techniques

### 🎨 UI/UX
- **Spotify-inspired Design**: Green and black color scheme with modern aesthetics
- **Chat Interface**: Real-time conversation with the AI assistant
- **Tool Selection**: Easy-to-use sidebar for selecting specific tools
- **File Upload**: Support for various audio and project file formats
- **Responsive Design**: Works on desktop and mobile devices

### 🚀 Technical Features
- **Full-Stack Architecture**: React frontend with FastAPI backend
- **Docker Support**: Easy deployment with Docker and docker-compose
- **UV Integration**: Modern Python dependency management
- **Advanced RAG System**: Retrieval-augmented generation with music production data
- **Real-time Chat**: Streaming responses with loading states
- **Tool Usage Display**: Visual indicators showing which tools are being used
- **Global State Management**: Persistent analysis data across tool calls

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key
- Tavily API key (optional, for web search)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MusicMateAI
```

### 2. Set Environment Variables
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Run with Docker
```bash
# Build and start the application
docker-compose up --build

# The application will be available at http://localhost:3000
```

### 4. Alternative: Development Setup

#### Using UV (Recommended)
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start the backend
cd backend
uv run python main.py

# In another terminal, start the frontend
cd frontend
npm install
npm start
```

#### Using Traditional Python
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend
cd backend
python main.py

# In another terminal, start the frontend
cd frontend
npm install
npm start
```

## Usage

### Basic Chat
1. Open the application in your browser
2. Type your question in the chat input
3. Press Enter or click Send to get a response

### Using Specific Tools
1. Select a tool from the sidebar (e.g., "Write Lyrics Tool")
2. Type your request in the chat
3. The AI will use the selected tool to provide a specialized response

### File Upload
1. Click the Upload button
2. Select your audio or project file
3. The file information will be included in your chat request

### Example Prompts
- **Lyrics**: "Write lyrics for a pop song about falling in love"
- **Instruments**: "Suggest instruments for a jazz track in C major"
- **Arrangement**: "Help me arrange a rock song with guitar, bass, and drums"
- **DAW Help**: "How do I use Flex Time in Logic Pro?"
- **Energy Analysis**: "Analyze the energy curve of my electronic track"
- **Music Theory**: "What are the chord progressions in C major?"
- **Production**: "How do I set up a basic mixing chain?"

## Project Structure

```
MusicMateAI/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application file
│   ├── clean_markdown.py   # Markdown processing
│   ├── rag.py              # RAG system with lazy initialization
│   └── midi_processor.py   # MIDI file analysis
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── FileUpload.tsx
│   │   │   └── ToolSelector.tsx
│   │   ├── App.tsx         # Main app component
│   │   └── index.tsx       # Entry point
│   ├── package.json        # Node.js dependencies
│   └── tailwind.config.js  # Tailwind CSS config
├── data/                   # Music production documents
│   ├── *.pdf              # DAW manuals, music theory, production guides
├── uploads/               # User uploaded files
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker orchestration
├── nginx.conf              # Nginx configuration
├── pyproject.toml          # UV project configuration
├── requirements.txt        # Python dependencies
├── uv.lock                 # UV dependency lock file
└── README.md              # This file
```

## API Endpoints

- `GET /` - Health check
- `GET /tools` - Get available tools
- `POST /chat` - Send chat message and get response

## Development

### Backend Development
```bash
cd backend
uv run python main.py
```

### Frontend Development
```bash
cd frontend
npm start
```

### Running Tests
```bash
# Backend tests
uv run pytest

# Frontend tests
cd frontend
npm test
```

### Code Formatting
```bash
# Backend
uv run black backend/
uv run isort backend/

# Frontend
cd frontend
npm run format
```

## Docker Commands

```bash
# Build the image
docker build -t music-production-ai .

# Run the container
docker run -p 3000:80 -e OPENAI_API_KEY=your_key music-production-ai

# Run with docker-compose
docker-compose up --build

# Stop the application
docker-compose down

# View logs
docker-compose logs -f
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `TAVILY_API_KEY` | Tavily API key for web search | No |
| `ENVIRONMENT` | Set to "development" for dev mode | No |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

### Advanced RAG & Retrieval
- **Multi-modal RAG**: Integrate audio analysis with text retrieval
- **Hybrid Search**: Combine dense vector search with keyword search
- **Real-time Learning**: Update knowledge base as users interact
- **Personalized Retrieval**: Adapt to user's music production style

### AI Features
- **Audio Analysis Integration**: MIDI to audio conversion and analysis
- **Multi-Agent System**: Specialized agents for mixing, mastering, composition
- **Conversational Memory**: Remember user preferences across sessions
- **Real-time Audio Processing**: Live feedback during production

### Production Workflow
- **DAW Plugin Development**: VST/AU plugins with AI integration
- **Project Templates**: AI-generated templates based on genre/style
- **Collaborative Features**: Multi-user sessions with AI assistance
- **Quality Assessment**: AI-powered mix/mastering evaluation

### Content & Learning
- **Interactive Tutorials**: AI-guided learning paths
- **Video Integration**: Extract knowledge from tutorials
- **Community Knowledge**: Crowdsourced tips and techniques
- **Adaptive Learning**: Personalized learning based on progress

### Technical Improvements
- **Performance Optimization**: Caching, distributed processing
- **Scalability**: Microservices, cloud deployment
- **Security & Privacy**: Data encryption, secure audio handling
- **Multi-language Support**: Global accessibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on GitHub or contact the development team.

---

**Note**: This application is designed for educational and creative purposes. Always respect copyright and licensing when using AI-generated content in your music projects.
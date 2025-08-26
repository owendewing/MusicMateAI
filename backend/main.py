from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import asyncio
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.documents import Document
import json
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from clean_markdown import clean_markdown
from rag import graph

# Load environment variables
load_dotenv()

# Global state for storing analysis data
global_midi_analysis = {}
global_lyrics_analysis = {}

app = FastAPI(title="Music Mate AI", version="1.0.0")

# CORS middleware - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    selected_tool: Optional[str] = None
    file_upload: Optional[str] = None
    file_path: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    error: Optional[str] = None

class ToolInfo(BaseModel):
    name: str
    description: str
    requires_file: bool = False

@tool
def write_lyrics_tool(prompt: str) -> str:
    """
    Generates lyrics based on a user prompt. Can mimic a specific artist's style, mood, or genre.
    Optionally continues from existing lyrics or expands on a given idea.
    This tool should be called after process_lyrics_file to generate new lyrics based on the analysis.
    The generated lyrics can be downloaded as a PDF file.
    """
    try:
        # Check if we have lyrics analysis data to work with
        global global_lyrics_analysis
        
        print(f"DEBUG: write_lyrics_tool called with prompt: {prompt}")
        print(f"DEBUG: global_lyrics_analysis: {global_lyrics_analysis}")
        
        if global_lyrics_analysis:
            # Use the analyzed lyrics data to inform the generation
            lyrics_data = global_lyrics_analysis
            
            # Create a context-aware prompt
            context_prompt = f"""
            Based on the analyzed lyrics file '{lyrics_data['filename']}' with the following characteristics:
            - Structure: {lyrics_data['structure']}
            - Content: {lyrics_data['content'][:500]}...
            
            User request: {prompt}
            
            Generate the actual lyrics based on the user's request. Make it specific and complete. You can ask clarifying questions if needed to better understand their vision.
            """
            
            # Use the LLM to generate contextual lyrics
            messages = [SystemMessage(content="You are a professional songwriter. Generate complete lyrics based on the user's request. You can ask clarifying questions if needed to better understand their vision.")]
            messages.append(HumanMessage(content=context_prompt))
            
            response = model.invoke(messages)
            lyrics_content = response.content
            
            # Format the response with correction question
            result = f"""
{lyrics_content}

Would you like me to make any corrections or changes to these lyrics?
            """
            return result
        else:
            # No lyrics file analyzed, generate from scratch
            context_prompt = f"""
            User request: {prompt}
            
            Generate the actual lyrics based on the user's request. Make it specific and complete. You can ask clarifying questions if needed to better understand their vision.
            """
            
            messages = [SystemMessage(content="You are a professional songwriter. Generate complete lyrics based on the user's request. You can ask clarifying questions if needed to better understand their vision.")]
            messages.append(HumanMessage(content=context_prompt))
            
            response = model.invoke(messages)
            lyrics_content = response.content
            
            # Format the response with correction question
            result = f"""
{lyrics_content}

Would you like me to make any corrections or changes to these lyrics?
            """
            return result
            
    except Exception as e:
        return f"Error generating lyrics: {str(e)}"

@tool
def suggest_instruments_or_samples_tool(key: str = None, genre: str = None, project_tracks: list = None, question: str = None) -> str:
    """
    Suggests instruments, sample packs, or loops for a song based on key, genre, existing project tracks, and the user's specific question.
    Returns contextual recommendations based on the query and song characteristics. If the user asks for instruments, sample packs, or loops, use this tool.
    If you find multiple instruments, sample packs, or loops that would work well for the song, format your response as a list of recommendations. If a user
    has uploaded midi files and asks for instruments, sample packs, or loops, use the processed midi file to suggest instruments, sample packs, or loops.
    """
    try:
        # Check if we have MIDI analysis data to work with
        global global_midi_analysis
        
        print(f"DEBUG: suggest_instruments_or_samples_tool called with key={key}, genre={genre}, question={question}")
        print(f"DEBUG: global_midi_analysis: {global_midi_analysis}")
        
        # Use MIDI data if available, otherwise use provided parameters
        if global_midi_analysis:
            midi_data = global_midi_analysis
            actual_key = midi_data.get('key', key)
            actual_tempo = midi_data.get('tempo_info', {}).get('primary_tempo')
            actual_time_signature = midi_data.get('time_signature')
            filename = midi_data.get('filename')
            
            print(f"DEBUG: Using MIDI data - Key: {actual_key}, Tempo: {actual_tempo}, File: {filename}")
            
            # Create a context-aware prompt using MIDI data
            context_prompt = f"""
            Based on the analyzed MIDI file '{filename}' with the following characteristics:
            - Key: {actual_key}
            - Tempo: {actual_tempo} BPM
            - Time Signature: {actual_time_signature}
            - Genre: {genre or 'Unknown'}
            - User question: {question or 'General instrument/sample suggestions'}
            
            Please suggest specific instruments, sample packs, or loops that would work well with this musical context.
            Consider the key, tempo, and style when making recommendations.
            """
        else:
            # No MIDI file analyzed, use provided parameters
            context_prompt = f"""
            Based on the following song characteristics:
            - Key: {key or 'Unknown'}
            - Genre: {genre or 'Unknown'}
            - User question: {question or 'General instrument/sample suggestions'}
            
            Please suggest specific instruments, sample packs, or loops that would work well for this context.
            """
        
        # Use the LLM to generate contextual suggestions
        messages = [SystemMessage(content="You are a music production expert who provides specific, contextual instrument and sample recommendations.")]
        messages.append(HumanMessage(content=context_prompt))
        
        print(f"DEBUG: Context prompt for instruments tool: {context_prompt}")
        response = model.invoke(messages)
        print(f"DEBUG: Instruments tool response: {response.content}")
        
        # Ensure we always return a response
        if response.content and response.content.strip():
            return response.content
        else:
            # Fallback response if the model returns empty content
            if global_midi_analysis:
                return f"Based on your MIDI file analysis (Key: {actual_key}, Tempo: {actual_tempo} BPM), here are some instrument and sample recommendations for making your song more danceable:\n\n- Kick Drums: Use punchy, tight kicks around 60-80Hz\n- Hi-Hats: Add crisp, bright hi-hats for energy\n- Synth Bass: Deep, rhythmic bass in the key of {actual_key}\n- Lead Synths: Bright, catchy leads for hooks\n- Percussion: Add shakers, tambourines for groove\n- Pads: Atmospheric pads to fill out the sound"
            else:
                return "Here are some general instrument and sample recommendations for making your song more danceable:\n\n- Kick Drums: Use punchy, tight kicks around 60-80Hz\n- Hi-Hats: Add crisp, bright hi-hats for energy\n- Synth Bass: Deep, rhythmic bass\n- Lead Synths: Bright, catchy leads for hooks\n- Percussion: Add shakers, tambourines for groove\n- Pads: Atmospheric pads to fill out the sound"
        
    except Exception as e:
        return f"Error generating instrument/sample suggestions: {str(e)}"

@tool
def arrangement_advice_tool(project_summary: str, target_style: str = None, mood: str = None) -> str:
    """
    Provides arrangement suggestions for a song. Can suggest bridges, breakdowns, drops, or track layering ideas.
    Input: textual summary of project (track types, key, tempo, etc.)
    """
    try:
        # Check if we have analysis data to work with
        global global_midi_analysis, global_lyrics_analysis
        
        # Build context from available data
        context_parts = []
        
        if global_midi_analysis:
            midi_data = global_midi_analysis
            context_parts.append(f"""
            MIDI Analysis Data:
            - File: {midi_data.get('filename')}
            - Key: {midi_data.get('key')}
            - Tempo: {midi_data.get('tempo_info', {}).get('primary_tempo')} BPM
            - Time Signature: {midi_data.get('time_signature')}
            - Duration: {midi_data.get('basic_info', {}).get('length_seconds', 0):.1f} seconds
            - Tracks: {midi_data.get('basic_info', {}).get('number_of_tracks', 0)}
            """)
        
        if global_lyrics_analysis:
            lyrics_data = global_lyrics_analysis
            context_parts.append(f"""
            Lyrics Analysis Data:
            - File: {lyrics_data.get('filename')}
            - Structure: {lyrics_data.get('structure')}
            - Word count: {lyrics_data.get('word_count')}
            - Line count: {lyrics_data.get('line_count')}
            """)
        
        # Create comprehensive context
        if context_parts:
            context = "\n".join(context_parts)
            full_prompt = f"""
            Based on the following analysis data:
            {context}
            
            Project Summary: {project_summary}
            Target Style: {target_style or 'Not specified'}
            Mood: {mood or 'Not specified'}
            
            Please provide specific arrangement suggestions including:
            - Song structure recommendations
            - Bridge and breakdown ideas
            - Track layering suggestions
            - Energy curve recommendations
            - Transition ideas between sections
            """
        else:
            # No analysis data available, use basic prompt
            full_prompt = f"""
            Project Summary: {project_summary}
            Target Style: {target_style or 'Not specified'}
            Mood: {mood or 'Not specified'}
            
            Please provide arrangement suggestions for this song.
            """
        
        # Use the LLM to generate arrangement advice
        messages = [SystemMessage(content="You are a music production expert who provides detailed arrangement and structure advice for songs.")]
        messages.append(HumanMessage(content=full_prompt))
        
        response = model.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"Error generating arrangement advice: {str(e)}"


@tool
def process_lyrics_file(question: str, filename: str = None) -> str:
    """
    Expert tool that analyzes uploaded lyrics files. If the user has uploaded a lyrics file, use this tool to break down the lyrics into sections, 
    such as verses, choruses, bridges, and drops. Identify characteristics of the lyrics, such as the style, mood, and structure of the lyrics.
    Identify areas where the lyrics could be improved, such as rhyme scheme, word choice, and overall structure, or areas that are unfinished. 
    This tool should be called first to analyze the lyrics, then the write_lyrics_tool should be called to generate new lyrics based on the analysis.
    """
    try:
        print(f"DEBUG: process_lyrics_file called with filename: {filename}")
        print(f"DEBUG: process_lyrics_file called with question: {question}")
        # Find the uploaded lyrics file
        upload_dir = Path("uploads")
        
        if filename:
            lyrics_file = upload_dir / filename
            if not lyrics_file.exists():
                return f"Lyrics file '{filename}' not found in uploads directory."
        else:
            # Use the most recently uploaded lyrics file
            lyrics_files = list(upload_dir.glob("*.txt")) + list(upload_dir.glob("*.pdf"))
            if not lyrics_files:
                return "No lyrics files found in uploads directory. Please upload a lyrics file first."
            
            lyrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            lyrics_file = lyrics_files[0]
        
        # Extract text from file
        file_content = ""
        if lyrics_file.suffix.lower() == '.pdf':
            try:
                # Try PyPDF2 first
                import PyPDF2
                with open(lyrics_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        file_content += page.extract_text() + "\n"
                print(f"DEBUG: Successfully read PDF with PyPDF2, content length: {len(file_content)}")
            except Exception as e:
                print(f"DEBUG: PyPDF2 failed: {e}")
                try:
                    # Fallback to PyMuPDF if PyPDF2 fails
                    import fitz  # PyMuPDF
                    doc = fitz.open(lyrics_file)
                    for page in doc:
                        file_content += page.get_text() + "\n"
                    doc.close()
                    print(f"DEBUG: Successfully read PDF with PyMuPDF, content length: {len(file_content)}")
                except Exception as e2:
                    print(f"DEBUG: PyMuPDF also failed: {e2}")
                    return f"Error reading PDF file with both PyPDF2 and PyMuPDF: {str(e)} and {str(e2)}"
        elif lyrics_file.suffix.lower() == '.txt':
            try:
                with open(lyrics_file, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                print(f"DEBUG: Successfully read text file, content length: {len(file_content)}")
            except Exception as e:
                return f"Error reading text file: {str(e)}"
        
        # Clean up the extracted text - handle extra spaces and line breaks
        import re
        file_content = re.sub(r'\s+', ' ', file_content)  # Replace multiple spaces with single space
        file_content = re.sub(r'\n\s*\n', '\n', file_content)  # Remove empty lines
        file_content = file_content.strip()
        
        print(f"DEBUG: File content length: {len(file_content)}")
        print(f"DEBUG: File content preview: {file_content[:200]}...")
        
        if not file_content.strip():
            return "File appears to be empty or unreadable."
        
        # Analyze lyrics content
        lines = [line.strip() for line in file_content.split('\n') if line.strip()]
        word_count = len(file_content.split())
        line_count = len(lines)
        
        print(f"DEBUG: Found {line_count} non-empty lines")
        print(f"DEBUG: First few lines: {lines[:5]}")
        
        # Look for song structure patterns (case insensitive)
        content_lower = file_content.lower()
        has_verse = 'verse' in content_lower
        has_chorus = 'chorus' in content_lower
        has_bridge = 'bridge' in content_lower
        has_drop = 'drop' in content_lower
        
        print(f"DEBUG: Structure analysis - Verse: {has_verse}, Chorus: {has_chorus}, Bridge: {has_bridge}, Drop: {has_drop}")
        
        
        # Store analysis data globally for other tools to use
        global global_lyrics_analysis
        global_lyrics_analysis = {
            'filename': lyrics_file.name,
            'word_count': word_count,
            'line_count': line_count,
            'has_verse': has_verse,
            'has_chorus': has_chorus,
            'has_bridge': has_bridge,
            'has_drop': has_drop,
            'content': file_content,
            'structure': {
                'verses': has_verse,
                'choruses': has_chorus,
                'bridges': has_bridge,
                'drops': has_drop
            }
        }
        
        # Return detailed analysis for the write_lyrics_tool to use
        result = f"""
Lyrics Analysis Complete:

File: {lyrics_file.name}
Structure: {'Has' if has_verse else 'No'} verses, {'Has' if has_chorus else 'No'} choruses, {'Has' if has_bridge else 'No'} bridges
Word Count: {word_count}
Lines: {line_count}


The lyrics have been analyzed and are ready for the write_lyrics_tool to generate new content based on this analysis.
        """
        print(f"DEBUG: process_lyrics_file returning: {result}")
        return result
        
    except Exception as e:
        return f"Error analyzing lyrics file: {str(e)}"

@tool
def rag_knowledge_base_DAW_tool(question: str) -> str:
    """
    Query the music production knowledge base for information about DAWs, music theory, 
    production techniques, and other music-related topics. This tool searches through 
    textbooks and manuals to provide accurate, educational information.
    """
    try:
        print(f"RAG tool called with question: {question}")
        
        # Use the RAG graph to get response with timeout
        import concurrent.futures
        
        def run_rag():
            try:
                print("Invoking RAG graph...")
                result = graph.invoke({"question": question})
                print(f"RAG graph result: {result}")
                print(f"RAG response content: {result.get('response', 'No response')}")
                return result
            except Exception as e:
                print(f"Error in RAG graph: {e}")
                return {"error": str(e)}
        
        # Run with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_rag)
            try:
                result = future.result(timeout=300)  # Increased timeout to 60 seconds
                if result and "response" in result:
                    response_text = result["response"].strip()
                    # Limit response length
                    if len(response_text) > 500:
                        response_text = response_text[:497] + "..."
                    return response_text
                elif result and "error" in result:
                    return f"Knowledge base error: {result['error']}"
                else:
                    return "The knowledge base returned an empty response. Please try rephrasing your question."
            except concurrent.futures.TimeoutError:
                return "The knowledge base query timed out. Please try rephrasing your question or ask about a different topic."
            except Exception as e:
                return f"Error querying knowledge base: {str(e)}"
                
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


@tool
def process_midi_file(question: str, filename: str = None) -> str:
    """
    Expert tool that analyzes MIDI files and provides musical insights.
    Use this when users ask about MIDI files, tempo, instruments, or musical elements.
    """
    try:
        # Find the uploaded MIDI file
        upload_dir = Path("uploads")
        
        print(f"DEBUG: process_midi_file called with filename: {filename}")
        print(f"DEBUG: process_midi_file called with question: {question}")
        print(f"DEBUG: process_midi_file called with all args: filename={filename}, question={question}")
        print(f"DEBUG: Current working directory: {Path.cwd()}")
        print(f"DEBUG: Uploads directory exists: {upload_dir.exists()}")
        if filename:
            # Use the specific filename if provided
            midi_file = upload_dir / filename
            print(f"DEBUG: Looking for file: {midi_file}")
            if not midi_file.exists():
                return f"MIDI file '{filename}' not found in uploads directory."
            print(f"DEBUG: Found file: {midi_file}")
        else:
            # Use the most recently uploaded file
            midi_files = list(upload_dir.glob("*.mid")) + list(upload_dir.glob("*.midi"))
            if not midi_files:
                return "No MIDI files found in uploads directory. Please upload a MIDI file first."
            
            # Use the most recently modified file
            midi_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            midi_file = midi_files[0]
            print(f"DEBUG: Using most recently uploaded file: {midi_file.name}")
        
        # Import and use MIDIProcessor
        from midi_processor import MIDIProcessor
        
        print(f"DEBUG: Starting MIDI analysis for file: {midi_file}")
        processor = MIDIProcessor(str(midi_file))
        analysis = processor.process()
        print(f"DEBUG: MIDI analysis completed for file: {midi_file}")
        
        if 'error' in analysis:
            return f"Error analyzing MIDI file: {analysis['error']}"
        
        # Extract key information
        basic_info = analysis.get('basic_info', {})
        tempo_info = analysis.get('tempo', {})
        notes_info = analysis.get('notes', {})
        insights = analysis.get('insights', [])
        
        # Determine key from most common notes
        key = "Unknown"
        if notes_info.get('most_common_notes'):
            most_common = notes_info['most_common_notes'][0][0] if notes_info['most_common_notes'] else "C4"
            # Simple key detection based on most common note
            note_name = most_common[:-1] if most_common[-1].isdigit() else most_common
            key = f"{note_name} Major" if note_name in ['C', 'G', 'D', 'A', 'E', 'B', 'F#'] else f"{note_name} Minor"
        
        # Store analysis data globally for other tools to use
        global global_midi_analysis
        global_midi_analysis = {
            'basic_info': basic_info,
            'tempo_info': tempo_info,
            'notes_info': notes_info,
            'key': key,
            'time_signature': analysis.get('time_signatures', {}).get('primary_time_signature', '4/4'),
            'insights': insights,
            'filename': basic_info.get('filename', 'Unknown')
        }
        print(f"DEBUG: Set global_midi_analysis: {global_midi_analysis}")
        
        # Return simple confirmation without formatting
        return f"MIDI file analyzed successfully. File: {basic_info.get('filename', 'Unknown')}. Key: {key}. Tempo: {tempo_info.get('primary_tempo', 'Unknown')} BPM. Duration: {basic_info.get('length_seconds', 0):.1f} seconds."
        
    except Exception as e:
        return f"Error processing MIDI file: {str(e)}"

# Initialize external tools
try:
    tavily_search_tool = TavilySearchResults(max_results=5)
except Exception as e:
    print(f"Warning: Tavily search tool not available: {e}")
    tavily_search_tool = None

# Tool belt
tool_belt = [
    write_lyrics_tool, 
    suggest_instruments_or_samples_tool, 
    arrangement_advice_tool, 
    process_midi_file,
    process_lyrics_file,
    rag_knowledge_base_DAW_tool,
]

# Add Tavily search tool if available
if tavily_search_tool:
    tool_belt.append(tavily_search_tool)

# Tool information for frontend
available_tools = [
    ToolInfo(name="write_lyrics_tool", description="Generate lyrics based on prompts, style, mood, or artist references"),
    ToolInfo(name="suggest_instruments_or_samples_tool", description="Get instrument and synth preset recommendations"),
    ToolInfo(name="arrangement_advice_tool", description="Get song arrangement and structure suggestions"),
    ToolInfo(name="process_midi_file", description="Analyze MIDI files and provide musical insights"),
    ToolInfo(name="process_lyrics_file", description="Analyze uploaded lyrics files and pass the information to the write_lyrics_tool"),
    ToolInfo(name="rag_knowledge_base_DAW_tool", description="Search the music production knowledge base for information about DAWs, music theory, production techniques, and other music-related topics"),
]

# Add Tavily search tool info if available
if tavily_search_tool:
    available_tools.append(ToolInfo(name="tavily_search_tool", description="Search the web for music production information"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
model = model.bind_tools(tool_belt)

# Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[Document]

# Model call function
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {
        "messages": [response],
        "context": state.get("context", [])
    }

# Tool node
tool_node = ToolNode(tool_belt)

# Build graph
uncompiled_graph = StateGraph(AgentState)
uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)
uncompiled_graph.set_entry_point("agent")

def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "action"
    return END

uncompiled_graph.add_conditional_edges("agent", should_continue)
uncompiled_graph.add_edge("action", "agent")
compiled_graph = uncompiled_graph.compile(checkpointer=None, interrupt_before=["action"])

@app.get("/")
async def root():
    return {"message": "Music Mate AI API"}

@app.get("/tools")
async def get_tools():
    """Get available tools for the frontend"""
    return {"tools": available_tools}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return the file path"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save the file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"filename": file.filename, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Extract filename from chat history or request
        uploaded_filename = None
        
        # First try to get filename from request
        if request.file_path:
            uploaded_filename = Path(request.file_path).name
            print(f"DEBUG: Got filename from request.file_path: {uploaded_filename}")
        elif request.file_upload:
            uploaded_filename = request.file_upload
            print(f"DEBUG: Got filename from request.file_upload: {uploaded_filename}")
        else:
            # Try to extract from chat history
            for msg in request.history:
                if msg.role == "assistant" and "File uploaded successfully" in msg.content:
                    # Extract filename from the upload confirmation message
                    content = msg.content
                    print(f"DEBUG: Looking for filename in: {content}")
                    
                    # Try different patterns - improved for better filename extraction
                    import re
                    patterns = [
                        r'File uploaded successfully: (.*?\.mid)',  # filename.mid
                        r'File uploaded successfully: (.*?\.midi)',  # filename.midi
                        r'File uploaded successfully: (.*?\.pdf)',   # filename.pdf
                        r'File uploaded successfully: (.*?\.txt)',   # filename.txt
                        r'File uploaded successfully: (.*?\.mid)\s',  # filename.mid with space
                        r'File uploaded successfully: (.*?\.midi)\s',  # filename.midi with space
                        r'File uploaded successfully: (.*?\.pdf)\s',  # filename.pdf with space
                        r'File uploaded successfully: (.*?\.txt)\s',  # filename.txt with space
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, content)
                        if match:
                            uploaded_filename = match.group(1)
                            print(f"DEBUG: Extracted filename with pattern '{pattern}': {uploaded_filename}")
                            break
                    
                    if not uploaded_filename:
                        print(f"DEBUG: No filename match found with any pattern")
                        print(f"DEBUG: Content to search: '{content}'")
                        
                        # Try more aggressive filename extraction
                        # Look for any text that contains a file extension
                        all_text = content.replace('✅', '').replace('File uploaded successfully:', '')
                        print(f"DEBUG: Cleaned text: '{all_text}'")
                        
                        # Try to find any filename with extension
                        filename_patterns = [
                            r'([\w\s\-\(\)\'\"]+\.(mid|midi|pdf|txt))',  # Very flexible pattern
                            r'([^\\s]+\.(mid|midi|pdf|txt))',  # Any non-whitespace + extension
                        ]
                        
                        for pattern in filename_patterns:
                            match = re.search(pattern, all_text)
                            if match:
                                uploaded_filename = match.group(1).strip()
                                print(f"DEBUG: Extracted filename with aggressive pattern '{pattern}': {uploaded_filename}")
                                break
                    break
        
        print(f"DEBUG: Final uploaded_filename: {uploaded_filename}")
        
        messages = []
        
        # Add system message to guide the AI
        system_message = """You are a music production AI assistant with access to specialized tools. 

MANDATORY LYRICS WORKFLOW:
When a lyrics file is uploaded and user asks for lyrics help:
1. FIRST: Call process_lyrics_file to analyze the file
2. SECOND: Call write_lyrics_tool to generate new lyrics
3. BOTH tools MUST be called - this is not optional

MANDATORY INSTRUMENT/SAMPLE WORKFLOW:
When a MIDI file is uploaded and user asks about instruments, samples, or making the song more danceable/poppy:
1. FIRST: Call process_midi_file to analyze the file (if not already done)
2. SECOND: Call suggest_instruments_or_samples_tool to provide specific recommendations
3. BOTH tools MUST be called - this is not optional

MANDATORY RAG WORKFLOW:
When users ask about DAW functionality, music theory, production techniques, or any music-related educational topics:
1. ALWAYS call rag_knowledge_base_DAW_tool to search the knowledge base
2. This tool searches through textbooks and manuals for accurate, educational information
3. Use this tool for questions about Logic Pro, Ableton, music theory, mixing, mastering, etc.

For arrangement advice:
1. If MIDI file is uploaded, call process_midi_file first
2. Then call arrangement_advice_tool with the analysis data

IMPORTANT: 
- When users ask about instruments, samples, or making songs more danceable/poppy, you MUST call suggest_instruments_or_samples_tool
- When users ask about DAW functionality, music theory, or production techniques, you MUST call rag_knowledge_base_DAW_tool
- Do not provide generic advice - use the appropriate tools to get specific, contextual information

FAILURE TO CALL REQUIRED TOOLS IS NOT ACCEPTABLE. You must call the appropriate tools when users ask relevant questions."""
        
        messages.append(SystemMessage(content=system_message))
        
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        messages.append(HumanMessage(content=request.message))
        
        # Get AI response
        response = model.invoke(messages)
        
        # Check if AI wants to use a tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Add the AI response with tool calls to messages
            messages.append(response)
            
            # Handle all tool calls
            tools_used = []
            for tool_call in response.tool_calls:
                # Debug logging
                print(f"DEBUG: Processing tool call: {tool_call}")
                print(f"DEBUG: Tool call type: {type(tool_call)}")
                
                # Extract tool call details
                if hasattr(tool_call, 'name'):
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                    tool_call_id = tool_call.id
                elif isinstance(tool_call, dict):
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    tool_call_id = tool_call.get('id')
                else:
                    # Fallback for unknown structure
                    tool_name = getattr(tool_call, 'name', 'unknown')
                    tool_args = getattr(tool_call, 'args', {})
                    tool_call_id = getattr(tool_call, 'id', f"call_{uuid.uuid4().hex[:12]}")
                
                print(f"DEBUG: Tool name: {tool_name}")
                print(f"DEBUG: Tool args: {tool_args}")
                print(f"DEBUG: Tool call ID: {tool_call_id}")
                
                # Find the tool
                tool_to_use = None
                for tool in tool_belt:
                    if hasattr(tool, 'name') and tool.name == tool_name:
                        tool_to_use = tool
                        break
                    elif hasattr(tool, '__name__') and tool.__name__ == tool_name:
                        tool_to_use = tool
                        break
                
                if tool_to_use:
                    # Execute the tool
                    # Handle different argument formats
                    if isinstance(tool_args, dict):
                        # For LangChain tools, we need to pass the input as a string or dict
                        if len(tool_args) == 1 and 'question' in tool_args:
                            # Add filename to tool arguments if available
                            if uploaded_filename and tool_name in ['process_midi_file', 'process_lyrics_file']:
                                tool_args['filename'] = uploaded_filename
                                print(f"DEBUG: Added filename '{uploaded_filename}' to tool '{tool_name}'")
                            # Add question to instrument and sample pack tools for contextual suggestions
                            elif tool_name in ['suggest_instruments_or_samples_tool']:
                                tool_args['question'] = tool_args['question']
                                print(f"DEBUG: Added question to tool '{tool_name}' for contextual suggestions")
                            else:
                                print(f"DEBUG: No filename added to tool '{tool_name}', uploaded_filename: {uploaded_filename}")
                            tool_result = tool_to_use.invoke(tool_args)
                        elif len(tool_args) == 2 and 'question' in tool_args and 'filename' in tool_args:
                            # Tool already has both question and filename, invoke directly
                            print(f"DEBUG: Tool '{tool_name}' has both question and filename, invoking directly")
                            tool_result = tool_to_use.invoke(tool_args)
                        else:
                            tool_result = tool_to_use.invoke(str(tool_args))
                    else:
                        # If tool_args is a string, create a dict with the question and filename
                        if uploaded_filename and tool_name in ['process_midi_file', 'process_lyrics_file']:
                            tool_args_dict = {'question': tool_args, 'filename': uploaded_filename}
                            print(f"DEBUG: Created dict with filename '{uploaded_filename}' for tool '{tool_name}'")
                            tool_result = tool_to_use.invoke(tool_args_dict)
                        elif tool_name in ['suggest_instruments_or_samples_tool']:
                            # For instrument and sample pack tools, pass the question for contextual suggestions
                            tool_args_dict = {'question': tool_args}
                            print(f"DEBUG: Created dict with question for tool '{tool_name}'")
                            tool_result = tool_to_use.invoke(tool_args_dict)
                        else:
                            # Even if no uploaded_filename, try to create a dict for the tool
                            if tool_name in ['process_midi_file', 'process_lyrics_file']:
                                tool_args_dict = {'question': tool_args}
                                print(f"DEBUG: Created dict without filename for tool '{tool_name}'")
                                tool_result = tool_to_use.invoke(tool_args_dict)
                            else:
                                tool_result = tool_to_use.invoke(tool_args)
                    
                    # Check if tool result is empty or None
                    if not tool_result or str(tool_result).strip() == "":
                        print(f"DEBUG: Tool '{tool_name}' returned empty result")
                        tool_result = f"Tool '{tool_name}' completed but returned no content. Please try rephrasing your request."
                    
                    tools_used.append(tool_name)
                    
                    # Create a proper tool message
                    from langchain_core.messages import ToolMessage
                    
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    )
                    messages.append(tool_message)
                else:
                    # If tool not found, create an error message
                    from langchain_core.messages import ToolMessage
                    
                    error_message = ToolMessage(
                        content="Tool not found or not available.",
                        tool_call_id=tool_call_id
                    )
                    messages.append(error_message)
            
            # Get final response after all tools are executed
            final_response = model.invoke(messages)
            cleaned_response = clean_markdown(final_response.content)
            
            # If final response is empty but tools were used, use the tool results directly
            if not cleaned_response.strip() and tools_used:
                # Find the most recent tool result from the messages
                tool_result = None
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content and 'tool' in str(type(msg)).lower():
                        tool_result = msg.content
                        break
                
                if tool_result and tool_result.strip():
                    cleaned_response = tool_result.strip()
                    print(f"DEBUG: Using tool result as response: {cleaned_response[:100]}...")
                else:
                    # Fallback responses for specific tools
                    if 'rag_knowledge_base_DAW_tool' in tools_used:
                        cleaned_response = "I've searched the knowledge base for you. The information should appear above. If you're not seeing it, please try rephrasing your question."
                    elif 'suggest_instruments_or_samples_tool' in tools_used:
                        cleaned_response = "I've analyzed your MIDI file and provided instrument/sample recommendations. The suggestions should appear above. If you're not seeing them, please try asking again."
                    elif 'process_lyrics_file' in tools_used and 'write_lyrics_tool' not in tools_used:
                        # Only analysis tool was called, automatically call write_lyrics_tool
                        print("DEBUG: Only process_lyrics_file was called, automatically calling write_lyrics_tool")
                        
                        # Find the write_lyrics_tool
                        write_lyrics_tool = None
                        for tool in tool_belt:
                            if hasattr(tool, 'name') and tool.name == 'write_lyrics_tool':
                                write_lyrics_tool = tool
                                break
                        
                        if write_lyrics_tool:
                            # Get the user's original request from the last message
                            user_request = request.message
                            try:
                                # Call write_lyrics_tool with the user's request
                                lyrics_result = write_lyrics_tool.invoke({'prompt': user_request})
                                tools_used.append('write_lyrics_tool')
                                cleaned_response = lyrics_result  # Use the actual lyrics result directly
                                print(f"DEBUG: Auto-called write_lyrics_tool, result: {lyrics_result}")
                                print(f"DEBUG: Final cleaned_response: {cleaned_response}")
                            except Exception as e:
                                print(f"DEBUG: Error auto-calling write_lyrics_tool: {e}")
                                cleaned_response = "I've analyzed your lyrics file. Now I need to generate the bridge lyrics for you. Let me do that now."
                        else:
                            cleaned_response = "I've analyzed your lyrics file. Now I need to generate the bridge lyrics for you. Let me do that now."
                    elif 'process_lyrics_file' in tools_used and 'write_lyrics_tool' in tools_used:
                        # Both tools were called but no response, provide fallback
                        cleaned_response = "I've analyzed your lyrics and generated new content. The lyrics should appear above. If you're not seeing them, please try asking again."
                    else:
                        # Other tools were used but no response
                        cleaned_response = "I've processed your request. If you're not seeing the results, please try rephrasing your question."
            
            print(f"DEBUG: Final response content: {final_response.content}")
            print(f"DEBUG: Cleaned response: {cleaned_response}")
            print(f"DEBUG: Tools used: {tools_used}")
            
            return ChatResponse(
                response=cleaned_response,
                tools_used=tools_used
            )
        else:
            # No tool calls, just return the response
            cleaned_response = clean_markdown(response.content)
            return ChatResponse(
                response=cleaned_response,
                tools_used=[]
            )
        
    except Exception as e:
        return ChatResponse(
            response="I encountered an error while processing your request. Please try again.",
            tools_used=[],
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
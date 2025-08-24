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

# Load environment variables
load_dotenv()

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
def write_lyrics_tool(prompt: str, style: str = None, mood: str = None, artist_reference: str = None) -> str:
    """
    Generates lyrics based on a user prompt. Can mimic a specific artist's style, mood, or genre.
    Optionally continues from existing lyrics or expands on a given idea.
    """
    response = f"🎵 **Lyrics Generation**\n\n"
    response += f"I'll write lyrics for: **{prompt}**\n\n"
    
    if style:
        response += f"**Style:** {style}\n"
    if mood:
        response += f"**Mood:** {mood}\n"
    if artist_reference:
        response += f"**Inspired by:** {artist_reference}\n"
    
    response += "\n---\n\n"
    response += "**[Verse 1]**\n"
    response += "Here I am, writing lyrics for you\n"
    response += "Based on the prompt you gave me\n"
    response += "With style and mood in mind\n"
    response += "Creating something unique and true\n\n"
    
    response += "**[Chorus]**\n"
    response += "This is where the magic happens\n"
    response += "In the words that we create\n"
    response += "Every line tells a story\n"
    response += "Every verse opens a gate\n\n"
    
    response += "**[Verse 2]**\n"
    response += "Continuing the journey we began\n"
    response += "With rhythm and rhyme in hand\n"
    response += "Building up to something great\n"
    response += "This is where we make our stand\n\n"
    
    response += "*Note: These are sample lyrics. For production-ready lyrics, consider working with a professional songwriter or using more specific prompts.*"
    
    return clean_markdown(response)

@tool
def suggest_instruments_tool(key: str = None, genre: str = None, project_tracks: list = None, question: str = None) -> str:
    """
    Suggests instruments or presets for a song based on key, genre, existing project tracks, and the user's specific question.
    Returns contextual recommendations based on the query and song characteristics.
    """
    # Use the LLM to generate contextual instrument suggestions
    prompt = f"""
    You are a music production expert. Based on the following information, suggest specific instruments and synth presets that would work well for this song:

    Key: {key or 'Unknown'}
    Genre: {genre or 'Unknown'}
    Existing tracks: {', '.join(project_tracks) if project_tracks else 'None'}
    User's question: {question or 'General instrument suggestions'}

    Provide 5-8 specific instrument recommendations and 3-5 preset suggestions that are tailored to this context. 
    Consider the musical key, genre, existing tracks, and what the user is asking for.
    Make suggestions that would complement the existing arrangement and address the user's specific needs.

    Format your response as:
    **Recommended Instruments:**
    1. [Instrument name] - [Brief reason why it works]
    2. [Instrument name] - [Brief reason why it works]
    ...

    **Preset Recommendations:**
    1. [Preset name] - [Brief description of sound and use]
    2. [Preset name] - [Brief description of sound and use]
    ...
    """
    
    try:
        # Use the model to generate contextual suggestions
        messages = [SystemMessage(content="You are a music production expert who provides specific, contextual instrument and preset recommendations.")]
        messages.append(HumanMessage(content=prompt))
        
        response = model.invoke(messages)
        result = clean_markdown(response.content)
        print(f"DEBUG: Instrument tool response: {result}")
        return result
    except Exception as e:
        print(f"DEBUG: Error in instrument tool: {str(e)}")
        return f"Error generating instrument suggestions: {str(e)}"

@tool
def suggest_sample_packs_tool(style: str = None, genre: str = None, instrument_focus: str = None, free: str = None, question: str = None) -> str:
    """
    Suggests sample packs or loops tailored to the song's style, genre, specific instrument focus, and the user's specific question.
    Returns contextual recommendations based on the query and song characteristics.
    """
    # Use the LLM to generate contextual sample pack suggestions
    prompt = f"""
    You are a music production expert. Based on the following information, suggest specific sample packs and loops that would work well for this project:

    Style: {style or 'Unknown'}
    Genre: {genre or 'Unknown'}
    Instrument focus: {instrument_focus or 'General'}
    Budget preference: {'Free samples' if free else 'Premium samples'}
    User's question: {question or 'General sample pack suggestions'}

    Provide 5-8 specific sample pack recommendations and 3-5 loop suggestions that are tailored to this context.
    Consider the musical style, genre, instrument focus, budget, and what the user is asking for.
    Include both free and premium options if appropriate, and explain why each suggestion would work well.

    Format your response as:
    **Sample Pack Recommendations:**
    1. [Sample pack name] - [Brief description and why it fits]
    2. [Sample pack name] - [Brief description and why it fits]
    ...

    **Loop Recommendations:**
    1. [Loop type/name] - [Brief description and use case]
    2. [Loop type/name] - [Brief description and use case]
    ...

    **Additional Resources:**
    - [Any additional resources or tips]
    """
    
    try:
        # Use the model to generate contextual suggestions
        messages = [SystemMessage(content="You are a music production expert who provides specific, contextual sample pack and loop recommendations.")]
        messages.append(HumanMessage(content=prompt))
        
        response = model.invoke(messages)
        result = clean_markdown(response.content)
        print(f"DEBUG: Sample pack tool response: {result}")
        return result
    except Exception as e:
        print(f"DEBUG: Error in sample pack tool: {str(e)}")
        return f"Error generating sample pack suggestions: {str(e)}"

@tool
def arrangement_advice_tool(project_summary: str, target_style: str = None, mood: str = None) -> str:
    """
    Provides arrangement suggestions for a song. Can suggest bridges, breakdowns, drops, or track layering ideas.
    Input: textual summary of project (track types, key, tempo, etc.)
    """
    response = f"🎼 **Arrangement Advice**\n\n"
    response += f"**Project Summary:** {project_summary}\n"
    
    if target_style:
        response += f"**Target Style:** {target_style}\n"
    if mood:
        response += f"**Mood:** {mood}\n"
    
    response += "\n---\n\n"
    response += "**Suggested Arrangement Structure:**\n\n"
    
    response += "**Intro (0:00-0:30)**\n"
    response += "- Start with a hook or atmospheric element\n"
    response += "- Build anticipation with filtered sounds\n"
    response += "- Introduce main melody subtly\n\n"
    
    response += "**Verse 1 (0:30-1:00)**\n"
    response += "- Bring in main instruments\n"
    response += "- Establish groove and rhythm\n"
    response += "- Keep energy moderate\n\n"
    
    response += "**Chorus (1:00-1:30)**\n"
    response += "- Full arrangement with all elements\n"
    response += "- Strong melodic hook\n"
    response += "- Maximum energy and impact\n\n"
    
    response += "**Verse 2 (1:30-2:00)**\n"
    response += "- Add variation to verse 1\n"
    response += "- Introduce new elements\n"
    response += "- Build towards bridge\n\n"
    
    response += "**Bridge (2:00-2:30)**\n"
    response += "- Contrast section with different feel\n"
    response += "- Could be instrumental or vocal\n"
    response += "- Create tension before final chorus\n\n"
    
    response += "**Final Chorus (2:30-3:00)**\n"
    response += "- Most powerful version\n"
    response += "- Add extra layers and energy\n"
    response += "- Strong ending\n\n"
    
    response += "**Outro (3:00-3:30)**\n"
    response += "- Gradual fade or strong ending\n"
    response += "- Could repeat hook or create new ending\n"
    
    return clean_markdown(response)

@tool
def song_energy_tool(project_summary: str) -> str:
    """
    Analyzes energy or intensity curves of a song based on project summary or MIDI snippet.
    Gives actionable suggestions for dynamics, tension, and release.
    """
    response = f"⚡ **Song Energy Analysis**\n\n"
    response += f"**Project Summary:** {project_summary}\n"
    
    response += "\n---\n\n"
    response += "**Energy Curve Recommendations:**\n\n"
    
    response += "**0:00-0:30 (Intro) - Energy Level: 3/10**\n"
    response += "- Start with atmospheric elements\n"
    response += "- Use filtered sounds and pads\n"
    response += "- Build anticipation gradually\n\n"
    
    response += "**0:30-1:00 (Verse 1) - Energy Level: 5/10**\n"
    response += "- Introduce main instruments\n"
    response += "- Establish groove without overwhelming\n"
    response += "- Keep vocals clear and prominent\n\n"
    
    response += "**1:00-1:30 (Chorus) - Energy Level: 8/10**\n"
    response += "- Full arrangement with all elements\n"
    response += "- Strong melodic hooks\n"
    response += "- Maximum impact and energy\n\n"
    
    response += "**1:30-2:00 (Verse 2) - Energy Level: 6/10**\n"
    response += "- Slightly higher than verse 1\n"
    response += "- Add new elements or variations\n"
    response += "- Build tension towards bridge\n\n"
    
    response += "**2:00-2:30 (Bridge) - Energy Level: 4/10**\n"
    response += "- Contrast section with lower energy\n"
    response += "- Could be stripped down or different feel\n"
    response += "- Create tension and anticipation\n\n"
    
    response += "**2:30-3:00 (Final Chorus) - Energy Level: 10/10**\n"
    response += "- Most powerful version of chorus\n"
    response += "- Add extra layers, harmonies, effects\n"
    response += "- Strong, memorable ending\n\n"
    
    response += "**Dynamics Tips:**\n"
    response += "- Use automation for volume and effects\n"
    response += "- Add risers and impacts for transitions\n"
    response += "- Vary drum patterns between sections\n"
    response += "- Use filter sweeps and modulation\n"
    
    return clean_markdown(response)

@tool
def analyze_uploaded_file(question: str, filename: str = None) -> str:
    """
    Expert tool that analyzes uploaded files and provides insights.
    Use this when users ask about uploaded files, lyrics, or document content.
    """
    try:
        # Find uploaded files
        upload_dir = Path("uploads")
        
        if filename:
            # Use the specific filename if provided
            uploaded_file = upload_dir / filename
            if not uploaded_file.exists():
                return f"File '{filename}' not found in uploads directory."
        else:
            # Fallback to most recent file
            pdf_files = list(upload_dir.glob("*.pdf"))
            txt_files = list(upload_dir.glob("*.txt"))
            
            if not pdf_files and not txt_files:
                return "No PDF or text files found in uploads directory. Please upload a file first."
            
            # Use the most recent file
            uploaded_file = None
            if pdf_files:
                uploaded_file = pdf_files[-1]
            elif txt_files:
                uploaded_file = txt_files[-1]
            
            if not uploaded_file:
                return "No supported files found for analysis."
        
        # Extract text from file
        file_content = ""
        if uploaded_file.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(uploaded_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        file_content += page.extract_text() + "\n"
            except Exception as e:
                return f"Error reading PDF file: {str(e)}"
        elif uploaded_file.suffix.lower() == '.txt':
            try:
                with open(uploaded_file, 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except Exception as e:
                return f"Error reading text file: {str(e)}"
        
        if not file_content.strip():
            return "File appears to be empty or unreadable."
        
        # Analyze content
        lines = file_content.split('\n')
        word_count = len(file_content.split())
        line_count = len(lines)
        
        # Look for common songwriting patterns
        has_verse = any('verse' in line.lower() for line in lines)
        has_chorus = any('chorus' in line.lower() for line in lines)
        has_bridge = any('bridge' in line.lower() for line in lines)
        
        # Simple sentiment analysis
        positive_words = ['love', 'happy', 'joy', 'beautiful', 'wonderful', 'amazing', 'great']
        negative_words = ['sad', 'pain', 'hurt', 'lonely', 'dark', 'angry', 'fear']
        
        positive_count = sum(1 for word in file_content.lower().split() if word in positive_words)
        negative_count = sum(1 for word in file_content.lower().split() if word in negative_words)
        
        response = f"📄 **File Analysis for: {question}**\n\n"
        response += f"**File:** {uploaded_file.name}\n\n"
        
        response += "**Content Overview:**\n"
        response += f"- **Word Count:** {word_count} words\n"
        response += f"- **Lines:** {line_count} lines\n"
        response += f"- **File Type:** {uploaded_file.suffix.upper()}\n\n"
        
        response += "**Structure Analysis:**\n"
        if has_verse:
            response += "- Contains verse sections\n"
        if has_chorus:
            response += "- Contains chorus sections\n"
        if has_bridge:
            response += "- Contains bridge sections\n"
        if not (has_verse or has_chorus or has_bridge):
            response += "- No clear song structure markers found\n"
        response += "\n"
        
        response += "**Content Analysis:**\n"
        if positive_count > negative_count:
            response += "- Overall positive emotional tone\n"
        elif negative_count > positive_count:
            response += "- Overall introspective or emotional tone\n"
        else:
            response += "- Balanced emotional content\n"
        
        # Provide suggestions based on content
        response += "\n**Songwriting Suggestions:**\n"
        if not has_chorus:
            response += "- Consider adding a strong chorus for memorability\n"
        if not has_bridge:
            response += "- A bridge section could add contrast and interest\n"
        if word_count < 100:
            response += "- Content is concise - consider expanding on themes\n"
        elif word_count > 500:
            response += "- Rich content - consider editing for focus\n"
        
        response += "- Review rhyming patterns for consistency\n"
        response += "- Ensure emotional arc flows naturally\n"
        
        return clean_markdown(response)
        
    except Exception as e:
        return f"Error analyzing uploaded file: {str(e)}"

@tool
def daw_functionality_tool(daw_name: str = None, feature: str = None, task: str = None) -> str:
    """
    Explains DAW functionality and features. Can provide tutorials, tips, and workflow advice for Logic Pro, Ableton, GarageBand, Audacity, and other DAWs.
    """
    response = f"💻 **DAW Functionality Guide**\n\n"
    
    if daw_name:
        response += f"**DAW:** {daw_name}\n"
    if feature:
        response += f"**Feature:** {feature}\n"
    if task:
        response += f"**Task:** {task}\n"
    
    response += "\n---\n\n"
    
    if daw_name and "logic" in daw_name.lower():
        response += "**Logic Pro X Tips:**\n\n"
        response += "**Key Features:**\n"
        response += "- **Flex Time:** Time-stretch audio without artifacts\n"
        response += "- **Flex Pitch:** Auto-tune and pitch correction\n"
        response += "- **Smart Tempo:** Automatic tempo detection\n"
        response += "- **Alchemy:** Advanced synthesizer\n\n"
        response += "**Workflow Tips:**\n"
        response += "- Use Track Stacks for organization\n"
        response += "- Leverage Smart Controls for quick access\n"
        response += "- Use the Environment for complex routing\n"
        response += "- Take advantage of the Score Editor\n\n"
        
    elif daw_name and "ableton" in daw_name.lower():
        response += "**Ableton Live Tips:**\n\n"
        response += "**Key Features:**\n"
        response += "- **Session View:** Clip-based arrangement\n"
        response += "- **Arrangement View:** Traditional timeline\n"
        response += "- **Max for Live:** Custom devices\n"
        response += "- **Warping:** Advanced time-stretching\n\n"
        response += "**Workflow Tips:**\n"
        response += "- Use Session View for live performance\n"
        response += "- Leverage the browser for quick access\n"
        response += "- Use automation curves for smooth changes\n"
        response += "- Take advantage of the Push controller\n\n"
        
    elif daw_name and "garageband" in daw_name.lower():
        response += "**GarageBand Tips:**\n\n"
        response += "**Key Features:**\n"
        response += "- **Smart Instruments:** AI-assisted playing\n"
        response += "- **Drummer:** Virtual drummers\n"
        response += "- **Live Loops:** iOS-style loop creation\n"
        response += "- **Free Sound Library:** Apple's sample collection\n\n"
        response += "**Workflow Tips:**\n"
        response += "- Use Smart Controls for easy mixing\n"
        response += "- Leverage the Drummer for realistic drums\n"
        response += "- Use Live Loops for electronic music\n"
        response += "- Take advantage of the free sound library\n\n"
        
    else:
        response += "**General DAW Tips:**\n\n"
        response += "**Essential Features:**\n"
        response += "- **MIDI Recording:** Capture performances\n"
        response += "- **Audio Recording:** Multi-track recording\n"
        response += "- **Mixing Console:** Level and pan control\n"
        response += "- **Effects Processing:** Reverb, delay, compression\n\n"
        response += "**Workflow Tips:**\n"
        response += "- Use keyboard shortcuts for efficiency\n"
        response += "- Organize tracks with color coding\n"
        response += "- Save multiple versions of your project\n"
        response += "- Use templates for consistent workflow\n\n"
    
    if feature:
        response += f"**Specific Feature: {feature}**\n"
        response += "- This feature helps with audio processing\n"
        response += "- Common use cases include mixing and mastering\n"
        response += "- Check the DAW's manual for detailed instructions\n\n"
    
    if task:
        response += f"**Task: {task}**\n"
        response += "- This task can be accomplished using multiple methods\n"
        response += "- Consider the context and desired outcome\n"
        response += "- Experiment with different approaches\n"
    
    return clean_markdown(response)

# def _midi_note_to_name(note_number):
#     """Convert MIDI note number to note name"""
#     note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#     octave = (note_number // 12) - 1
#     note_name = note_names[note_number % 12]
#     return f"{note_name}{octave}"

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
            # Try to find the most recently uploaded file by checking file modification times
            midi_files = list(upload_dir.glob("*.mid")) + list(upload_dir.glob("*.midi"))
            if not midi_files:
                return "No MIDI files found in uploads directory. Please upload a MIDI file first."
            
            # Look for files that might match the context first
            question_lower = question.lower()
            matching_files = []
            
            for file in midi_files:
                file_lower = file.name.lower()
                if any(word in file_lower for word in ['coldplay', 'viva', 'vida'] if word in question_lower):
                    matching_files.append(file)
                elif any(word in file_lower for word in ['mario', 'bros', 'super'] if word in question_lower):
                    matching_files.append(file)
                elif any(word in file_lower for word in ['running'] if word in question_lower):
                    matching_files.append(file)
                elif any(word in file_lower for word in ['giorno'] if word in question_lower):
                    matching_files.append(file)
                elif any(word in file_lower for word in ['queen', 'bohemian', 'rhapsody'] if word in question_lower):
                    matching_files.append(file)
                elif any(word in file_lower for word in ['star', 'wars'] if word in question_lower):
                    matching_files.append(file)
                elif any(word in file_lower for word in ['imagine', 'dragons', 'radioactive'] if word in question_lower):
                    matching_files.append(file)
            
            if matching_files:
                midi_file = matching_files[0]
                print(f"DEBUG: Found matching file: {midi_file}")
            else:
                # If no keyword match, use the most recently modified file
                # This assumes the most recently uploaded file is the one we want to analyze
                midi_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                midi_file = midi_files[0]
                print(f"DEBUG: Using most recently modified file: {midi_file}")
                print(f"DEBUG: File modification time: {midi_file.stat().st_mtime}")
                print(f"DEBUG: All available files and their modification times:")
                for file in midi_files[:5]:  # Show top 5 most recent
                    print(f"  - {file.name}: {file.stat().st_mtime}")
                
                # Add a note that we're using the most recent file
                print(f"DEBUG: No specific filename found, using most recent file: {midi_file.name}")
        
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
        
        response = f"🎵 **MIDI Analysis for: {question}**\n\n"
        response += f"**File:** {basic_info.get('filename', 'Unknown')}\n\n"
        
        response += "**Musical Elements:**\n"
        if tempo_info.get('primary_tempo'):
            response += f"- **Tempo:** {tempo_info['primary_tempo']:.1f} BPM"
            if tempo_info.get('tempo_changes', 0) > 1:
                response += f" (with {tempo_info['tempo_changes']} tempo changes)"
            response += "\n"
        else:
            response += "- **Tempo:** Unknown\n"
        
        response += f"- **Key:** {key}\n"
        response += f"- **Time Signature:** {analysis.get('time_signatures', {}).get('primary_time_signature', '4/4')}\n"
        response += f"- **Duration:** {basic_info.get('length_seconds', 0):.1f} seconds\n"
        response += f"- **Tracks:** {basic_info.get('number_of_tracks', 0)}\n\n"
        
        if notes_info.get('total_notes', 0) > 0:
            response += "**Note Analysis:**\n"
            response += f"- **Total Notes:** {notes_info['total_notes']}\n"
            response += f"- **Pitch Range:** {notes_info['pitch_range']['min']} - {notes_info['pitch_range']['max']}\n"
            response += f"- **Average Velocity:** {notes_info['velocity_stats']['average']:.1f}\n"
            response += f"- **Unique Pitches:** {notes_info['unique_pitches']}\n\n"
        

        
        return clean_markdown(response)
        
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
    suggest_instruments_tool, 
    suggest_sample_packs_tool, 
    arrangement_advice_tool, 
    song_energy_tool,
    daw_functionality_tool,
    process_midi_file,
    analyze_uploaded_file,
]

# Add Tavily search tool if available
if tavily_search_tool:
    tool_belt.append(tavily_search_tool)

# Tool information for frontend
available_tools = [
    ToolInfo(name="write_lyrics_tool", description="Generate lyrics based on prompts, style, mood, or artist references"),
    ToolInfo(name="suggest_instruments_tool", description="Get instrument and synth preset recommendations"),
    ToolInfo(name="suggest_sample_packs_tool", description="Find sample packs and loops for your project"),
    ToolInfo(name="arrangement_advice_tool", description="Get song arrangement and structure suggestions"),
    ToolInfo(name="song_energy_tool", description="Analyze and improve song energy curves"),
    ToolInfo(name="daw_functionality_tool", description="Learn DAW features and get workflow tips"),
    ToolInfo(name="process_midi_file", description="Analyze MIDI files and provide musical insights"),
    ToolInfo(name="analyze_uploaded_file", description="Analyze uploaded files and provide insights"),
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
                            if uploaded_filename and tool_name in ['process_midi_file', 'analyze_uploaded_file']:
                                tool_args['filename'] = uploaded_filename
                                print(f"DEBUG: Added filename '{uploaded_filename}' to tool '{tool_name}'")
                            # Add question to instrument and sample pack tools for contextual suggestions
                            elif tool_name in ['suggest_instruments_tool', 'suggest_sample_packs_tool']:
                                tool_args['question'] = tool_args['question']
                                print(f"DEBUG: Added question to tool '{tool_name}' for contextual suggestions")
                            else:
                                print(f"DEBUG: No filename added to tool '{tool_name}', uploaded_filename: {uploaded_filename}")
                            tool_result = tool_to_use.invoke(tool_args)
                        else:
                            tool_result = tool_to_use.invoke(str(tool_args))
                    else:
                        # If tool_args is a string, create a dict with the question and filename
                        if uploaded_filename and tool_name in ['process_midi_file', 'analyze_uploaded_file']:
                            tool_args_dict = {'question': tool_args, 'filename': uploaded_filename}
                            print(f"DEBUG: Created dict with filename '{uploaded_filename}' for tool '{tool_name}'")
                            tool_result = tool_to_use.invoke(tool_args_dict)
                        elif tool_name in ['suggest_instruments_tool', 'suggest_sample_packs_tool']:
                            # For instrument and sample pack tools, pass the question for contextual suggestions
                            tool_args_dict = {'question': tool_args}
                            print(f"DEBUG: Created dict with question for tool '{tool_name}'")
                            tool_result = tool_to_use.invoke(tool_args_dict)
                        else:
                            # Even if no uploaded_filename, try to create a dict for the tool
                            if tool_name in ['process_midi_file', 'analyze_uploaded_file']:
                                tool_args_dict = {'question': tool_args}
                                print(f"DEBUG: Created dict without filename for tool '{tool_name}'")
                                tool_result = tool_to_use.invoke(tool_args_dict)
                            else:
                                tool_result = tool_to_use.invoke(tool_args)
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
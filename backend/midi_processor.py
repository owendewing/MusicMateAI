#!/usr/bin/env python3
"""
MIDI File Processor for Music Mate AI

This script processes MIDI files to extract musical information that can be used
by the AI agent to provide insights and recommendations.
"""

import mido
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import argparse

class MIDIProcessor:
    def __init__(self, midi_file_path: str):
        self.midi_file_path = Path(midi_file_path)
        self.midi_data = None
        self.analysis = {}
        
    def _midi_note_to_name(self, note_number: int) -> str:
        """Convert MIDI note number to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (note_number // 12) - 1
        note_name = note_names[note_number % 12]
        return f"{note_name}{octave}"
    
    def load_midi(self) -> bool:
        """Load and parse the MIDI file."""
        try:
            if not self.midi_file_path.exists():
                print(f"Error: MIDI file not found at {self.midi_file_path}")
                return False
                
            self.midi_data = mido.MidiFile(self.midi_file_path)
            return True
        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return False
    
    def analyze_basic_info(self) -> Dict[str, Any]:
        """Extract basic MIDI file information."""
        if not self.midi_data:
            return {}
            
        info = {
            'filename': self.midi_file_path.name,
            'ticks_per_beat': self.midi_data.ticks_per_beat,
            'length_seconds': self.midi_data.length,
            'number_of_tracks': len(self.midi_data.tracks)
        }
        
        return info
    
    def analyze_tempo(self) -> Dict[str, Any]:
        """Extract tempo information from the MIDI file."""
        if not self.midi_data:
            return {}
            
        tempos = []
        tempo_times = []
        
        for track in self.midi_data.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'set_tempo':
                    bpm = mido.tempo2bpm(msg.tempo)
                    tempos.append(bpm)
                    tempo_times.append(current_time)
        
        tempo_info = {
            'tempos': tempos,
            'tempo_times': tempo_times,
            'primary_tempo': tempos[0] if tempos else None,
            'tempo_changes': len(tempos),
            'average_tempo': sum(tempos) / len(tempos) if tempos else None
        }
        
        return tempo_info
    
    def analyze_time_signatures(self) -> Dict[str, Any]:
        """Extract time signature information."""
        if not self.midi_data:
            return {}
            
        time_signatures = []
        ts_times = []
        
        for track in self.midi_data.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'time_signature':
                    ts = f"{msg.numerator}/{msg.denominator}"
                    time_signatures.append(ts)
                    ts_times.append(current_time)
        
        ts_info = {
            'time_signatures': time_signatures,
            'ts_times': ts_times,
            'primary_time_signature': time_signatures[0] if time_signatures else None,
            'time_signature_changes': len(time_signatures)
        }
        
        return ts_info
    
    def analyze_notes(self) -> Dict[str, Any]:
        """Extract detailed note information."""
        if not self.midi_data:
            return {}
            
        all_notes = []
        track_notes = {}
        
        for track_num, track in enumerate(self.midi_data.tracks):
            track_notes[track_num] = []
            current_time = 0
            active_notes = {}
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_name = self._midi_note_to_name(msg.note)
                    note_info = {
                        'note': note_name,
                        'pitch': msg.note,
                        'velocity': msg.velocity,
                        'start_time': current_time,
                        'track': track_num
                    }
                    active_notes[msg.note] = note_info
                    track_notes[track_num].append(note_info)
                    
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        note_info = active_notes[msg.note]
                        note_info['end_time'] = current_time
                        note_info['duration'] = current_time - note_info['start_time']
                        all_notes.append(note_info)
                        del active_notes[msg.note]
        
        # Calculate statistics
        if all_notes:
            pitches = [note['pitch'] for note in all_notes]
            velocities = [note['velocity'] for note in all_notes]
            durations = [note['duration'] for note in all_notes]
            
            note_counts = {}
            for note in all_notes:
                note_name = note['note']
                note_counts[note_name] = note_counts.get(note_name, 0) + 1
            
            most_common_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            note_analysis = {
                'total_notes': len(all_notes),
                'unique_pitches': len(set(pitches)),
                'pitch_range': {
                    'min': min(pitches),
                    'max': max(pitches),
                    'range': max(pitches) - min(pitches)
                },
                'velocity_stats': {
                    'min': min(velocities),
                    'max': max(velocities),
                    'average': sum(velocities) / len(velocities)
                },
                'duration_stats': {
                    'min': min(durations),
                    'max': max(durations),
                    'average': sum(durations) / len(durations)
                },
                'most_common_notes': most_common_notes,
                'track_notes': track_notes,
                'all_notes': all_notes
            }
        else:
            note_analysis = {
                'total_notes': 0,
                'message': 'No notes found in MIDI file'
            }
        
        return note_analysis
    
    def analyze_tracks(self) -> Dict[str, Any]:
        """Analyze individual tracks."""
        if not self.midi_data:
            return {}
            
        track_analysis = {}
        
        for track_num, track in enumerate(self.midi_data.tracks):
            track_info = {
                'name': track.name if hasattr(track, 'name') else f'Track {track_num}',
                'total_messages': len(track),
                'note_messages': len([msg for msg in track if msg.type in ['note_on', 'note_off']]),
                'control_messages': len([msg for msg in track if msg.type.startswith('control_')]),
                'meta_messages': len([msg for msg in track if msg.is_meta])
            }
            
            # Count specific message types
            message_types = {}
            for msg in track:
                msg_type = msg.type
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            track_info['message_types'] = message_types
            track_analysis[f'track_{track_num}'] = track_info
        
        return track_analysis
    
    def generate_musical_insights(self) -> List[str]:
        """Generate musical insights based on the analysis."""
        # Return empty list to remove hardcoded insights
        return []
    
    def process(self) -> Dict[str, Any]:
        """Process the MIDI file and return comprehensive analysis."""
        if not self.load_midi():
            return {'error': 'Failed to load MIDI file'}
        
        self.analysis = {
            'basic_info': self.analyze_basic_info(),
            'tempo': self.analyze_tempo(),
            'time_signatures': self.analyze_time_signatures(),
            'notes': self.analyze_notes(),
            'tracks': self.analyze_tracks()
        }
        
        self.analysis['insights'] = self.generate_musical_insights()
        
        return self.analysis
    
    def save_analysis(self, output_path: str) -> bool:
        """Save the analysis to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.analysis, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False
    
    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        if not self.analysis:
            print("No analysis available. Run process() first.")
            return
        
        basic = self.analysis.get('basic_info', {})
        tempo = self.analysis.get('tempo', {})
        notes = self.analysis.get('notes', {})
        
        print(f"\n🎵 MIDI Analysis: {basic.get('filename', 'Unknown')}")
        print("=" * 50)
        
        print(f"Ticks per beat: {basic.get('ticks_per_beat', 'Unknown')}")
        print(f"Duration: {basic.get('length_seconds', 0):.1f} seconds")
        print(f"Tracks: {basic.get('number_of_tracks', 0)}")
        
        if tempo.get('primary_tempo'):
            print(f"Tempo: {tempo['primary_tempo']:.1f} BPM")
        
        if notes.get('total_notes', 0) > 0:
            print(f"Total Notes: {notes['total_notes']}")
            print(f"Pitch Range: {notes['pitch_range']['min']} - {notes['pitch_range']['max']}")
            print(f"Average Velocity: {notes['velocity_stats']['average']:.1f}")
        


def main():
    parser = argparse.ArgumentParser(description='Process MIDI files for musical analysis')
    parser.add_argument('midi_file', help='Path to the MIDI file to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file for analysis results')
    parser.add_argument('--summary', '-s', action='store_true', help='Print human-readable summary')
    
    args = parser.parse_args()
    
    processor = MIDIProcessor(args.midi_file)
    analysis = processor.process()
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return 1
    
    if args.output:
        if processor.save_analysis(args.output):
            print(f"Analysis saved to {args.output}")
        else:
            print("Failed to save analysis")
    
    if args.summary:
        processor.print_summary()
    
    return 0

if __name__ == "__main__":
    exit(main()) 
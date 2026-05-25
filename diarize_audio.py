#!/usr/bin/env python3
"""
Enhanced Audio Diarizer - Processes all WAV files in a folder
Saves speaker segments as WAV files (skips segments shorter than 2 seconds)
"""

import argparse
import numpy as np
from pathlib import Path
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio import Audio as PyannoteAudio
import warnings
warnings.filterwarnings("ignore")

def process_file(input_file, output_dir, pipeline, token, min_duration=2.0):
    """Process a single audio file"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*60}")
    
    # Load audio
    audio_loader = PyannoteAudio(sample_rate=16000, mono=True)
    waveform, sample_rate = audio_loader(str(input_file))
    duration = waveform.shape[1] / sample_rate
    print(f"  Audio duration: {duration:.1f} seconds")
    
    # Run diarization
    print(f"  Running diarization...")
    try:
        diarization = pipeline(str(input_file))
        print(f"  ✓ Diarization complete")
    except Exception as e:
        print(f"  ✗ Diarization failed: {e}")
        return 0, 0
    
    # Convert to numpy
    if hasattr(waveform, 'numpy'):
        audio_np = waveform.numpy()
    else:
        audio_np = np.array(waveform)
    
    if audio_np.ndim > 1:
        audio_np = audio_np[0]
    
    # Process segments
    segments_saved = 0
    segments_skipped = 0
    
    for turn, speaker in diarization.speaker_diarization:
        start_time = turn.start
        end_time = turn.end
        segment_duration = end_time - start_time
        
        # Skip short segments
        if segment_duration < min_duration:
            segments_skipped += 1
            continue
        
        # Convert time to samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure bounds
        start_sample = max(0, min(start_sample, len(audio_np) - 1))
        end_sample = max(start_sample + 1, min(end_sample, len(audio_np)))
        
        if start_sample >= end_sample:
            segments_skipped += 1
            continue
        
        # Extract and save
        segment_audio = audio_np[start_sample:end_sample]
        speaker_clean = speaker.replace(" ", "_").replace(":", "_").replace("/", "_")
        filename = f"{input_file.stem}_{speaker_clean}_{start_time:.2f}-{end_time:.2f}_{segment_duration:.2f}s.wav"
        filepath = output_dir / filename
        
        try:
            sf.write(str(filepath), segment_audio, sample_rate)
            segments_saved += 1
            print(f"    Saved: {filename} ({segment_duration:.1f}s)")
        except Exception as e:
            print(f"    Error saving: {e}")
    
    print(f"  ✓ Saved {segments_saved} segments, skipped {segments_skipped}")
    return segments_saved, segments_skipped

def main():
    parser = argparse.ArgumentParser(description='Diarize all WAV files in a folder')
    parser.add_argument('--input', '-i', required=True, help='Input directory with WAV files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for segments')
    parser.add_argument('--token', '-t', required=True, help='HuggingFace token')
    parser.add_argument('--min-duration', type=float, default=2.0,
                       help='Minimum segment duration in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print("\n" + "="*70)
    print("ENHANCED AUDIO DIARIZER")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Min segment duration: {args.min_duration}s")
    
    # Validate input directory
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(input_dir.glob("*.wav"))
    print(f"\nFound {len(wav_files)} WAV files to process")
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    # Load PyAnnote model (once for all files)
    print(f"\n[1/2] Loading PyAnnote model...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1"
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Process each file
    print(f"\n[2/2] Processing files...")
    total_saved = 0
    total_skipped = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        print(f"\n--- File {i}/{len(wav_files)} ---")
        saved, skipped = process_file(wav_file, output_dir, pipeline, args.token, args.min_duration)
        total_saved += saved
        total_skipped += skipped
    
    # Summary
    print("\n" + "="*70)
    print("DIARIZATION COMPLETE!")
    print("="*70)
    print(f"Files processed: {len(wav_files)}")
    print(f"Total segments saved: {total_saved}")
    print(f"Total segments skipped: {total_skipped}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()

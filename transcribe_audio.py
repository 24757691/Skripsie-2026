#!/usr/bin/env python3
"""
Slave 4: Transcription
Transcribes all WAV files in a directory and appends to one text file
Each transcription on its own line, in chronological order
"""

import os
import sys
import re
import torch
import soundfile as sf
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ======================================================
# MODEL PATHS BY LANGUAGE
# ======================================================
MODEL_PATHS = {
    "afrikaans": "/home/wentzel_viljoen/projects/skripsie/training/models/whisper-small-afrikaans-fleurs",
    "isixhosa": "TheirStory-Inc/whisper-small-xhosa"
}
# ======================================================

def get_timestamp(filename):
    """Extract start time from filename for sorting"""
    # Filename format: ..._SPEAKER_XX_12.34-56.78_...
    match = re.search(r'_(\d+\.\d+)-', str(filename))
    if match:
        return float(match.group(1))
    return 0

def transcribe_audio(input_dir, output_file, language):
    """Transcribe all WAV files and append to output file in chronological order"""
    
    print(f"\n📝 Transcribing audio from: {input_dir}")
    print(f"📄 Output file: {output_file}")
    print(f"🌐 Language: {language}")
    
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    
    # Get model path
    if language not in MODEL_PATHS:
        print(f"✗ Unknown language: {language}")
        print(f"  Available: {list(MODEL_PATHS.keys())}")
        return 0
    
    model_path = MODEL_PATHS[language]
    print(f"  Using model: {model_path}")
    
    # Get all WAV files and sort by timestamp
    wav_files = sorted(list(input_dir.glob("*.wav")), key=lambda x: get_timestamp(x))
    print(f"  Found {len(wav_files)} audio files")
    
    if not wav_files:
        print("  No WAV files found!")
        return 0
    
    # Load model and processor
    print("\n  Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        processor = WhisperProcessor.from_pretrained(model_path)
        print(f"  ✓ Model loaded on {device}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return 0
    
    # Write header
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(f"# Transcriptions for {language.upper()}\n")
        out_f.write(f"# Model: {model_path}\n")
        out_f.write(f"# Date: {__import__('datetime').datetime.now().isoformat()}\n")
        out_f.write(f"# Total segments: {len(wav_files)}\n")
        out_f.write("="*70 + "\n\n")
    
    successful = 0
    failed = 0
    
    for idx, audio_path in enumerate(wav_files, 1):
        filename = audio_path.name
        print(f"\n  [{idx}/{len(wav_files)}] Transcribing: {filename}")
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Process
            input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Append to file - ONE LINE PER SEGMENT
            with open(output_file, 'a', encoding='utf-8') as out_f:
                out_f.write(f"{transcription}\n")
            
            successful += 1
            print(f"    ✓ {transcription[:80]}{'...' if len(transcription) > 80 else ''}")
            
        except Exception as e:
            failed += 1
            print(f"    ✗ Error: {e}")
    
    print(f"\n✅ Transcription complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_file}")
    
    # Show first 5 lines as preview
    if successful > 0:
        print(f"\n📄 Preview of first 5 transcriptions:")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('=')]
            for i, line in enumerate(lines[:5], 1):
                print(f"  {i}. {line[:100]}{'...' if len(line) > 100 else ''}")
    
    return successful

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 transcribe_audio.py <input_dir> <output_file> <language>")
        print("  language: afrikaans or isixhosa")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    language = sys.argv[3]
    
    transcribe_audio(input_dir, output_file, language)

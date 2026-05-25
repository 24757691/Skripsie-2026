#!/usr/bin/env python3
"""
Master Pipeline for Afrikaans/isiXhosa ASR Data Collection
Runs all slave scripts sequentially to build a massive database
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# ======================================================
# CONFIGURATION
# ======================================================
BASE_DIR = Path("/home/wentzel_viljoen/projects/skripsie")
DATA_DIR = BASE_DIR / "data"
PIPELINE_DIR = BASE_DIR / "pipeline"

# Slave scripts paths
SCRAPER_SCRIPT = PIPELINE_DIR / "youtube_scraper.py"
DOWNLOADER_SCRIPT = PIPELINE_DIR / "audio_downloader.py"
DIARIZE_SCRIPT = PIPELINE_DIR / "diarize_audio.py"
TRANSCRIBE_SCRIPT = PIPELINE_DIR / "transcribe_audio.py"

# Output paths
LINKS_FILE = DATA_DIR / "youtube_links.txt"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
SEGMENTS_DIR = DATA_DIR / "segments"
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.txt"

# HuggingFace token
HF_TOKEN = "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"
# ======================================================

def run_slave(script_path, description, args=None):
    """Run a slave script and check if it succeeded"""
    print("\n" + "="*70)
    print(f"▶ RUNNING: {description}")
    print("="*70)
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} FAILED!")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline for ASR data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_pipeline.py --language afrikaans --max-videos 100
  python master_pipeline.py --language isixhosa --max-videos 50
  python master_pipeline.py --language afrikaans --skip-scrape
        """
    )
    
    parser.add_argument("--language", "-l", required=True,
                       choices=["afrikaans", "isixhosa"],
                       help="Language to process")
    
    parser.add_argument("--max-videos", "-m", type=int, default=100,
                       help="Maximum number of videos to scrape (default: 100)")
    
    parser.add_argument("--skip-scrape", action="store_true",
                       help="Skip YouTube scraping (use existing links file)")
    
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip audio download (use existing raw audio)")
    
    parser.add_argument("--skip-diarize", action="store_true",
                       help="Skip diarization (use existing segments)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MASTER PIPELINE - ASR DATA COLLECTION")
    print("="*70)
    print(f"Language: {args.language}")
    print(f"Max videos: {args.max_videos}")
    print(f"Base directory: {BASE_DIR}")
    
    # Create necessary directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track pipeline status
    pipeline_success = True
    
    # ======================================================
    # SLAVE 1: YouTube Scraper
    # ======================================================
    if not args.skip_scrape:
        print("\n" + "="*70)
        print("SLAVE 1: YouTube Link Scraper")
        print("="*70)
        
        # Run the scraper - it saves to pipeline/afrikaans_links.txt or pipeline/isixhosa_links.txt
        success = run_slave(
            SCRAPER_SCRIPT,
            "YouTube Scraper",
            [args.language, "--max-videos", str(args.max_videos)]
        )
        
        if not success:
            pipeline_success = False
            print("\n❌ Pipeline stopped due to Slave 1 failure")
            sys.exit(1)
        
        # Find where the scraper saved the file (it saves in the current working directory)
        scraper_output = Path.cwd() / f"{args.language}_links.txt"
        
        if scraper_output.exists():
            shutil.copy(scraper_output, LINKS_FILE)
            print(f"  ✓ Copied links from {scraper_output} to {LINKS_FILE}")
        else:
            print(f"  ⚠ Warning: Could not find {scraper_output}")
            print(f"  Looking for links file in current directory...")
            
            # Try to find any links file
            for f in Path.cwd().glob("*_links.txt"):
                print(f"  Found: {f}")
                shutil.copy(f, LINKS_FILE)
                break
    else:
        print("\n⏭ Skipping Slave 1 (YouTube Scraper)")
        if not LINKS_FILE.exists():
            print(f"⚠ Warning: Links file not found at {LINKS_FILE}")
    
    # ======================================================
    # SLAVE 2: Audio Downloader
    # ======================================================
    if not args.skip_download:
        print("\n" + "="*70)
        print("SLAVE 2: Audio Downloader")
        print("="*70)
        
        if not LINKS_FILE.exists():
            print(f"✗ Links file not found: {LINKS_FILE}")
            print("  Please run scraper first or provide links file")
            pipeline_success = False
            sys.exit(1)
        
        success = run_slave(
            DOWNLOADER_SCRIPT,
            "Audio Downloader",
            [str(LINKS_FILE), str(RAW_AUDIO_DIR)]
        )
        
        if not success:
            pipeline_success = False
            print("\n❌ Pipeline stopped due to Slave 2 failure")
            sys.exit(1)
    else:
        print("\n⏭ Skipping Slave 2 (Audio Downloader)")
    
    # ======================================================
    # SLAVE 3: Diarization
    # ======================================================
    if not args.skip_diarize:
        print("\n" + "="*70)
        print("SLAVE 3: Audio Diarization")
        print("="*70)
        
        # Check if there are audio files to process
        audio_files = list(RAW_AUDIO_DIR.glob("*.wav"))
        if not audio_files:
            print(f"⚠ No audio files found in {RAW_AUDIO_DIR}")
            print("  Skipping diarization")
        else:
            success = run_slave(
                DIARIZE_SCRIPT,
                "Diarization",
                ["--input", str(RAW_AUDIO_DIR), "--output", str(SEGMENTS_DIR), "--token", HF_TOKEN, "--min-duration", "1.0"]
            )
            
            if not success:
                pipeline_success = False
                print("\n❌ Pipeline stopped due to Slave 3 failure")
                sys.exit(1)
    else:
        print("\n⏭ Skipping Slave 3 (Diarization)")
    
    # ======================================================
    # SLAVE 4: Transcription
    # ======================================================
    print("\n" + "="*70)
    print("SLAVE 4: Transcription")
    print("="*70)
    
    # Check if there are segments to transcribe
    segment_files = list(SEGMENTS_DIR.glob("*.wav"))
    if not segment_files:
        print(f"⚠ No segment files found in {SEGMENTS_DIR}")
        print("  Skipping transcription")
    else:
        success = run_slave(
            TRANSCRIBE_SCRIPT,
            "Transcription",
            [str(SEGMENTS_DIR), str(TRANSCRIPTS_FILE), args.language]
        )
        
        if not success:
            pipeline_success = False
            print("\n❌ Pipeline stopped due to Slave 4 failure")
            sys.exit(1)
    
    # ======================================================
    # FINAL SUMMARY
    # ======================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    if pipeline_success:
        print("\n✅ All slaves completed successfully!")
        print(f"\nOutput files:")
        print(f"  YouTube links: {LINKS_FILE}")
        print(f"  Raw audio: {RAW_AUDIO_DIR}/")
        print(f"  Diarized segments: {SEGMENTS_DIR}/")
        print(f"  Transcripts: {TRANSCRIPTS_FILE}")
        
        # Count files
        audio_count = len(list(RAW_AUDIO_DIR.glob("*.wav")))
        segment_count = len(list(SEGMENTS_DIR.glob("*.wav")))
        
        print(f"\nStatistics:")
        print(f"  Audio files downloaded: {audio_count}")
        print(f"  Segments created: {segment_count}")
        
        # Count transcript lines
        if TRANSCRIPTS_FILE.exists():
            with open(TRANSCRIPTS_FILE, 'r') as f:
                lines = f.readlines()
                text_lines = [l for l in lines if l.startswith("TEXT:")]
            print(f"  Transcript lines: {len(text_lines)}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

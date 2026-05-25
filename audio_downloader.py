#!/usr/bin/env python3
"""
Slave 2: Audio Downloader
Downloads audio from YouTube links and converts to 16kHz WAV
"""

import os
import sys
import subprocess
from pathlib import Path

def download_audio(links_file, output_dir):
    """Download audio from YouTube links using yt-dlp"""
    
    print(f"\n📥 Downloading audio from: {links_file}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read links
    with open(links_file, 'r') as f:
        links = [line.strip() for line in f if line.strip()]
    
    print(f"  Found {len(links)} links")
    
    # yt-dlp options
    ydl_opts = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--postprocessor-args', 'ffmpeg:-ar 16000 -ac 1',
        '--output', f'{output_dir}/%(title)s_%(id)s.%(ext)s',
        '--quiet',
        '--no-warnings'
    ]
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(links, 1):
        print(f"\n  [{i}/{len(links)}] Downloading: {url[:60]}...")
        
        try:
            # Run yt-dlp
            result = subprocess.run(
                ydl_opts + [url],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per video
            )
            
            if result.returncode == 0:
                successful += 1
                print(f"    ✓ Downloaded")
            else:
                failed += 1
                print(f"    ✗ Failed: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"    ✗ Timeout (5 min)")
        except Exception as e:
            failed += 1
            print(f"    ✗ Error: {e}")
    
    print(f"\n✅ Download complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    
    return successful

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 audio_downloader.py <links_file> <output_dir>")
        sys.exit(1)
    
    links_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    download_audio(links_file, output_dir)

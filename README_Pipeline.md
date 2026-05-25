ASR Data Collection Pipeline
Overview
This pipeline automates the end-to-end process of collecting, processing, and transcribing speech data for low-resource South African languages (Afrikaans and isiXhosa). It transforms YouTube news broadcasts into speaker-separated, timestamped transcriptions suitable for ASR training and evaluation.

Pipeline Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   YouTube       │    │   Audio         │    │   Speaker       │    │   Transcription │
│   Scraper       │───▶│   Downloader    │───▶│   Diarization   │───▶│   Script        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
        ▼                      ▼                      ▼                      ▼
   links.txt              raw_audio/              segments/              transcripts.txt
   
Prerequisites
Hardware Requirements
Component	Minimum	Recommended
GPU	4GB VRAM	8GB+ VRAM
RAM	8GB	16GB
Storage	10GB	20GB

Software Requirements
Python 3.10+
Conda (recommended) or virtual environment
Windows Subsystem for Linux (WSL) if on Windows

ffmpeg

External Accounts
Service	        Purpose	                            Required
Google          Cloud Platform	YouTube API access	Yes
Hugging Face	  PyAnnote model access	              Yes

Installation
1. Clone or Download Scripts
Place all pipeline scripts in a single directory:
master_pipeline.py
youtube_scraper.py
audio_downloader.py
diarize_audio.py
transcribe_audio.py

2. Install Dependencies
pip install google-api-python-client yt-dlp pyannote.audio transformers torch torchaudio soundfile librosa jiwer pandas numpy

4. Install ffmpeg
Ubuntu/WSL:
sudo apt update
sudo apt install ffmpeg

4. Set Up YouTube API Key
Go to Google Cloud Console
Create a new project
Enable YouTube Data API v3
Create credentials → API key
Restrict the key to YouTube Data API v3 only
Open youtube_scraper.py and replace the API key:
python
YOUR_API_KEY = "YOUR_API_KEY_HERE"

6. Set Up Hugging Face Authentication
Create an account at huggingface.co
Go to Settings → Access Tokens → New token (read permissions)
Accept terms for pyannote/speaker-diarization-community-1
In master_pipeline.py, update the token:
python
HF_TOKEN = "hf_YOUR_TOKEN_HERE"

6. Configure rclone (Optional, for Google Drive backup)
curl https://rclone.org/install.sh | sudo bash
rclone config
# Follow prompts to connect Google Drive

Usage
Basic Usage
Run full pipeline for Afrikaans (10 videos):
python master_pipeline.py --language afrikaans --max-videos 10

Run full pipeline for isiXhosa:
python master_pipeline.py --language isixhosa --max-videos 10

Skip Flags (Resume from Interruption)
Scenario	Command
Resume from download stage	python master_pipeline.py --language afrikaans --skip-scrape
Resume from diarization stage	python master_pipeline.py --language afrikaans --skip-scrape --skip-download
Only transcribe existing segments	python master_pipeline.py --language afrikaans --skip-scrape --skip-download --skip-diarize

Command-Line Arguments
Argument	          Description	                       Default
--language, -l	    Language: afrikaans or isixhosa	   Required
--max-videos, -m	  Maximum videos to scrape	         100
--skip-scrape	      Skip YouTube scraping	             False
--skip-download	    Skip audio download	               False
--skip-diarize	    Skip speaker diarization	         False

Output Structure
After successful execution:
├── pipeline/                          # Scripts directory
└── data/
    ├── youtube_links.txt              # Scraped YouTube URLs
    ├── raw_audio/                     # Downloaded WAV files
    │   ├── video1.wav
    │   └── ...
    ├── segments/                      # Speaker-separated segments
    │   ├── video1_SPEAKER_00_0.0-12.5.wav
    │   └── ...
    └── transcripts.txt                # Final transcriptions (one per line)
    
File Formats
Input Data Format (for fine-tuning)
dataset_folder/
├── audio/
│   ├── sample_001.wav
│   └── ...
└── transcriptions.tsv
TSV format (tab-separated, no header):

filename_1	Transcription text here
filename_2	Another transcription

Output Transcript Format
# Transcriptions for AFRIKAANS
# Model: /path/to/model
# Date: 2026-05-25T10:30:00
# Total segments: 42
======================================================================

First transcription line.
Second transcription line.
Third transcription line.
Troubleshooting
Issue	Solution
YouTube API key invalid	Regenerate key in Google Cloud Console
PyAnnote model access denied	Accept terms on Hugging Face and use valid token
Diarization takes too long	Process shorter audio files or use GPU
Out of memory	Reduce batch size in fine-tuning script
No segments created	Check diarization output; may need longer processing

Individual Script Usage
YouTube Scraper (Standalone)
python youtube_scraper.py afrikaans --max-videos 10

Audio Downloader (Standalone)
python audio_downloader.py /path/to/links.txt /path/to/output/dir

Diarization (Standalone)
python diarize_audio.py --input /path/to/raw_audio --output /path/to/segments --token hf_TOKEN --min-duration 1.0

Transcription (Standalone)
python transcribe_audio.py /path/to/segments /path/to/transcripts.txt afrikaans

License
This project is for academic research purposes. Third-party libraries and models retain their respective licenses.

References
Hugging Face Transformers: https://github.com/huggingface/transformers

PyAnnote: https://github.com/pyannote/pyannote-audio

yt-dlp: https://github.com/yt-dlp/yt-dlp

OpenAI Whisper: https://github.com/openai/whisper

README: Universal Whisper Fine-Tuning Module
Overview
This module fine-tunes pre-trained Whisper models on language-specific data for low-resource South African languages (Afrikaans and isiXhosa). The script automatically detects available GPU resources, optimises training parameters, and produces portable fine-tuned models.

Prerequisites
Hardware Requirements
Component	Minimum	Recommended
GPU	6GB VRAM (GTX 1660)	24GB VRAM (RTX 3090/4090)
RAM	16GB	32GB
Storage	10GB free	20GB free
Software Requirements
Python 3.10+

CUDA-capable GPU (recommended)

rclone (for Google Drive integration)

Installation
1. Install Dependencies
bash
pip install transformers datasets evaluate jiwer accelerate torchaudio soundfile librosa tensorboard pandas
2. Install rclone (Optional, for Google Drive)
bash
curl https://rclone.org/install.sh | sudo bash
rclone config
# Follow prompts to connect Google Drive
Data Preparation
Input Format
Your training data must be structured as:

text
dataset_folder/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── transcriptions.tsv
TSV format (tab-separated, no header):

text
sample_001	Transcription text here
sample_002	Another transcription
Audio requirements:

Format: WAV (any sample rate, will be resampled to 16kHz)

Channels: Mono or stereo (converted to mono)

Duration: Any (padded/truncated to 30 seconds)

Upload to Google Drive (Recommended)
bash
tar -czvf your_data.tar.gz dataset_folder/
rclone copy your_data.tar.gz "gdrive:/Skripsie Training Data/"
Configuration
Edit the configuration section at the top of universal_trainer.py:

python
# ======================================================
# CONFIGURATION - EDIT THESE BEFORE RUNNING
# ======================================================

# Training data (must be .tar.gz file on Google Drive)
DATA_TAR = "your_data.tar.gz"

# Starting model - CHOOSE ONE:
# Option A: Local fine-tuned model (.tar.gz file on Google Drive)
MODEL_TAR = ""  # e.g., "whisper-small-afrikaans-slr-model.tar.gz"
# Option B: HuggingFace model (leave MODEL_TAR empty)
MODEL_NAME = "openai/whisper-small"  # or "TheirStory-Inc/whisper-small-xhosa"

# Output settings
OUTPUT_NAME = "whisper-small-language-finetuned"
LANGUAGE = "afrikaans"  # Language code (afrikaans, xhosa, etc.)

# Training settings
BASE_BATCH_SIZE = 8
MAX_STEPS = 5000
LEARNING_RATE = 1e-5

# Google Drive paths
DRIVE_FOLDER = "Skripsie Training Data"

# ======================================================
Usage
Running on Vast.ai (Recommended for Training)
Rent an instance with PyTorch template and RTX 3090/4090

SSH into the instance:

bash
ssh -p [PORT] root@[IP_ADDRESS]
Install dependencies:

bash
pip install transformers datasets evaluate jiwer accelerate torchaudio soundfile librosa tensorboard pandas
Set up rclone:

bash
mkdir -p /root/.config/rclone
nano /root/.config/rclone/rclone.conf
# Paste your rclone config from local machine
Download and run the script:

bash
cd /workspace
rclone copy "gdrive:/Skripsie Training Data/universal_trainer.py" ./
rclone copy "gdrive:/Skripsie Training Data/your_data.tar.gz" ./
python universal_trainer.py
Running Locally
bash
python universal_trainer.py
After Training
The fine-tuned model is automatically:

Saved to /workspace/[OUTPUT_NAME]/

Compressed to [OUTPUT_NAME].tar.gz

Uploaded to Google Drive at Skripsie Training Data/models/

Training Hyperparameters
Parameter	Value	Description
Learning rate	1e-5	Controls weight change magnitude
Warmup steps	500	Gradual LR increase to prevent instability
Max steps	5000	Total training updates
Batch size	8-16 (auto-detected)	Samples processed simultaneously
Gradient accumulation	1-2 (auto-detected)	Simulates larger batch sizes
FP16	True	Mixed precision training
Evaluation strategy	Every 500 steps	Validation frequency
Early stopping patience	3	Stops if no improvement
Save total limit	3	Keep only latest checkpoints
Output Files
The fine-tuned model directory contains:

File	Purpose
config.json	Model architecture configuration
model.safetensors	Trained weights (~500MB)
tokenizer.json	Tokenizer for text encoding
vocab.json	Vocabulary mapping
merges.txt	Byte-pair encoding merges
generation_config.json	Generation parameters
preprocessor_config.json	Audio preprocessing settings
Testing Your Fine-Tuned Model
python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf

model = WhisperForConditionalGeneration.from_pretrained("/path/to/fine-tuned/model")
processor = WhisperProcessor.from_pretrained("/path/to/fine-tuned/model")

audio, sr = sf.read("test_audio.wav")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

with torch.no_grad():
    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)
Troubleshooting
Issue	Solution
Out of memory	Reduce BASE_BATCH_SIZE or use --skip-diarize
evaluate.load("wer") fails	Replace with jiwer.wer() in compute_metrics
Model not uploading	Check rclone config and Google Drive space
Training too slow	Use GPU instance on Vast.ai
References
Hugging Face Fine-Tuning Guide: https://huggingface.co/blog/fine-tune-whisper

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.

Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. OpenAI.

"""
UNIVERSAL WHISPER FINE-TUNING SCRIPT
------------------------------------
Self-optimizes for available GPUs
Works with:
  - Local fine-tuned models (.tar.gz)
  - HuggingFace base models (e.g., "openai/whisper-small")
  - HuggingFace fine-tuned models (e.g., "TheirStory-Inc/whisper-small-xhosa")

HOW TO USE:
1. Edit the CONFIGURATION section below
2. Run: python3 universal_trainer_complete.py
3. Your fine-tuned model will be saved back to Google Drive
"""

# ======================================================
# CONFIGURATION - EDIT THESE BEFORE RUNNING
# ======================================================

# Training data (must be .tar.gz file on Google Drive)
DATA_TAR = "fleurs_isixhosa_data.tar.gz"

# Starting model - CHOOSE ONE:
# Option A: Local fine-tuned model (.tar.gz file on Google Drive)
MODEL_TAR = ""  # e.g., "whisper-afrikaans-slr-model.tar.gz"
# Option B: HuggingFace model (leave MODEL_TAR empty)
MODEL_NAME = "openai/whisper-small"  # e.g., "openai/whisper-small"

# Output settings
OUTPUT_NAME = "whisper-small-xhosa-fleurs"
LANGUAGE = "xhosa"  # Language code (afrikaans, xhosa, etc.)

# Training settings
BASE_BATCH_SIZE = 8
MAX_STEPS = 5000
LEARNING_RATE = 1e-5

# Google Drive paths
DRIVE_FOLDER = "Skripsie Training Data"

# ======================================================
# END OF CONFIGURATION
# ======================================================

import os
import sys
import torch
import pandas as pd
import shutil
import jiwer
import subprocess
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate
from dataclasses import dataclass
from typing import Any

# ======================================================
# AUTO-DETECT GPU SETUP
# ======================================================
GPU_COUNT = torch.cuda.device_count()
GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]

def is_high_end_gpu(name):
    name = name.lower()
    high_end = ['rtx 4090', 'rtx 4080', 'rtx 5090', 'rtx 5080', 'a100', 'h100', 'v100']
    return any(h in name for h in high_end)

# Calculate optimal batch size based on GPU count and type
if GPU_COUNT == 0:
    PER_DEVICE_BATCH = BASE_BATCH_SIZE // 2
    if PER_DEVICE_BATCH < 1:
        PER_DEVICE_BATCH = 1
    print("⚠️ No GPU detected! Using CPU (very slow)")
elif GPU_COUNT == 1:
    if is_high_end_gpu(GPU_NAMES[0]):
        PER_DEVICE_BATCH = BASE_BATCH_SIZE * 2  # 16 for high-end
    else:
        PER_DEVICE_BATCH = BASE_BATCH_SIZE      # 8 for standard
else:
    # Multi-GPU: use base batch size per GPU
    PER_DEVICE_BATCH = BASE_BATCH_SIZE

EFFECTIVE_BATCH = PER_DEVICE_BATCH * max(1, GPU_COUNT)
GRADIENT_ACCUMULATION = max(1, BASE_BATCH_SIZE * 2 // EFFECTIVE_BATCH)

print("="*70)
print("GPU AUTO-DETECTION RESULTS")
print("="*70)
print(f"GPUs detected: {GPU_COUNT}")
for i, name in enumerate(GPU_NAMES):
    print(f"  GPU {i}: {name}")
print(f"\nOptimized training settings:")
print(f"  Per-device batch size: {PER_DEVICE_BATCH}")
print(f"  Total effective batch: {EFFECTIVE_BATCH}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max steps: {MAX_STEPS}")

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def run_command(cmd, description):
    """Run a shell command and print progress"""
    print(f"\n▶ {description}...")
    result = os.system(cmd)
    if result != 0:
        print(f"  ❌ Failed: {cmd}")
        return False
    print(f"  ✓ Done")
    return True

def load_dataset():
    """Load dataset from extracted folder"""
    print(f"\n📂 Loading dataset...")
    
    # Find the extracted folder
    possible_folders = [d for d in Path('/workspace').iterdir() 
                        if d.is_dir() and d.name not in ['workspace', 'whisper', 'checkpoints']]
    
    # Look for folder containing transcriptions.tsv
    data_dir = None
    for folder in possible_folders:
        if (folder / "transcriptions.tsv").exists():
            data_dir = folder
            break
    
    if data_dir is None:
        raise FileNotFoundError("Could not find dataset folder with transcriptions.tsv")
    
    tsv_path = data_dir / "transcriptions.tsv"
    audio_dir = data_dir / "audio"
    
    print(f"  Found data in: {data_dir}")
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['filename', 'text'])
    print(f"  Loaded {len(df)} samples from TSV")
    
    audio_paths = []
    texts = []
    missing = 0
    
    for _, row in df.iterrows():
        filename = row['filename'].strip()
        text = row['text'].strip()
        
        # Remove .wav extension if present
        if filename.endswith('.wav'):
            filename = filename[:-4]
        
        audio_path = audio_dir / f"{filename}.wav"
        
        if audio_path.exists():
            audio_paths.append(str(audio_path))
            texts.append(text)
        else:
            missing += 1
    
    print(f"  Found {len(audio_paths)} valid audio files")
    if missing > 0:
        print(f"  ⚠️ {missing} files missing from audio folder")
    
    if len(audio_paths) == 0:
        # Debug output
        print("\n  Debug - First 5 TSV filenames:")
        for _, row in df.head().iterrows():
            fname = row['filename'].strip()
            if fname.endswith('.wav'):
                fname = fname[:-4]
            print(f"    {fname}")
        
        print("\n  Debug - First 5 audio files:")
        for f in list(audio_dir.glob("*.wav"))[:5]:
            print(f"    {f.stem}")
        
        raise ValueError("No matching audio files found!")
    
    dataset = Dataset.from_dict({"audio_path": audio_paths, "sentence": texts})
    
    def load_audio(batch):
        import soundfile as sf
        import librosa
        audio_array = []
        for path in batch["audio_path"]:
            audio, sr = sf.read(path)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio_array.append({"array": audio, "sampling_rate": 16000})
        return {"audio": audio_array}
    
    dataset = dataset.map(load_audio, batched=True, batch_size=100)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    dataset_dict = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    print(f"  Train size: {len(dataset_dict['train'])}")
    print(f"  Validation size: {len(dataset_dict['validation'])}")
    
    return dataset_dict

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(
        batch["sentence"], truncation=True, max_length=448
    ).input_ids
    return batch

@dataclass
class DataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    wer_metric = 100 * jiwer.wer(label_str, pred_str)
    return {"wer": wer}

# ======================================================
# MAIN FUNCTION
# ======================================================

def main():
    print("="*70)
    print("UNIVERSAL WHISPER FINE-TUNING SCRIPT")
    print("="*70)
    
    # Step 1: Check rclone configuration
    if not Path("/root/.config/rclone/rclone.conf").exists():
        print("\n❌ rclone not configured!")
        print("Please run: rclone config to set up Google Drive")
        sys.exit(1)
    
    os.chdir("/workspace")
    
    # Step 2: Download data if not already present
    if not Path(DATA_TAR).exists():
        if not run_command(f"rclone copy 'gdrive:/{DRIVE_FOLDER}/{DATA_TAR}' ./", 
                           f"Downloading {DATA_TAR}"):
            sys.exit(1)
    
    # Step 3: Download model if it's a local tar file
    model_is_local = False
    if MODEL_TAR and MODEL_TAR != "":
        if not Path(MODEL_TAR).exists():
            if not run_command(f"rclone copy 'gdrive:/{DRIVE_FOLDER}/{MODEL_TAR}' ./",
                               f"Downloading {MODEL_TAR}"):
                sys.exit(1)
        run_command(f"tar -xzvf {MODEL_TAR}", "Extracting model")
        model_is_local = True
    
    # Step 4: Extract training data
    if not run_command(f"tar -xzvf {DATA_TAR}", "Extracting training data"):
        sys.exit(1)
    
    # Step 5: Determine model path
    if MODEL_NAME and MODEL_NAME != "":
        MODEL_PATH = MODEL_NAME
        print(f"\n  Using HuggingFace model: {MODEL_PATH}")
    elif model_is_local:
        # Find the extracted model folder
        model_folders = [d for d in Path('.').iterdir() 
                        if d.is_dir() and 'whisper' in d.name.lower()]
        if model_folders:
            MODEL_PATH = str(model_folders[0])
            print(f"\n  Using local model: {MODEL_PATH}")
        else:
            MODEL_PATH = "openai/whisper-small"
            print(f"\n  Using base model: {MODEL_PATH}")
    else:
        MODEL_PATH = "openai/whisper-small"
        print(f"\n  Using base model: {MODEL_PATH}")
    
    # Step 6: Load dataset
    dataset = load_dataset()
    
    # Step 7: Load processor
    print("\n🔧 Loading processor...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language=LANGUAGE, task="transcribe")
    
    # Step 8: Prepare dataset
    print("\n📊 Preparing dataset...")
    train_dataset = dataset["train"].map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["validation"].map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=dataset["validation"].column_names
    )
    
    # Step 9: Load model
    print("\n🤖 Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Step 10: Data collator
    data_collator = DataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Step 11: Training arguments
    output_dir = f"/workspace/{OUTPUT_NAME}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        save_total_limit=3,
        dataloader_num_workers=min(8, os.cpu_count()),
        ddp_find_unused_parameters=False if GPU_COUNT > 1 else None,
    )
    
    # Step 12: Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Step 13: Train
    print("\n🏋️ Starting fine-tuning...")
    print(f"  Output: {output_dir}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Language: {LANGUAGE}")
    print(f"  Steps: {MAX_STEPS}")
    print(f"  Batch per GPU: {PER_DEVICE_BATCH}")
    print(f"  GPUs: {GPU_COUNT}")
    print(f"  Total batch: {PER_DEVICE_BATCH * max(1, GPU_COUNT)}")
    print("")
    
    trainer.train()
    
    # Step 14: Save final model
    print("\n💾 Saving final model...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Clean up checkpoints
    for checkpoint_dir in Path(output_dir).glob("checkpoint-*"):
        print(f"  Removing {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
    
    # Step 15: Compress and upload
    print("\n📤 Compressing and uploading to Google Drive...")
    final_tar = f"{OUTPUT_NAME}.tar.gz"
    os.system(f"tar -czvf {final_tar} {OUTPUT_NAME}/")
    os.system(f"rclone copy {final_tar} 'gdrive:/{DRIVE_FOLDER}/models/'")
    
    print("\n" + "="*70)
    print("✅ PROCESS COMPLETE!")
    print("="*70)
    print(f"\nFinal model saved to: {output_dir}")
    print(f"Uploaded to: gdrive:/{DRIVE_FOLDER}/models/{final_tar}")
    print("\nYou can download it from Google Drive or use it for future training.")

if __name__ == "__main__":
    main()

# ==========================================================================
# BirdCLEF+ 2026 -- Kaggle Submission Notebook (CPU-only)
# ==========================================================================
# This competition does not allow GPU notebooks, so this version forces CPU
# inference and is written to keep peak memory bounded across all ~600 test
# files -- the most common causes of mid-run "fragmentation" / OOM-style
# errors on Kaggle's CPU containers are (a) rebuilding heavy objects (audio
# transforms, DataLoaders) on every iteration of a long loop instead of
# reusing them, (b) accumulating all predictions in memory before writing
# anything to disk, and (c) PyTorch oversubscribing threads relative to the
# container's actual CPU quota. This script addresses all three.
#
# BEFORE RUNNING:
#   1. Upload your checkpoint as a Kaggle Dataset and attach it to this notebook.
#   2. Set MODEL_CHECKPOINT_PATH below to point at it.
#   3. Make sure CLIP_LENGTH_SEC matches what the checkpoint was trained with.
# ==========================================================================

import os

# Cap BLAS/OMP thread pools BEFORE importing torch.
# FIX: Changed to direct assignment to overwrite Kaggle's default container settings.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import gc
import csv
import glob
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from torchvision.models import efficientnet_b3
from tqdm import tqdm

torch.set_num_threads(4)  # adjust to match the notebook's actual CPU allocation

# ==========================================
# CONFIG -- adjust these for your setup
# ==========================================
# FIX: Corrected standard Kaggle competition directory path
COMP_DIR                = "<insert_competition_folder_here>"
TEST_SOUNDSCAPES_DIR    = os.path.join(COMP_DIR, "test_soundscapes")
TAXONOMY_PATH           = os.path.join(COMP_DIR, "taxonomy.csv")
SAMPLE_SUBMISSION_PATH  = os.path.join(COMP_DIR, "sample_submission.csv")

# Point this at whichever checkpoint you attached as a Kaggle Dataset.
MODEL_CHECKPOINT_PATH = "<insert_model_path_here>"

CLIP_LENGTH_SEC  = 5.0     # must match training
SAMPLE_RATE      = 32000   # must match training
INFER_BATCH_SIZE = 16      # smaller than training's batch size on purpose
GC_EVERY_N_FILES = 25      # how often to force a garbage-collection pass

OUTPUT_PATH = "/kaggle/working/submission.csv"

DEVICE = torch.device("cpu")  # hardcoded: this competition disallows GPU


# ==========================================
# MODEL DEFINITION (must match train_birdclef.py exactly)
# ==========================================
class EfficientBirbNN(nn.Module):
    def __init__(self, num_classes=234, pretrained=False):
        super().__init__()
        self.base_model = efficientnet_b3(weights=None)  # never download weights for inference

        original_conv = self.base_model.features[0][0]
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# ==========================================
# AUDIO -> SPECTROGRAM HELPER
# ==========================================
mel_spect = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, n_mels=128)
amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')


def load_and_chunk(audio_path, clip_length_sec=CLIP_LENGTH_SEC, sample_rate=SAMPLE_RATE):
    """
    Loads one soundscape file and yields (spectrogram, end_time_seconds) for
    each non-overlapping clip-length window. Plain generator over tensors.
    """
    chunk_size = int(sample_rate * clip_length_sec)

    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    num_chunks = waveform.shape[1] // chunk_size

    for idx in range(num_chunks):
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size
        chunk = waveform[:, start_idx:end_idx]

        spectrogram = mel_spect(chunk)
        spectrogram = amp_to_db(spectrogram)
        mean, std = spectrogram.mean(), spectrogram.std() + 1e-6
        spectrogram = (spectrogram - mean) / std

        end_time = (idx + 1) * int(clip_length_sec)
        yield spectrogram, end_time

    del waveform  # explicit: drop the full waveform before the next file loads


def _run_batch_and_write(model, spec_batch, end_time_batch, filename, idx_to_label, num_classes, writer):
    """Stacks a list of single spectrograms into one batch, runs the model
    once, and immediately writes each resulting row to the CSV writer."""
    batch_tensor = torch.stack(spec_batch, dim=0).to(DEVICE)
    logits = model(batch_tensor)
    probs = torch.sigmoid(logits).numpy()

    for i, end_time in enumerate(end_time_batch):
        row = {"row_id": f"{filename}_{end_time}"}
        row.update({idx_to_label[c]: float(probs[i, c]) for c in range(num_classes)})
        writer.writerow(row)

    del batch_tensor, logits, probs


# ==========================================
# MAIN INFERENCE ROUTINE
# ==========================================
if __name__ == "__main__":
    print(f"Running inference on: {DEVICE} (torch.get_num_threads()={torch.get_num_threads()})")

    # --- Label ordering must reproduce training exactly ---
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    label_order = list(taxonomy['primary_label'].unique())
    idx_to_label = {i: label for i, label in enumerate(label_order)}
    num_classes = len(label_order)
    print(f"Loaded {num_classes} species from taxonomy.csv")

    # --- Column order resolution ---
    if os.path.exists(SAMPLE_SUBMISSION_PATH):
        column_order = list(pd.read_csv(SAMPLE_SUBMISSION_PATH, nrows=0).columns)
        missing = [c for c in column_order if c != "row_id" and c not in label_order]
        if missing:
            print(f"Warning: {len(missing)} sample_submission columns aren't in taxonomy.csv "
                  f"label set: {missing[:5]}{'...' if len(missing) > 5 else ''}. They'll be filled with 0.")
    else:
        column_order = ["row_id"] + label_order

    # --- Load model checkpoint ---
    model = EfficientBirbNN(num_classes=num_classes, pretrained=False).to(DEVICE)
    # FIX: Added weights_only=False to ensure compatibility with PyTorch dictionary loading
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(best_val_auc={checkpoint.get('best_val_auc', 'n/a')})")

    test_files = sorted(glob.glob(os.path.join(TEST_SOUNDSCAPES_DIR, "*.ogg")))
    print(f"Found {len(test_files)} test soundscape files")

    # FIX: Using context manager 'with open(...)' to ensure file is safely flushed/closed
    with open(OUTPUT_PATH, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=column_order, restval=0.0, extrasaction='ignore')
        writer.writeheader()

        rows_written = 0
        with torch.inference_mode():
            for file_idx, audio_path in enumerate(tqdm(test_files, desc="Soundscapes")):
                filename = os.path.splitext(os.path.basename(audio_path))[0]

                try:
                    spec_batch, end_time_batch = [], []
                    for spectrogram, end_time in load_and_chunk(audio_path):
                        spec_batch.append(spectrogram)
                        end_time_batch.append(end_time)

                        if len(spec_batch) == INFER_BATCH_SIZE:
                            _run_batch_and_write(model, spec_batch, end_time_batch, filename, idx_to_label, num_classes, writer)
                            rows_written += len(spec_batch)
                            spec_batch, end_time_batch = [], []

                    if spec_batch:  # leftover partial batch for this file
                        _run_batch_and_write(model, spec_batch, end_time_batch, filename, idx_to_label, num_classes, writer)
                        rows_written += len(spec_batch)

                except Exception as e:
                    print(f"Skipping unreadable/corrupted file {audio_path}: {e}")
                    # FIX: Prevent Submission Scoring Errors by generating dummy rows for failed files
                    try:
                        info = torchaudio.info(audio_path)
                        duration_sec = info.num_frames / info.sample_rate
                        num_chunks = int(duration_sec // CLIP_LENGTH_SEC)
                        for idx in range(num_chunks):
                            end_time = (idx + 1) * int(CLIP_LENGTH_SEC)
                            # writing only row_id leverages restval=0.0 to safely pad the missing features
                            writer.writerow({"row_id": f"{filename}_{end_time}"}) 
                            rows_written += 1
                    except Exception as meta_e:
                        print(f"Could not read metadata for dummy row injection: {meta_e}")
                    continue

                # Periodically force garbage collection
                if (file_idx + 1) % GC_EVERY_N_FILES == 0:
                    gc.collect()

    print(f"Wrote {rows_written} rows to {OUTPUT_PATH}")
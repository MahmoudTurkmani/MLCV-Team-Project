# ==========================================================================
# BirdCLEF+ 2026 -- Kaggle Submission Notebook (CPU-only, Bird-MAE)
# ==========================================================================
# Adapted from the EfficientNet-B3 baseline to run the Bird-MAE classifier
# from train_birdmae.py (BirdMAEClassifier = frozen/fine-tuned Bird-MAE
# backbone + linear/MLP head). Everything about the CPU-safety pattern from
# the baseline is preserved (capped thread pools, streaming CSV writes,
# periodic gc.collect(), per-file try/except with dummy-row fallback) -- only
# the model and the preprocessing path change, because Bird-MAE consumes RAW
# WAVEFORMS and does its own mel extraction internally via its
# feature_extractor. Feeding it a manually-computed mel spectrogram (like the
# EfficientNet path did) would double-process the audio and produce garbage.
#
# BEFORE RUNNING -- read this whole block, Bird-MAE needs more setup than
# the EfficientNet baseline did:
#
#   1. Upload your fine-tuned checkpoint (*.pth) as a Kaggle Dataset and
#      attach it to this notebook. Set MODEL_CHECKPOINT_PATH below.
#
#   2. Set BIRDMAE_MODEL_SIZE / HEAD_TYPE / HIDDEN_DIM / DROPOUT to EXACTLY
#      match what you trained with. Unlike the checkpoint dict itself (which
#      only stores epoch/state_dict/best_val_auc -- no architecture info),
#      these are NOT recoverable from the .pth file. If they don't match,
#      load_state_dict() will throw a shape-mismatch error, or worse, silently
#      load into the wrong-sized head if you disable strict loading.
#
#   3. Kaggle competition notebooks run WITHOUT internet access. Bird-MAE is
#      pulled from the HuggingFace Hub via AutoModel.from_pretrained(), which
#      will fail offline. You must pre-download the model snapshot on a
#      machine WITH internet, then upload it as a second Kaggle Dataset:
#
#         from huggingface_hub import snapshot_download
#         snapshot_download(
#             repo_id="DBD-research-group/Bird-MAE-Base",   # match training size
#             local_dir="./birdmae_base_snapshot",
#         )
#         # zip ./birdmae_base_snapshot and upload it as a Kaggle Dataset
#
#      This must be a snapshot of the SAME size (Base/Large/Huge) used for
#      training -- attach it and set BIRDMAE_LOCAL_DIR below. trust_remote_code
#      needs the custom modeling_*.py files that snapshot_download pulls down
#      alongside the weights, so don't cherry-pick just the .bin/.safetensors.
#
#   4. Bird-MAE-Huge run frozen on CPU is slow per the same warning in
#      train_birdmae.py -- across ~600 test files this can matter a lot for
#      Kaggle's submission time limit. If your checkpoint was trained on
#      Bird-MAE-Base or -Large, prefer that for CPU inference. You cannot
#      swap sizes freely though: the checkpoint's weights were trained
#      against one specific backbone's embedding dimension.
#
#   5. Make sure CLIP_LENGTH_SEC matches what the checkpoint was trained with.
# ==========================================================================

import os

# Cap BLAS/OMP thread pools BEFORE importing torch/transformers.
# FIX: Changed to direct assignment to overwrite Kaggle's default container settings.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ADAPT: force offline mode before transformers is imported. Kaggle
# competition notebooks have no internet at submission time -- without this,
# AutoModel.from_pretrained()/AutoFeatureExtractor.from_pretrained() will try
# to hit the Hub, hang or error, and torch will never even start inference.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gc
import sys
import csv
import glob
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoFeatureExtractor
from tqdm import tqdm

torch.set_num_threads(4)  # adjust to match the notebook's actual CPU allocation

# ==========================================
# CONFIG -- adjust these for your setup
# ==========================================
# FIX: Corrected standard Kaggle competition directory path
COMP_DIR                = "/kaggle/input/competitions/birdclef-2026"
TEST_SOUNDSCAPES_DIR    = os.path.join(COMP_DIR, "test_soundscapes")
TAXONOMY_PATH           = os.path.join(COMP_DIR, "taxonomy.csv")
SAMPLE_SUBMISSION_PATH  = os.path.join(COMP_DIR, "sample_submission.csv")

# Point this at whichever fine-tuned checkpoint you attached as a Kaggle Dataset.
MODEL_CHECKPOINT_PATH = "/kaggle/input/datasets/nurmars/birdmae-base-mlp-ep8/birdmae-base-mlp-ep8.pth"

# ADAPT: local, offline snapshot of the Bird-MAE repo (see step 3 above).
# This is a directory path (attached Kaggle Dataset), NOT the Hub repo id --
# the Hub id only works when internet access is available.
BIRDMAE_LOCAL_DIR = "/kaggle/input/datasets/nurmars/birdmae-base-mlp-ep8/birdmae_base_snapshot"

# --- Architecture config: MUST match the training run that produced
#     MODEL_CHECKPOINT_PATH exactly (see point 2 above). ---
BIRDMAE_MODEL_SIZE = "Base"     # "Base", "Large", or "Huge" -- whatever was trained
HEAD_TYPE           = "mlp"  # "linear" or "mlp"
HIDDEN_DIM           = 512
DROPOUT              = 0.3

CLIP_LENGTH_SEC  = 5.0     # must match training
SAMPLE_RATE      = 32000   # must match training
# Bird-MAE's forward() loops the feature_extractor per-sample before doing
# ONE batched backbone pass (see BirdMAEClassifier docstring) -- batching
# still helps the backbone pass, but the per-sample extraction loop means
# gains are smaller than with a fully vectorized model, and each item in the
# batch costs real CPU time regardless of batch size. Kept smaller than the
# EfficientNet baseline's default for that reason.
INFER_BATCH_SIZE = 8
GC_EVERY_N_FILES = 25      # how often to force a garbage-collection pass

OUTPUT_PATH = "/kaggle/working/submission.csv"

DEVICE = torch.device("cpu")  # hardcoded: this competition disallows GPU


# ==========================================
# MODEL DEFINITION (must match train_birdmae.py's BirdMAEClassifier exactly)
# ==========================================
class BirdMAEClassifier(nn.Module):
    """
    Wraps a pretrained Bird-MAE backbone + its own feature_extractor into a
    standard classifier: raw waveform batch in, per-species logits out.

    Copied to match train_birdmae.py so the checkpoint's state_dict keys
    line up. See that file for the full design rationale.
    """

    def __init__(self, model_name="DBD-research-group/Bird-MAE-Huge",
                 head="linear", num_classes=None, sample_rate=32000, freeze_backbone=True,
                 hidden_dim=512, dropout=0.3):
        super().__init__()
        self.sample_rate = sample_rate
        self.freeze_backbone = freeze_backbone

        # ADAPT: local_files_only=True enforces the offline path explicitly
        # (belt-and-suspenders alongside the HF_HUB_OFFLINE env var above) and
        # fails fast with a clear error if BIRDMAE_LOCAL_DIR wasn't set up
        # correctly, rather than silently hanging trying to reach the Hub.
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        # low_cpu_mem_usage=False: this checkpoint's custom modeling code
        # (modeling_bird_mae.py) calls .item() on tensors during __init__,
        # which isn't safe under transformers' default fast-init (meta
        # device) path. Forcing the classic init path avoids that crash.
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=False,
            local_files_only=True
        )

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        embed_dim = self._infer_embed_dim()
        match head:
            case "linear":
                self.classifier = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, hidden_dim),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
            case "mlp":
                self.classifier = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
            case _:
                print("Classification head has to be either 'linear' or 'mlp'!!!")
                sys.exit(1)

    def _extract_spec(self, waveform_np):
        feat = self.feature_extractor(waveform_np, return_tensors="pt")
        if hasattr(feat, "keys"):
            key = "input_values" if "input_values" in feat else list(feat.keys())[0]
            return feat[key]
        return feat

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _infer_embed_dim(self):
        with torch.no_grad():
            dummy = np.zeros(int(self.sample_rate * 5), dtype=np.float32)
            spec = self._extract_spec(dummy)
            out = self.backbone(spec)
            emb = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
        return emb.shape[-1]

    def forward(self, waveform_batch):
        # waveform_batch: tensor [B, 1, T]
        device = waveform_batch.device
        specs = []
        for wf in waveform_batch:
            wf_np = wf.squeeze(0).detach().cpu().numpy().astype(np.float32)
            specs.append(self._extract_spec(wf_np))
        specs = torch.cat(specs, dim=0).to(device)

        with torch.set_grad_enabled(not self.freeze_backbone and self.training):
            out = self.backbone(specs)
        emb = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        if emb.dim() == 3:
            emb = emb.mean(dim=1)

        return self.classifier(emb)


# ==========================================
# AUDIO -> RAW WAVEFORM CHUNK HELPER
# ==========================================
# ADAPT: no mel_spect/AmplitudeToDB/normalization here anymore -- Bird-MAE's
# feature_extractor does its own mel extraction with its own pretrained
# normalization stats inside the model's forward(). Precomputing our own
# spectrogram (like the EfficientNet baseline did) would double-process the
# audio and silently produce wrong predictions.

def load_and_chunk(audio_path, clip_length_sec=CLIP_LENGTH_SEC, sample_rate=SAMPLE_RATE):
    """
    Loads one soundscape file and yields (raw_waveform_chunk, end_time_seconds)
    for each non-overlapping clip-length window. Plain generator over tensors,
    same chunking convention as the EfficientNet baseline (and as
    train_birdmae.py's __getitem__: pad/truncate to exactly chunk_size).
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
        chunk = waveform[:, start_idx:end_idx]  # [1, chunk_size], exactly what BirdMAEClassifier.forward expects per-item

        end_time = (idx + 1) * int(clip_length_sec)
        yield chunk, end_time

    del waveform  # explicit: drop the full waveform before the next file loads


def _run_batch_and_write(model, wf_batch, end_time_batch, filename, idx_to_label, num_classes, writer):
    """Stacks a list of single raw-waveform chunks into one batch, runs the
    model once, and immediately writes each resulting row to the CSV writer."""
    batch_tensor = torch.stack(wf_batch, dim=0).to(DEVICE)  # [B, 1, chunk_size]
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

    # --- Build model (loads Bird-MAE backbone + feature_extractor from the
    #     local offline snapshot, then the fine-tuned head) ---
    print(f"Loading Bird-MAE-{BIRDMAE_MODEL_SIZE} from local snapshot: {BIRDMAE_LOCAL_DIR}")
    model = BirdMAEClassifier(
        model_name=BIRDMAE_LOCAL_DIR,
        head=HEAD_TYPE,
        num_classes=num_classes,
        sample_rate=SAMPLE_RATE,
        freeze_backbone=True,   # inference-only: doesn't matter for grads, but keeps .train() calls (if any) pinned correctly
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

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
                    wf_batch, end_time_batch = [], []
                    for chunk, end_time in load_and_chunk(audio_path):
                        wf_batch.append(chunk)
                        end_time_batch.append(end_time)

                        if len(wf_batch) == INFER_BATCH_SIZE:
                            _run_batch_and_write(model, wf_batch, end_time_batch, filename, idx_to_label, num_classes, writer)
                            rows_written += len(wf_batch)
                            wf_batch, end_time_batch = [], []

                    if wf_batch:  # leftover partial batch for this file
                        _run_batch_and_write(model, wf_batch, end_time_batch, filename, idx_to_label, num_classes, writer)
                        rows_written += len(wf_batch)

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


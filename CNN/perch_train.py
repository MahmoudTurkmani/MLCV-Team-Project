# =============================================================================
# BirdCLEF+ 2026 — Perch v2 Embedding + MLP Head (TensorFlow/Keras)
# =============================================================================
#
# WHY PERCH INSTEAD OF TRAINING EFFICIENTNET FROM SCRATCH:
#   Perch v2 is a 12M-parameter EfficientNet-B3 pretrained specifically on
#   bird vocalizations from Xeno-Canto, iNaturalist, Animal Sound Archive,
#   and FSD50k — a far more relevant pretraining target than ImageNet.
#   It produces 1536-dimensional embeddings that already encode species-
#   discriminative acoustic features. Public 2026 competition notebooks score
#   ~0.925 with Perch + a simple MLP head, versus ~0.74 training a vanilla
#   EfficientNet from scratch on the same data.
#
# TWO-PHASE ARCHITECTURE:
#   Phase 1 (BUILD_CACHE=True): Run Perch v2 over every 5-second training
#     clip, optionally with N_AUGMENT_PASSES augmented copies, and save
#     (embedding, target) pairs to disk as compressed .npz files.
#     Run this ONCE on a GPU machine. Perch is frozen — nothing is trained.
#
#   Phase 2 (BUILD_CACHE=False): Load cached embeddings, train a lightweight
#     MLP head (1536 → 512 → 234). Very fast (~2 min/epoch even on CPU).
#     All W&B logging, early stopping, and validation coverage logic is
#     identical to the PyTorch EfficientNet script.
#
# PERCH MODEL:
#   GPU training (Phase 1): "perch_v2" requires TF >= 2.20.0rc0
#     pip install "tensorflow[and-cuda]~=2.20.0rc0"
#   CPU inference (Kaggle submission): "perch_v2_cpu" works with standard TF
#     Kaggle local path: /kaggle/input/bird-vocalization-classifier/
#                        tensorflow2/perch_v2_cpu/1
#   Hub URL: https://www.kaggle.com/models/google/bird-vocalization-classifier/
#            frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1
#
# USAGE:
#   1. Set COMP_DIR, PERCH_MODEL_PATH below
#   2. Run with BUILD_CACHE=True on GPU machine to extract embeddings
#   3. Run with BUILD_CACHE=False to train the MLP head
# =============================================================================

import os
import ast
import glob
import time
import math
import random
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import wandb
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Suppress TF verbose logging except errors
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# =============================================================================
# AUDIO UTILITIES (numpy equivalents of the PyTorch noise functions)
# =============================================================================

def load_clip(audio_path, start_sec, end_sec, target_sr=32000):
    """Load one clip segment from an audio file.
    Uses librosa so OGG/MP3/FLAC/WAV all work without codec juggling."""
    duration = end_sec - start_sec
    waveform, _ = librosa.load(
        audio_path, sr=target_sr, offset=start_sec, duration=duration, mono=True
    )
    target_len = int(target_sr * duration)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    return waveform[:target_len].astype(np.float32)


def add_pink_noise(waveform, snr_db=15):
    """Pink noise (1/f spectrum) in numpy."""
    white = np.random.randn(len(waveform)).astype(np.float32)
    fft   = np.fft.rfft(white)
    freqs = np.arange(1, len(fft) + 1, dtype=np.float32)
    fft  *= 1.0 / np.sqrt(freqs)
    pink  = np.fft.irfft(fft, n=len(waveform)).astype(np.float32)
    pink /= (pink.std() + 1e-8)
    sig_power   = np.mean(waveform ** 2)
    noise_power = np.mean(pink ** 2)
    if sig_power < 1e-7:
        return waveform
    factor = np.sqrt((sig_power / (noise_power + 1e-10)) * 10 ** (-snr_db / 10.0))
    return waveform + pink * factor


def add_white_noise(waveform, snr_db=15):
    noise = np.random.randn(len(waveform)).astype(np.float32)
    sig_power   = np.mean(waveform ** 2)
    noise_power = np.mean(noise ** 2)
    if sig_power < 1e-7:
        return waveform
    factor = np.sqrt((sig_power / (noise_power + 1e-10)) * 10 ** (-snr_db / 10.0))
    return waveform + noise * factor


def add_esc50_noise(waveform, esc50_clips, snr_db=15, target_sr=32000):
    """Mix a random ESC-50 clip into the waveform."""
    if not esc50_clips:
        return waveform
    noise_path = esc50_clips[np.random.randint(len(esc50_clips))]
    try:
        noise, _ = librosa.load(noise_path, sr=target_sr, mono=True)
    except Exception:
        return waveform
    noise = noise.astype(np.float32)
    target_len = len(waveform)
    if len(noise) >= target_len:
        start = np.random.randint(0, len(noise) - target_len + 1)
        noise = noise[start:start + target_len]
    else:
        reps  = (target_len // len(noise)) + 1
        noise = np.tile(noise, reps)[:target_len]
    sig_power   = np.mean(waveform ** 2)
    noise_power = np.mean(noise ** 2)
    if sig_power < 1e-7 or noise_power < 1e-7:
        return waveform
    factor = np.sqrt((sig_power / noise_power) * 10 ** (-snr_db / 10.0))
    return waveform + noise * factor


def pitch_shift(waveform, n_semitones, target_sr=32000):
    """Approximate pitch shift via librosa (phase vocoder)."""
    try:
        shifted = librosa.effects.pitch_shift(
            waveform, sr=target_sr, n_steps=n_semitones
        )
        return shifted[:len(waveform)].astype(np.float32)
    except Exception:
        return waveform


# =============================================================================
# LABEL UTILITIES
# =============================================================================

def make_target(row, label_to_idx, num_classes):
    """Soft multi-label target vector identical to BirbSet in the PyTorch script."""
    target = np.zeros(num_classes, dtype=np.float32)
    primary = label_to_idx.get(row['primary_label'])
    if primary is None:
        return target
    rating     = row.get('rating', 0)
    confidence = 1.0 if (pd.isna(rating) or rating == 0) else float(rating) / 5.0
    target[primary] = confidence
    raw_sec = row.get('secondary_labels', '[]')
    if raw_sec and raw_sec not in ('[]', '', None):
        try:
            for sec in ast.literal_eval(raw_sec):
                idx = label_to_idx.get(sec)
                if idx is not None:
                    target[idx] = confidence * 0.3
        except (ValueError, SyntaxError):
            pass
    return target


# =============================================================================
# PERCH v2 WRAPPER
# =============================================================================

class PerchEmbedder:
    """
    Thin wrapper around the Perch v2 SavedModel that handles:
      - Dynamic discovery of the serving_default input key (varies by version)
      - Batched embedding extraction from raw waveforms
      - Graceful fallback between infer_tf() method and serving_default signature

    Perch v2 expects float32 waveforms of shape (batch, 160000) at 32kHz.
    Output embedding has shape (batch, 1536).
    """
    EMBED_DIM  = 1536
    CLIP_LEN   = 5
    SAMPLE_RATE = 32000
    CLIP_SAMPLES = CLIP_LEN * SAMPLE_RATE  # 160 000

    def __init__(self, model_path_or_url):
        print(f"Loading Perch v2 from: {model_path_or_url}")
        self.model = hub.load(model_path_or_url)
        # Discover the inference callable and input key once at construction.
        self._signature, self._input_key = self._discover_api()
        print(f"Perch loaded. Input key: '{self._input_key}'  "
              f"Embedding dim: {self.EMBED_DIM}")

    def _discover_api(self):
        """Return (callable_signature, input_key_name)."""
        # Prefer the direct infer_tf method if it exists (Perch v1, SurfPerch)
        if hasattr(self.model, 'infer_tf'):
            return None, '__infer_tf__'
        # Fall back to serving_default signature (Perch v2 SavedModel)
        sig = self.model.signatures.get(
            "serving_default",
            next(iter(self.model.signatures.values()), None)
        )
        if sig is None:
            raise RuntimeError("Could not discover a callable signature on the Perch model.")
        input_key = list(sig.structured_input_signature[1].keys())[0]
        return sig, input_key

    def embed(self, waveforms_np):
        """
        Args:
            waveforms_np: numpy array of shape (batch, 160000), float32, 32kHz

        Returns:
            embeddings: numpy array of shape (batch, 1536)
        """
        waveforms_tf = tf.constant(waveforms_np, dtype=tf.float32)
        if self._input_key == '__infer_tf__':
            out = self.model.infer_tf(waveforms_tf)
            # infer_tf may return a tuple (logits, embeddings) or a dict
            if isinstance(out, dict):
                return out['embedding'].numpy()
            return out[1].numpy()
        else:
            out = self._signature(**{self._input_key: waveforms_tf})
            return out['embedding'].numpy()


# =============================================================================
# PHASE 1: EMBEDDING EXTRACTION
# =============================================================================

def extract_and_cache(df, audio_root, embedder, label_to_idx, num_classes,
                       clip_length=5.0, sample_rate=32000,
                       n_augment_passes=0,
                       use_pink_noise=False, use_white_noise=False,
                       use_esc50_noise=False, esc50_path=None,
                       use_pitch_shift=False,
                       batch_size=64, cache_dir="./embedding_cache"):
    """
    Pre-extract Perch v2 embeddings from all training clips and save them.
    This function is called once (Phase 1) and the results reused in Phase 2.

    For each audio file, clips are extracted at non-overlapping clip_length
    windows. If n_augment_passes > 0, additional augmented copies are produced
    -- one copy per pass, each with a fresh random augmentation draw. This
    multiplies the effective training set size by (1 + n_augment_passes).

    Args:
        n_augment_passes: number of additional augmented extraction passes.
                          0 = clean only, 2 = clean + 2 augmented copies.
    """
    os.makedirs(cache_dir, exist_ok=True)
    esc50_clips = []
    if use_esc50_noise and esc50_path:
        esc50_clips = glob.glob(os.path.join(esc50_path, "audio", "*.wav"))
        print(f"ESC-50: {len(esc50_clips)} clips found.")

    # Tally passes: 0 = clean, 1..N = augmented
    total_passes = 1 + n_augment_passes
    all_embeddings = []
    all_targets    = []

    for pass_idx in range(total_passes):
        pass_label = "clean" if pass_idx == 0 else f"aug_pass_{pass_idx}"
        cache_path = os.path.join(cache_dir, f"embeddings_{pass_label}.npz")

        if os.path.exists(cache_path):
            print(f"[{pass_label}] Cache found — loading: {cache_path}")
            data = np.load(cache_path)
            all_embeddings.append(data['embeddings'])
            all_targets.append(data['targets'])
            continue

        print(f"\n[{pass_label}] Extracting embeddings for {len(df)} rows...")
        is_augment_pass = pass_idx > 0

        clip_embeddings = []
        clip_targets    = []
        batch_waves, batch_tgts = [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Files [{pass_label}]"):
            audio_path = os.path.join(audio_root, os.path.normpath(row['filename']))
            try:
                info_duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"  Skipping {audio_path}: {e}")
                continue

            target = make_target(row, label_to_idx, num_classes)
            clip_samples = int(sample_rate * clip_length)  # 160 000 for 5s @ 32kHz
            pos = 0.0
            while pos < info_duration:
                end_pos = min(pos + clip_length, info_duration)
                try:
                    waveform = load_clip(audio_path, pos, end_pos, target_sr=sample_rate)
                except Exception as e:
                    print(f"  Skipping clip at {pos:.1f}s in {audio_path}: {e}")
                    pos += clip_length
                    continue

                # Normalise to exactly clip_samples regardless of whether this
                # is a full-length clip or a short tail at the end of the file.
                # Perch expects every input to be (batch, 160000) -- any length
                # mismatch causes np.stack to raise, which was the crash here.
                if len(waveform) < clip_samples:
                    waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
                elif len(waveform) > clip_samples:
                    waveform = waveform[:clip_samples]

                if is_augment_pass:
                    # Pick one noise type randomly from the enabled ones
                    active_noise_fns = []
                    if use_pink_noise:
                        active_noise_fns.append('pink')
                    if use_white_noise:
                        active_noise_fns.append('white')
                    if use_esc50_noise and esc50_clips:
                        active_noise_fns.append('esc50')

                    if active_noise_fns and np.random.rand() < 0.5:
                        noise_type = active_noise_fns[np.random.randint(len(active_noise_fns))]
                        snr = np.random.uniform(10.0, 25.0)
                        if noise_type == 'pink':
                            waveform = add_pink_noise(waveform, snr_db=snr)
                        elif noise_type == 'white':
                            waveform = add_white_noise(waveform, snr_db=snr)
                        elif noise_type == 'esc50':
                            waveform = add_esc50_noise(waveform, esc50_clips, snr_db=snr)

                    if use_pitch_shift and np.random.rand() < 0.5:
                        semitones = np.random.uniform(-3.0, 3.0)
                        waveform  = pitch_shift(waveform, semitones, target_sr=sample_rate)

                batch_waves.append(waveform)
                batch_tgts.append(target)

                if len(batch_waves) == batch_size:
                    embs = embedder.embed(np.stack(batch_waves, axis=0))
                    clip_embeddings.append(embs)
                    clip_targets.extend(batch_tgts)
                    batch_waves, batch_tgts = [], []

                pos += clip_length

        # Final partial batch
        if batch_waves:
            embs = embedder.embed(np.stack(batch_waves, axis=0))
            clip_embeddings.append(embs)
            clip_targets.extend(batch_tgts)

        embeddings_arr = np.concatenate(clip_embeddings, axis=0).astype(np.float32)
        targets_arr    = np.array(clip_targets, dtype=np.float32)

        np.savez_compressed(cache_path, embeddings=embeddings_arr, targets=targets_arr)
        print(f"[{pass_label}] Saved {len(embeddings_arr)} embeddings → {cache_path}")

        all_embeddings.append(embeddings_arr)
        all_targets.append(targets_arr)

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_targets    = np.concatenate(all_targets, axis=0)
    print(f"\nTotal embeddings: {final_embeddings.shape}  "
          f"Targets: {final_targets.shape}")
    return final_embeddings, final_targets


# =============================================================================
# VALIDATION COVERAGE FIX (identical logic to PyTorch script)
# =============================================================================

def ensure_val_coverage(full_df, df_train, df_val,
                         label_col='primary_label', filename_col='filename'):
    val_classes = set(df_val[label_col].unique())
    all_classes = set(full_df[label_col].unique())
    missing     = sorted(all_classes - val_classes)

    if not missing:
        print(f"✅ Val class coverage: {len(val_classes)}/{len(all_classes)} (100%)")
        return df_train, df_val

    print(f"⚠️  {len(missing)} classes missing from val — moving one file each...")
    indices_to_move = []
    for cls in missing:
        candidate_files = df_train.loc[df_train[label_col] == cls, filename_col].unique()
        if len(candidate_files) == 0:
            continue
        file_sizes  = {f: int((df_train[filename_col] == f).sum()) for f in candidate_files}
        chosen_file = max(file_sizes, key=file_sizes.get)
        indices_to_move.extend(df_train.index[df_train[filename_col] == chosen_file].tolist())

    if indices_to_move:
        move_mask  = df_train.index.isin(indices_to_move)
        rows_moved = df_train[move_mask].copy()
        df_train   = df_train[~move_mask].reset_index(drop=True)
        df_val     = pd.concat([df_val, rows_moved], ignore_index=True)

    print(f"✅ Val coverage after fix: {len(df_val[label_col].unique())}/{len(all_classes)}")
    return df_train, df_val


# =============================================================================
# PHASE 2: MLP HEAD
# =============================================================================

def build_mlp_head(num_classes, embed_dim=1536, hidden_dims=(512, 256),
                    dropout=0.3):
    """
    Lightweight MLP that maps Perch embeddings to per-class logits.

    Two hidden layers with BatchNorm + ReLU + Dropout is the most common
    configuration in top BirdCLEF solutions using Perch embeddings. The head
    is deliberately small — Perch's representations are already highly
    discriminative, so a large head tends to overfit rather than improve.

    Returns a Keras functional model.
    """
    inputs = tf.keras.Input(shape=(embed_dim,), name="embedding")
    x = inputs
    for units in hidden_dims:
        x = tf.keras.layers.Dense(units, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="PerchMLPHead")


# =============================================================================
# FOCAL LOSS (TF equivalent of the PyTorch version)
# =============================================================================

class FocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss for multi-label classification.

    Identical formulation to the PyTorch FocalLoss — gamma=1.0 default,
    float32 cast to avoid float16 numerical issues, p_t clamped to prevent
    NaN gradients from (1-p_t)^gamma at the boundary.
    """
    def __init__(self, gamma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred_logits):
        y_true        = tf.cast(y_true, tf.float32)
        y_pred_logits = tf.cast(y_pred_logits, tf.float32)
        bce    = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred_logits)
        probs  = tf.sigmoid(y_pred_logits)
        p_t    = probs * y_true + (1.0 - probs) * (1.0 - y_true)
        p_t    = tf.clip_by_value(p_t, 1e-6, 1.0 - 1e-6)
        return tf.reduce_mean((1.0 - p_t) ** self.gamma * bce)


# =============================================================================
# MIXUP FOR EMBEDDINGS
# =============================================================================

def apply_mixup(x_batch, y_batch, alpha=0.4):
    """
    MixUp on embedding + target pairs.
    At embedding level, MixUp is just a linear interpolation -- no rectangular
    patch logic needed (that's CutMix, which doesn't have a clean equivalent
    for 1D embedding vectors). Keeps implementation trivial and effective.
    """
    batch_size = tf.shape(x_batch)[0]
    lam        = float(np.random.beta(alpha, alpha))
    perm       = tf.random.shuffle(tf.range(batch_size))
    x_mixed    = lam * x_batch + (1.0 - lam) * tf.gather(x_batch, perm)
    y_mixed    = lam * y_batch + (1.0 - lam) * tf.gather(y_batch, perm)
    return x_mixed, y_mixed


# =============================================================================
# TRAINING LOOP
# =============================================================================

@tf.function
def train_step(model, x_batch, y_batch, optimizer, criterion):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss   = criterion(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    # Gradient clipping (same max_norm=1.0 as the PyTorch script)
    grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, global_norm


def train_epoch(model, embeddings, targets, sample_weights, optimizer, criterion,
                epoch, batch_size, use_mixup=False, mixup_alpha=0.4):
    num_samples = len(embeddings)
    indices = np.random.choice(
        num_samples, size=num_samples, replace=True,
        p=sample_weights / sample_weights.sum()
    )
    total_loss    = 0.0
    total_gnorm   = 0.0
    nan_batches   = 0
    num_batches   = 0

    pbar = tqdm(range(0, num_samples, batch_size),
                desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)

    for start in pbar:
        batch_idx = indices[start:start + batch_size]
        x_batch   = tf.constant(embeddings[batch_idx], dtype=tf.float32)
        y_batch   = tf.constant(targets[batch_idx],    dtype=tf.float32)

        if use_mixup:
            x_batch, y_batch = apply_mixup(x_batch, y_batch, alpha=mixup_alpha)

        loss, gnorm = train_step(model, x_batch, y_batch, optimizer, criterion)
        loss_val  = float(loss)
        gnorm_val = float(gnorm)

        if math.isnan(loss_val) or math.isnan(gnorm_val):
            nan_batches += 1
        else:
            total_loss  += loss_val
            total_gnorm += gnorm_val

        num_batches += 1
        pbar.set_postfix({
            'loss':      f"{loss_val:.4f}",
            'gnorm':     f"{gnorm_val:.3f}" if not math.isnan(gnorm_val) else "NaN",
            'nan_b':     nan_batches,
        })

    valid_batches = num_batches - nan_batches
    return (total_loss / valid_batches if valid_batches > 0 else float('nan'),
            total_gnorm / valid_batches if valid_batches > 0 else float('nan'),
            nan_batches)


def validate_epoch(model, embeddings, targets, criterion, epoch, batch_size=256):
    """
    Runs validation, computing ROC-AUC with the same class-coverage logic as
    the PyTorch script: only classes with both a positive and negative example
    are included in the macro average, with diagnostics logged for the rest.
    """
    num_samples = len(embeddings)
    all_preds, all_targets = [], []
    total_loss = 0.0

    pbar = tqdm(range(0, num_samples, batch_size),
                desc=f"Epoch {epoch} [Val]", dynamic_ncols=True, leave=False)

    for start in pbar:
        x_batch = tf.constant(embeddings[start:start + batch_size], dtype=tf.float32)
        y_batch = tf.constant(targets[start:start + batch_size],    dtype=tf.float32)
        logits  = model(x_batch, training=False)
        loss    = criterion(y_batch, logits)
        probs   = tf.sigmoid(logits).numpy()
        all_preds.append(probs)
        all_targets.append(y_batch.numpy())
        total_loss += float(loss)
        pbar.set_postfix({'loss': f"{float(loss):.4f}"})

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    binary_targets = (all_targets > 0.0).astype(int)

    has_positive  = np.any(binary_targets == 1, axis=0)
    has_negative  = np.any(binary_targets == 0, axis=0)
    valid_classes = has_positive & has_negative
    num_total     = binary_targets.shape[1]
    num_valid     = int(valid_classes.sum())

    per_class_auc = None
    val_auc = 0.0
    if num_valid > 0:
        try:
            val_auc = roc_auc_score(
                binary_targets[:, valid_classes],
                all_preds[:, valid_classes],
                average='macro'
            )
            per_class_auc = roc_auc_score(
                binary_targets[:, valid_classes],
                all_preds[:, valid_classes],
                average=None
            )
        except ValueError as e:
            print(f"ROC-AUC error: {e}")

    scored_targets = binary_targets[:, valid_classes].astype(bool) if num_valid > 0 else None
    scored_preds   = all_preds[:, valid_classes] if num_valid > 0 else None
    mean_prob_pos  = float(scored_preds[scored_targets].mean()) if (scored_targets is not None and scored_targets.any()) else float('nan')
    mean_prob_neg  = float(scored_preds[~scored_targets].mean()) if (scored_targets is not None and (~scored_targets).any()) else float('nan')

    extra = {
        "num_valid_classes": num_valid,
        "num_skipped_classes": num_total - num_valid,
        "auc_min": float(per_class_auc.min()) if per_class_auc is not None else float('nan'),
        "auc_max": float(per_class_auc.max()) if per_class_auc is not None else float('nan'),
        "auc_std": float(per_class_auc.std()) if per_class_auc is not None else float('nan'),
        "mean_prob_positive": mean_prob_pos,
        "mean_prob_negative": mean_prob_neg,
    }
    num_batches = math.ceil(num_samples / batch_size)
    return total_loss / num_batches, val_auc, extra


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    COMP_DIR      = os.path.join("..", "birdclef-2026")
    AUDIO_DIR     = os.path.join(COMP_DIR, "train_audio")
    ESC50_PATH    = os.path.join("..", "ESC-50-master")

    # Perch model path / URL. Options:
    #   GPU:  "/path/to/bird-vocalization-classifier/tensorflow2/perch_v2/1"
    #   CPU:  "/path/to/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
    #   Hub:  "https://www.kaggle.com/models/google/bird-vocalization-classifier/
    #          frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1"
    PERCH_MODEL_PATH = "perch_v2_cpu"

    # Phase control
    BUILD_CACHE       = False   # True → run Phase 1 (embedding extraction)
                               # False → skip to Phase 2 (head training)
    CACHE_DIR         = "./embedding_cache"
    N_AUGMENT_PASSES  = 2      # 0 = clean only; 2 = clean + 2 augmented copies

    # Augmentation flags (applied during Phase 1 augment passes)
    USE_PINK_NOISE    = False
    USE_WHITE_NOISE   = False
    USE_ESC50_NOISE   = False
    USE_PITCH_SHIFT   = False

    # Phase 2 training
    MAX_EPOCHS   = 50
    PATIENCE     = 15
    BATCH_SIZE   = 256
    LEARNING_RATE = 1e-3    # MLP head trains much faster than a full CNN backbone;
                            # 1e-3 is a good starting point, cosine decays to 1e-5
    FOCAL_GAMMA  = 1.0
    USE_MIXUP    = False
    MIXUP_ALPHA  = 0.4

    # -------------------------------------------------------------------------
    # NAMING: same automatic tag system as the PyTorch script so runs never
    # collide in W&B and checkpoints never overwrite each other
    # -------------------------------------------------------------------------
    tag_parts = ["perch"]
    if USE_PINK_NOISE:   tag_parts.append("pink")
    if USE_WHITE_NOISE:  tag_parts.append("white")
    if USE_ESC50_NOISE:  tag_parts.append("esc50")
    if USE_PITCH_SHIFT:  tag_parts.append("pitch")
    if USE_MIXUP:        tag_parts.append("mixup")
    tag_parts.append(f"aug{N_AUGMENT_PASSES}")
    run_tag         = "_".join(tag_parts)
    checkpoint_path = f"best_perch_head_{run_tag}.weights.h5"

    # -------------------------------------------------------------------------
    # DATA SETUP
    # -------------------------------------------------------------------------
    full_df       = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
    taxonomy_df   = pd.read_csv(os.path.join(COMP_DIR, "taxonomy.csv"))
    label_order   = list(taxonomy_df['primary_label'].unique())
    label_to_idx  = {l: i for i, l in enumerate(label_order)}
    num_classes   = len(label_to_idx)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(
        sgkf.split(X=full_df, y=full_df['primary_label'], groups=full_df['filename'])
    )
    df_train = full_df.iloc[train_idx].reset_index(drop=True)
    df_val   = full_df.iloc[val_idx].reset_index(drop=True)

    df_train, df_val = ensure_val_coverage(full_df, df_train, df_val)
    print(f"Train rows: {len(df_train)}  Val rows: {len(df_val)}")

    # -------------------------------------------------------------------------
    # PHASE 1 — EMBEDDING EXTRACTION
    # -------------------------------------------------------------------------
    if BUILD_CACHE:
        embedder = PerchEmbedder(PERCH_MODEL_PATH)

        print("\n--- Extracting TRAIN embeddings ---")
        train_emb, train_tgt = extract_and_cache(
            df=df_train, audio_root=AUDIO_DIR, embedder=embedder,
            label_to_idx=label_to_idx, num_classes=num_classes,
            n_augment_passes=N_AUGMENT_PASSES,
            use_pink_noise=USE_PINK_NOISE, use_white_noise=USE_WHITE_NOISE,
            use_esc50_noise=USE_ESC50_NOISE, esc50_path=ESC50_PATH,
            use_pitch_shift=USE_PITCH_SHIFT,
            batch_size=64, cache_dir=os.path.join(CACHE_DIR, "train"),
        )
        print("\n--- Extracting VAL embeddings (clean only, no augmentation) ---")
        val_emb, val_tgt = extract_and_cache(
            df=df_val, audio_root=AUDIO_DIR, embedder=embedder,
            label_to_idx=label_to_idx, num_classes=num_classes,
            n_augment_passes=0,  # always clean for validation
            batch_size=64, cache_dir=os.path.join(CACHE_DIR, "val"),
        )
        print("\nPhase 1 complete. Re-run with BUILD_CACHE=False to train the head.")

    else:
        # -------------------------------------------------------------------------
        # LOAD CACHED EMBEDDINGS
        # -------------------------------------------------------------------------
        print("Loading cached embeddings...")
        train_parts = sorted(glob.glob(os.path.join(CACHE_DIR, "train", "*.npz")))
        val_parts   = sorted(glob.glob(os.path.join(CACHE_DIR, "val",   "*.npz")))
        if not train_parts or not val_parts:
            raise FileNotFoundError(
                f"No .npz cache files found in {CACHE_DIR}. "
                "Run with BUILD_CACHE=True first."
            )
        train_emb = np.concatenate([np.load(p)['embeddings'] for p in train_parts], axis=0)
        train_tgt = np.concatenate([np.load(p)['targets']    for p in train_parts], axis=0)
        val_emb   = np.concatenate([np.load(p)['embeddings'] for p in val_parts],   axis=0)
        val_tgt   = np.concatenate([np.load(p)['targets']    for p in val_parts],   axis=0)
        print(f"Train: {train_emb.shape}  Val: {val_emb.shape}")

        # -------------------------------------------------------------------------
        # WEIGHTED SAMPLER (inverse frequency, identical rationale to PyTorch script)
        # -------------------------------------------------------------------------
        primary_labels = np.argmax(train_tgt, axis=1)
        label_counts   = Counter(primary_labels.tolist())
        sample_weights = np.array(
            [1.0 / label_counts[l] for l in primary_labels], dtype=np.float32
        )

        # -------------------------------------------------------------------------
        # MODEL + OPTIMISER + LOSS
        # -------------------------------------------------------------------------
        run = wandb.init(
            entity="pumpkin_person-tu-dresden",
            project="CNN-Birds",
            name=f"Perch_{run_tag}",
            config={
                "architecture": "Perch_v2 + MLP",
                "dataset": "BirdClef+ 2026",
                "n_augment_passes": N_AUGMENT_PASSES,
                "use_pink_noise": USE_PINK_NOISE,
                "use_white_noise": USE_WHITE_NOISE,
                "use_esc50_noise": USE_ESC50_NOISE,
                "use_pitch_shift": USE_PITCH_SHIFT,
                "use_mixup": USE_MIXUP,
                "mixup_alpha": MIXUP_ALPHA,
                "focal_gamma": FOCAL_GAMMA,
                "learning_rate": LEARNING_RATE,
                "epochs": MAX_EPOCHS,
            }
        )

        model     = build_mlp_head(num_classes)
        criterion = FocalLoss(gamma=FOCAL_GAMMA)

        # Warmup 3 epochs → cosine decay (same schedule as the PyTorch script)
        WARMUP_EPOCHS = 3
        total_steps   = math.ceil(len(train_emb) / BATCH_SIZE) * MAX_EPOCHS
        warmup_steps  = math.ceil(len(train_emb) / BATCH_SIZE) * WARMUP_EPOCHS

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=total_steps - warmup_steps,
            alpha=1e-5 / LEARNING_RATE,
        )
        # Wrap with linear warmup manually via a lambda
        def warmup_cosine(step):
            step = tf.cast(step, tf.float32)
            warmup = tf.minimum(step / max(warmup_steps, 1), 1.0)
            return LEARNING_RATE * warmup * lr_schedule(tf.maximum(step - warmup_steps, 0))

        optimizer  = tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=1e-4
        )

        # -------------------------------------------------------------------------
        # TRAINING LOOP
        # -------------------------------------------------------------------------
        best_val_auc     = -1.0
        patience_counter = 0

        print("Starting head training...")
        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_start = time.time()
            train_loss, avg_gnorm, nan_batches = train_epoch(
                model, train_emb, train_tgt, sample_weights,
                optimizer, criterion, epoch, BATCH_SIZE,
                use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA,
            )
            val_loss, val_auc, val_extra = validate_epoch(
                model, val_emb, val_tgt, criterion, epoch, batch_size=512
            )
            epoch_time = time.time() - epoch_start
            current_lr = float(optimizer.learning_rate)

            if nan_batches > 0:
                print(f"⚠️  {nan_batches} NaN batches this epoch")
            print(
                f"Epoch {epoch} → Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"ROC-AUC: {val_auc:.4f} | GNorm: {avg_gnorm:.3f} | "
                f"Classes: {val_extra['num_valid_classes']}/{num_classes} | "
                f"Time: {epoch_time:.1f}s"
            )
            run.log({
                "Training Loss": train_loss,
                "Val Loss": val_loss,
                "Val ROC-AUC": val_auc,
                "Grad Norm (avg)": avg_gnorm,
                "NaN Grad Batches": nan_batches,
                "Learning Rate": current_lr,
                "Patience": patience_counter / PATIENCE,
                "Epoch Time (s)": epoch_time,
                "Val Classes Scored": val_extra["num_valid_classes"],
                "Val Classes Skipped": val_extra["num_skipped_classes"],
                "Val AUC Min (per-class)": val_extra["auc_min"],
                "Val AUC Max (per-class)": val_extra["auc_max"],
                "Val AUC Std (per-class)": val_extra["auc_std"],
                "Val Mean Prob (positives)": val_extra["mean_prob_positive"],
                "Val Mean Prob (negatives)": val_extra["mean_prob_negative"],
            })

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # NaN weight guard
                has_nan = any(
                    np.isnan(w.numpy()).any() for w in model.trainable_variables
                )
                if has_nan:
                    print("⛔ NaN in weights — skipping checkpoint.")
                else:
                    model.save_weights(checkpoint_path)
                    print(f"--> 🔥 New Best Saved (ROC-AUC: {best_val_auc:.4f})")
                    artifact = wandb.Artifact(
                        name=f"perch_head_{run_tag}", type="model",
                        metadata={"epoch": epoch, "val_auc": best_val_auc}
                    )
                    artifact.add_file(checkpoint_path)
                    run.log_artifact(artifact)
            else:
                patience_counter += 1
                print(f"--> No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch}.")
                break

        print(f"\nTraining complete. Best Val ROC-AUC: {best_val_auc:.4f}")
        run.finish()
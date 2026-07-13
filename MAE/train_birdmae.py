import os
import ast
import glob
import time
import math
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
import wandb
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, update_bn as swa_update_bn
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm  # Changed from tqdm.notebook for terminal compatibility
import soundfile

# ==========================================
# DATASET CLASSES
# ==========================================
import os
import ast
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def _parse_timestamp_to_seconds(ts):
    if pd.isna(ts):
        return 0.0
    
    ts_str = str(ts).strip()
    if ":" in ts_str:
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
    
    return float(ts_str)

class BirbSet(Dataset):
    def __init__(self, df, root, clip_length, label_to_idx, is_train=False,
                 use_pink_noise=False, use_white_noise=False, use_spec_augment=False,
                 use_pitch_shift=False, use_esc50_noise=False, esc50_path=None,
                 esc50_categories=None, n_mels=128, n_fft=1024,
                 soundscape_clips_df=None,
                 raw_waveform_output=False):
        self.clips            = []
        self.start_times      = []
        self.end_times        = []
        self.labels           = []
        self.secondary_labels = []
        self.ratings          = []

        self.clip_length      = clip_length   
        self.sample_rate      = 32000         
        self.label_to_idx     = label_to_idx
        self.is_train         = is_train
        self.root             = root
        
        # --- Augmentation Toggle Flags ---
        self.use_pink_noise   = use_pink_noise
        self.use_white_noise  = use_white_noise
        self.use_spec_augment = use_spec_augment
        self.use_pitch_shift  = use_pitch_shift
        self.use_esc50_noise  = use_esc50_noise

        # When True, __getitem__ returns the raw (augmented) waveform instead
        # of a computed mel spectrogram. Needed for Bird-MAE, whose own
        # feature_extractor does its own mel extraction with its own
        # pretrained normalization stats -- feeding it OUR spectrogram would
        # be double-processed and wrong. SpecAugment operates on a mel
        # spectrogram we no longer produce here, so force it off.
        self.raw_waveform_output = raw_waveform_output
        if raw_waveform_output:
            self.use_spec_augment = False

        # Pre-index ESC-50 clip paths once at construction so __getitem__ just
        # picks a random path rather than globbing on every sample access.
        # When esc50_categories is provided, only clips whose category matches
        # an entry in the list are used; the filter is applied via the ESC-50
        # metadata CSV (meta/esc50.csv), which maps filename → category.
        self.esc50_clips = []
        if self.is_train and self.use_esc50_noise:
            if esc50_path is not None:
                esc50_audio_dir = os.path.join(esc50_path, "audio")
                meta_csv        = os.path.join(esc50_path, "meta", "esc50.csv")

                if esc50_categories and os.path.exists(meta_csv):
                    # --- Category-filtered loading ---
                    meta_df = pd.read_csv(meta_csv)
                    # ESC-50 categories use underscores (e.g. "sea_waves").
                    # Normalise the user's list the same way to make matching
                    # case- and separator-insensitive.
                    norm = lambda s: s.lower().replace(" ", "_").replace("-", "_")
                    allowed = {norm(c) for c in esc50_categories}
                    meta_df['_norm_cat'] = meta_df['category'].apply(norm)
                    filtered = meta_df[meta_df['_norm_cat'].isin(allowed)]

                    self.esc50_clips = [
                        os.path.join(esc50_audio_dir, fname)
                        for fname in filtered['filename']
                        if os.path.exists(os.path.join(esc50_audio_dir, fname))
                    ]

                    found_cats  = sorted(filtered['category'].unique().tolist())
                    missed_cats = [c for c in esc50_categories
                                   if norm(c) not in {norm(f) for f in found_cats}]

                    if self.esc50_clips:
                        print(f"ESC-50 noise enabled: {len(self.esc50_clips)} clips "
                              f"from {len(found_cats)} category/ies: {found_cats}")
                    else:
                        print(f"Warning: ESC-50 category filter matched 0 clips. "
                              f"Check spelling against meta/esc50.csv. "
                              f"Requested: {esc50_categories}")
                    if missed_cats:
                        print(f"  ⚠️  Categories not found in ESC-50 metadata: {missed_cats}")

                elif esc50_categories and not os.path.exists(meta_csv):
                    # Metadata CSV missing — fall back to loading everything
                    # and warn loudly so the user knows filtering didn't apply.
                    print(f"Warning: esc50_categories requested but metadata CSV "
                          f"not found at {meta_csv}. Falling back to ALL clips.")
                    self.esc50_clips = glob.glob(os.path.join(esc50_audio_dir, "*.wav"))

                else:
                    # No category filter — use all clips as before
                    self.esc50_clips = glob.glob(os.path.join(esc50_audio_dir, "*.wav"))
                    if self.esc50_clips:
                        print(f"ESC-50 noise enabled: {len(self.esc50_clips)} clips "
                              f"(no category filter)")

                if not self.esc50_clips:
                    print(f"Warning: no usable ESC-50 clips at {esc50_audio_dir}. "
                          f"ESC-50 noise disabled.")
            else:
                print("Warning: use_esc50_noise=True but esc50_path=None. "
                      "ESC-50 noise disabled.")

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        self.mel_spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        
        # Only initialize masking modules if we are training AND SpecAugment is enabled
        if self.is_train and self.use_spec_augment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=24)
        
        for _, entry in df.iterrows():
            curr_audio_loc = os.path.join(self.root, os.path.normpath(entry["filename"]))
            try:
                info = soundfile.info(curr_audio_loc)
                duration = info.duration
            except Exception as e:
                print(f"Skipping metadata read error for {curr_audio_loc}: {e}")
                continue
            
            pos = 0.0
            while pos < duration:
                end_pos = min(pos + self.clip_length, duration)
                
                self.clips.append(curr_audio_loc)
                self.labels.append(self.label_to_idx[entry['primary_label']])
                self.ratings.append(entry.get('rating'))
                self.secondary_labels.append(entry.get('secondary_labels', '[]'))
                
                self.start_times.append(pos)                     
                self.end_times.append(end_pos)     
                
                pos += self.clip_length

        # --- Soundscape clip injection ---
        # Each row in soundscape_clips_df is already a single annotated 5-second
        # window (start_time/end_time come from the competition labels rather than
        # being derived by uniform chunking). We add them directly to the internal
        # clip lists using the exact annotated boundaries, so __getitem__ loads
        # the precise window that was labelled rather than an arbitrary alignment.
        if soundscape_clips_df is not None and len(soundscape_clips_df) > 0:
            skipped = 0
            for _, clip in soundscape_clips_df.iterrows():
                if clip['primary_label'] not in self.label_to_idx:
                    skipped += 1
                    continue
                self.clips.append(clip['audio_path'])
                self.labels.append(self.label_to_idx[clip['primary_label']])
                self.ratings.append(clip.get('rating', np.nan))
                self.secondary_labels.append(clip.get('secondary_labels', '[]'))
                self.start_times.append(_parse_timestamp_to_seconds(clip['start_time']))
                self.end_times.append(_parse_timestamp_to_seconds(clip['end_time']))
            added = len(soundscape_clips_df) - skipped
            print(f"BirbSet: added {added} soundscape clips "
                  f"({skipped} skipped — label not in taxonomy).")

    def __len__(self):
        return len(self.clips)
        
    def _add_pink_noise(self, waveform, snr_db=10):
        """Generates true Pink Noise (1/f spectrum) and blends it with the waveform."""
        white_noise = torch.randn_like(waveform)
        X_white = torch.fft.rfft(white_noise)
        
        freqs = torch.arange(1, X_white.shape[-1] + 1, device=waveform.device)
        multiplier = 1.0 / torch.sqrt(freqs)
        X_pink = X_white * multiplier
        
        pink_noise = torch.fft.irfft(X_pink, n=waveform.shape[-1])
        pink_noise = pink_noise / torch.std(pink_noise)
        
        sig_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(pink_noise ** 2)
        
        if sig_power < 1e-7:
            return waveform
            
        factor = torch.sqrt((sig_power / noise_power) * (10 ** (-snr_db / 10.0)))
        return waveform + pink_noise * factor

    def _add_white_noise(self, waveform, snr_db=10):
        """Adds white Gaussian noise at the target SNR.

        White noise has a flat power spectrum -- it's a useful baseline because
        it degrades all frequency bands equally and is computationally trivial.
        In practice it tends to be a weaker augmentation than pink or ESC-50
        noise for bird recordings, but worth including when combining noise types
        since the random picker keeps total noise probability fixed regardless of
        how many types are active.
        """
        noise = torch.randn_like(waveform)
        sig_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        if sig_power < 1e-7:
            return waveform
        factor = torch.sqrt((sig_power / noise_power) * (10 ** (-snr_db / 10.0)))
        return waveform + noise * factor

    def _pitch_shift(self, waveform, n_semitones):
        """Approximate pitch shift via the resampling trick.

        Resamples the waveform as if it had been captured at sample_rate*factor,
        then converts back to sample_rate. This changes pitch proportionally to
        factor and also slightly changes duration (by the same factor), which is
        accepted for data augmentation since the mel spectrogram captures the
        pitch contour regardless. Avoids the phase-vocoder overhead of a true
        PSOLA or time-stretch approach.

        At ±3 semitones the duration change is ~19%, which is then corrected by
        trimming or zero-padding back to the original length. This is a standard
        trade-off used in BirdCLEF top solutions.
        """
        factor = 2.0 ** (n_semitones / 12.0)
        # Treating waveform as captured at sample_rate*factor and resampling to
        # sample_rate: factor>1 → shorter output (higher pitch); factor<1 →
        # longer output (lower pitch).
        shifted = torchaudio.functional.resample(
            waveform,
            orig_freq=int(self.sample_rate * factor),
            new_freq=self.sample_rate
        )
        target_len = waveform.shape[1]
        if shifted.shape[1] >= target_len:
            return shifted[:, :target_len]
        return F.pad(shifted, (0, target_len - shifted.shape[1]))

    def _add_esc50_noise(self, waveform, snr_db=10):
        """Loads a random ESC-50 clip and mixes it into the waveform at target SNR.

        ESC-50 contains 2000 five-second clips across 50 environmental categories
        (rain, traffic, wind, animals, etc.). Using real recorded background noise
        rather than synthetic noise directly addresses the domain gap between the
        clean Xeno-canto training recordings and the continuous noisy soundscapes
        the competition scores against. This is the most targeted augmentation for
        the ~20-point LB gap we observed.

        Requires esc50_path to be set in BirbSet constructor. Download ESC-50 from:
        https://github.com/karoldvl/ESC-50  (place the extracted folder so that
        esc50_path/audio/*.wav exists)
        """
        esc_path = self.esc50_clips[np.random.randint(len(self.esc50_clips))]
        try:
            noise_wave, sr = torchaudio.load(esc_path)
            if sr != self.sample_rate:
                noise_wave = torchaudio.functional.resample(noise_wave, sr, self.sample_rate)
            if noise_wave.shape[0] > 1:
                noise_wave = noise_wave.mean(dim=0, keepdim=True)

            target_len = waveform.shape[1]
            if noise_wave.shape[1] >= target_len:
                # Random crop so we don't always use the same portion of each clip
                start = np.random.randint(0, noise_wave.shape[1] - target_len + 1)
                noise_wave = noise_wave[:, start:start + target_len]
            else:
                # Tile to fill length (ESC-50 clips are 5s, so this only triggers
                # if CLIP_LENGTH_SEC > 5 or after resampling rounding edge cases)
                repeats = (target_len // noise_wave.shape[1]) + 1
                noise_wave = noise_wave.repeat(1, repeats)[:, :target_len]

            sig_power   = torch.mean(waveform ** 2)
            noise_power = torch.mean(noise_wave ** 2)
            if sig_power < 1e-7 or noise_power < 1e-7:
                return waveform

            factor = torch.sqrt((sig_power / noise_power) * (10 ** (-snr_db / 10.0)))
            return waveform + noise_wave * factor
        except Exception:
            # Silently skip if a specific ESC-50 clip can't be loaded
            return waveform

    def __getitem__(self, idx):
        audio_clip = self.clips[idx]
        try:
            frame_offset = int(self.start_times[idx] * self.sample_rate)
            num_frames   = int((self.end_times[idx] - self.start_times[idx]) * self.sample_rate)

            waveform, _ = torchaudio.load(
                audio_clip, frame_offset=frame_offset, num_frames=num_frames
            )

            chunk_size  = int(self.sample_rate * self.clip_length) 
            current_len = waveform.shape[1]

            if current_len > chunk_size:
                waveform = waveform[:, :chunk_size]
            elif current_len < chunk_size:
                waveform = torch.nn.functional.pad(waveform, (0, chunk_size - current_len))

            if self.is_train:
                # --- Pitch shift (waveform-domain, applied first so noise is
                #     added on top of the shifted signal rather than the reverse) ---
                if self.use_pitch_shift and np.random.rand() < 0.5:
                    semitones = np.random.uniform(-3.0, 3.0)
                    waveform  = self._pitch_shift(waveform, n_semitones=semitones)

                # --- Noise augmentations: one active type is picked at random per
                #     sample so total noise probability stays fixed at 0.5 regardless
                #     of how many noise flags are on. Having all three on doesn't
                #     mean 3x more noise -- it means more *variety* of noise. ---
                active_noise_fns = []
                if self.use_pink_noise:
                    active_noise_fns.append(self._add_pink_noise)
                if self.use_white_noise:
                    active_noise_fns.append(self._add_white_noise)
                if self.use_esc50_noise and self.esc50_clips:
                    active_noise_fns.append(self._add_esc50_noise)

                if active_noise_fns and np.random.rand() < 0.5:
                    noise_fn = active_noise_fns[np.random.randint(len(active_noise_fns))]
                    snr      = np.random.uniform(10.0, 25.0)
                    waveform = noise_fn(waveform, snr_db=snr)

            # --- Bird-MAE path: bail out here with raw (augmented) audio.
            #     Target construction is identical either way, so it's
            #     duplicated rather than shared to keep each path self-
            #     contained and easy to follow. ---
            if self.raw_waveform_output:
                target = torch.zeros(len(self.label_to_idx), dtype=torch.float32)
                primary  = self.labels[idx]
                rating   = self.ratings[idx]
                confidence = 1.0 if pd.isna(rating) or rating == 0 else rating / 5.0
                target[primary] = confidence

                raw_secondary = self.secondary_labels[idx]
                if raw_secondary and raw_secondary not in ('[]', '', None):
                    for sec_label in ast.literal_eval(raw_secondary):
                        if sec_label in self.label_to_idx:
                            target[self.label_to_idx[sec_label]] = confidence * 0.3

                return waveform, target

            # Convert to dB
            spectrogram = self.mel_spect(waveform)
            spectrogram = self.amp_to_db(spectrogram)
            
            mean, std   = spectrogram.mean(), spectrogram.std() + 1e-6
            spectrogram = (spectrogram - mean) / std

            # --- SpecAugment (spectrogram-domain, applied after mel conversion) ---
            if self.is_train and self.use_spec_augment:
                if np.random.rand() < 0.5:
                    spectrogram = self.freq_mask(spectrogram)
                if np.random.rand() < 0.5:
                    spectrogram = self.time_mask(spectrogram)

            target = torch.zeros(len(self.label_to_idx), dtype=torch.float32)
            primary  = self.labels[idx]
            rating   = self.ratings[idx]
            
            confidence = 1.0 if pd.isna(rating) or rating == 0 else rating / 5.0 
            target[primary] = confidence

            raw_secondary = self.secondary_labels[idx]
            if raw_secondary and raw_secondary not in ('[]', '', None):
                for sec_label in ast.literal_eval(raw_secondary):   
                    if sec_label in self.label_to_idx:
                        target[self.label_to_idx[sec_label]] = confidence * 0.3

            return spectrogram, target
            
        except Exception as e:
            print(f"Skipping corrupted/missing file at index {idx} -> {e}. Path: {audio_clip}")
            return self.__getitem__((idx + 1) % len(self))


class SoundscapeDataset(Dataset):
    def __init__(self, audio_path, clip_length=5.0, sample_rate=32000, n_mels=128, n_fft=1024):
        self.clip_length = clip_length
        self.sample_rate = sample_rate
        self.chunk_size = int(self.sample_rate * self.clip_length)
        
        self.waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            self.waveform = torchaudio.functional.resample(self.waveform, sr, self.sample_rate)
            
        if self.waveform.shape[0] > 1:
            self.waveform = self.waveform.mean(dim=0, keepdim=True)
            
        self.num_chunks = self.waveform.shape[1] // self.chunk_size
        
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        self.mel_spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=n_fft, n_mels=n_mels,
        )

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx   = start_idx + self.chunk_size
        
        chunk = self.waveform[:, start_idx:end_idx]
        
        spectrogram = self.mel_spect(chunk)
        spectrogram = self.amp_to_db(spectrogram)
        mean, std   = spectrogram.mean(), spectrogram.std() + 1e-6
        spectrogram = (spectrogram - mean) / std
        
        end_time = (idx + 1) * int(self.clip_length)
        
        return spectrogram, end_time

# ==========================================
# DATALOADER WORKER SEEDING
# ==========================================
def seed_worker(worker_id):
    """
    PyTorch reseeds its own RNG per DataLoader worker automatically, but NOT
    NumPy's or Python's `random` module global RNG. Since pink-noise gating
    (np.random.rand()/np.random.uniform()) relies on NumPy's global RNG, worker
    processes forked together can inherit correlated RNG state, reducing the
    effective randomness of that augmentation across workers/batches. This
    derives a distinct seed per worker from torch's per-worker seed and applies
    it to both NumPy and Python's random module.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========================================
# MIXUP / CUTMIX ("MIXCUT") AUGMENTATION
# ==========================================
def apply_mixcut(spectrograms, targets, alpha=0.4, cutmix_prob=0.5):
    """
    Batch-level MixUp/CutMix augmentation for multi-label spectrogram classification.

    Each call randomly picks one of:
      - MixUp:  linearly blends two full spectrograms (and their targets)
      - CutMix: pastes a rectangular time/freq patch from one spectrogram into
                another, mixing targets in proportion to the swapped area

    Targets are multi-hot/soft-label vectors (BirbSet already encodes primary +
    secondary label confidences), so they're blended linearly with the same
    lambda used for the spectrograms -- this is a standard, simple extension of
    MixUp/CutMix to multi-label settings.

    Operates on an already-batched, already-GPU tensor pair, intended to run
    right after spectrograms/targets are moved to device and before the
    autocast forward pass.
    """
    batch_size = spectrograms.size(0)
    if batch_size < 2:
        return spectrograms, targets

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(batch_size, device=spectrograms.device)

    if np.random.rand() < cutmix_prob:
        # --- CutMix: swap a rectangular region across the (mel, time) axes ---
        _, _, H, W = spectrograms.shape
        cut_ratio = np.sqrt(max(1.0 - lam, 0.0))
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)

        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = int(np.clip(cy - cut_h // 2, 0, H)), int(np.clip(cy + cut_h // 2, 0, H))
        x1, x2 = int(np.clip(cx - cut_w // 2, 0, W)), int(np.clip(cx + cut_w // 2, 0, W))

        spectrograms[:, :, y1:y2, x1:x2] = spectrograms[perm, :, y1:y2, x1:x2]

        # Recompute lambda to reflect the actual swapped area (handles edge clipping)
        swapped_area = (y2 - y1) * (x2 - x1)
        lam = 1.0 - (swapped_area / (H * W))
    else:
        # --- MixUp: linear blend of full spectrograms ---
        spectrograms = lam * spectrograms + (1.0 - lam) * spectrograms[perm]

    mixed_targets = lam * targets + (1.0 - lam) * targets[perm]

    return spectrograms, mixed_targets


def apply_mixcut_waveform(waveforms, targets, alpha=0.4, cutmix_prob=0.5):
    """
    Waveform-domain counterpart to apply_mixcut, used with Bird-MAE since
    mixing now has to happen BEFORE Bird-MAE's own feature_extractor turns
    each clip into a spectrogram (mixing its output would require re-deriving
    two different mel spectrograms and interleaving them, which the
    feature_extractor doesn't expose a hook for).

    - MixUp: linear blend of two raw waveforms (and their targets) -- direct
      1D analogue of the spectrogram version.
    - "CutMix": splices a contiguous time segment from one waveform into
      another, rather than a 2D (mel, time) rectangle, since there's no mel
      axis yet at this stage. Targets are mixed in proportion to the swapped
      duration, same idea as the 2D version.

    waveforms: tensor [B, 1, T]
    """
    batch_size = waveforms.size(0)
    if batch_size < 2:
        return waveforms, targets

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(batch_size, device=waveforms.device)

    if np.random.rand() < cutmix_prob:
        T = waveforms.shape[-1]
        cut_ratio = np.sqrt(max(1.0 - lam, 0.0))
        cut_len = int(T * cut_ratio)

        cx = np.random.randint(T)
        x1, x2 = int(np.clip(cx - cut_len // 2, 0, T)), int(np.clip(cx + cut_len // 2, 0, T))

        waveforms = waveforms.clone()
        waveforms[:, :, x1:x2] = waveforms[perm, :, x1:x2]

        swapped_len = x2 - x1
        lam = 1.0 - (swapped_len / T)
    else:
        waveforms = lam * waveforms + (1.0 - lam) * waveforms[perm]

    mixed_targets = lam * targets + (1.0 - lam) * targets[perm]

    return waveforms, mixed_targets

# ==========================================
# TRAIN / VALIDATE EPOCH
# ==========================================
def train_epoch(model, dataloader, optimizer, criterion, scaler, epoch,
                 use_mixcut=False, mixcut_alpha=0.4, mixcut_prob=0.5, device=None):
    device = device if device is not None else next(model.parameters()).device
    model.train()
    total_loss     = 0.0
    total_grad_norm = 0.0
    num_batches    = 0
    nan_grad_batches = 0   # batches where GradScaler detected NaN and skipped the update

    if len(dataloader) == 0:
        print("Warning: DataLoader has 0 batches. Check your dataset.")
        return 0.0, 0.0, 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)

    for waveforms, targets in pbar:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)

        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        if use_mixcut:
            waveforms, targets = apply_mixcut_waveform(
                waveforms, targets, alpha=mixcut_alpha, cutmix_prob=mixcut_prob
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            logits = model(waveforms)
            loss   = criterion(logits, targets)

        scaler.scale(loss).backward()

        # Unscale before clipping so max_norm applies to true gradient magnitude.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_val = float(grad_norm)

        if not math.isnan(grad_norm_val) and not math.isinf(grad_norm_val):
            total_grad_norm += grad_norm_val
        else:
            nan_grad_batches += 1

        scaler.step(optimizer)
        scaler.update()

        loss_val     = loss.item()
        total_loss  += loss_val if not math.isnan(loss_val) else 0.0
        num_batches += 1
        pbar.set_postfix({
            'loss':      f"{loss_val:.4f}",
            'grad_norm': f"{grad_norm_val:.3f}" if not math.isnan(grad_norm_val) else "NaN",
            'nan_b':     nan_grad_batches,
        })

    valid_batches = num_batches - nan_grad_batches
    avg_grad_norm = total_grad_norm / valid_batches if valid_batches > 0 else float('nan')
    return total_loss / num_batches, avg_grad_norm, nan_grad_batches


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, epoch, device=None):
    device = device if device is not None else next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", dynamic_ncols=True)

    for waveforms, targets in pbar:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)

        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        with autocast(device_type=device.type):
            logits = model(waveforms)
            loss = criterion(logits, targets)

        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    binary_targets = (all_targets > 0.0).astype(int)

    has_positive = np.any(binary_targets == 1, axis=0)
    has_negative = np.any(binary_targets == 0, axis=0)
    valid_classes = has_positive & has_negative
    num_total_classes = binary_targets.shape[1]
    num_valid_classes = int(valid_classes.sum())

    per_class_auc = None
    if num_valid_classes == 0:
        print("Warning: No classes with both positive and negative examples found for ROC-AUC calculation in this split.")
        val_auc = 0.0
    else:
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
            print(f"ROC-AUC calculation error: {e}")
            val_auc = 0.0

    mean_prob_pos, mean_prob_neg = float('nan'), float('nan')
    if num_valid_classes > 0:
        scored_targets = binary_targets[:, valid_classes].astype(bool)
        scored_preds = all_preds[:, valid_classes]
        if scored_targets.any():
            mean_prob_pos = float(scored_preds[scored_targets].mean())
        if (~scored_targets).any():
            mean_prob_neg = float(scored_preds[~scored_targets].mean())

    map_valid_classes = has_positive            # mAP only needs positives
    num_map_classes   = int(map_valid_classes.sum())
    val_map = float('nan')
    per_class_ap = None
    if num_map_classes > 0:
        try:
            val_map = average_precision_score(
                binary_targets[:, map_valid_classes],
                all_preds[:, map_valid_classes],
                average='macro'
            )
            per_class_ap = average_precision_score(
                binary_targets[:, map_valid_classes],
                all_preds[:, map_valid_classes],
                average=None
            )
        except ValueError as e:
            print(f"mAP calculation error: {e}")

    extra_metrics = {
        "num_valid_classes": num_valid_classes,
        "num_skipped_classes": num_total_classes - num_valid_classes,
        "auc_min": float(per_class_auc.min()) if per_class_auc is not None else float('nan'),
        "auc_max": float(per_class_auc.max()) if per_class_auc is not None else float('nan'),
        "auc_std": float(per_class_auc.std()) if per_class_auc is not None else float('nan'),
        "mean_prob_positive": mean_prob_pos,
        "mean_prob_negative": mean_prob_neg,
        "val_map":          val_map,
        "num_map_classes":  num_map_classes,
        "map_min":  float(per_class_ap.min()) if per_class_ap is not None else float('nan'),
        "map_max":  float(per_class_ap.max()) if per_class_ap is not None else float('nan'),
        "map_std":  float(per_class_ap.std()) if per_class_ap is not None else float('nan'),
    }

    return total_loss / len(dataloader), val_auc, extra_metrics


# ==========================================
# FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification.

    Focal loss down-weights easy (already well-classified) examples and
    concentrates gradient signal on hard ones. In a 234-class rare-species
    dataset this mostly means the model spends less time over-fitting on
    common species it already predicts confidently and more time learning
    the rare ones it's still failing on.

    NUMERICAL STABILITY NOTE: the focal weighting term (1 - p_t) ** gamma
    is computed in float32 regardless of AMP context. When the model becomes
    confident (p_t → 1), this term underflows to 0 in float16, and the
    backward pass through x**gamma at x≈0 produces NaN gradients in half
    precision -- even though the loss *value* stays finite because
    binary_cross_entropy_with_logits uses a numerically stable internal path.
    Casting to float32 here and clamping p_t eliminates this failure mode.

    Gamma=1.0 default (reduced from 2.0): gamma=2 is very aggressive for a
    soft multi-label problem with MixCut active -- with soft targets the
    effective p_t is moderate early in training, but as the model converges
    focal weights → 0 and gradient signal vanishes, showing up as suspiciously
    low train loss (0.001x) well before the model has actually learned rare
    classes. Start at 1.0 and only raise if easy-class dominance is clearly
    visible in per-class AUC diagnostics after a full run.
    """
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        # Float32 cast: keeps the stable log-sum-exp path inside
        # binary_cross_entropy_with_logits while ensuring the focal
        # weighting backward pass can't produce NaN from float16 underflow.
        logits  = logits.float()
        targets = targets.float()

        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1.0 - probs) * (1.0 - targets)
        p_t    = p_t.clamp(min=1e-6, max=1.0 - 1e-6)   # prevent (1-1)^gamma → NaN grad
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ==========================================
# MODEL DEFINITION
# ==========================================
class BirdMAEClassifier(nn.Module):
    """
    Wraps a pretrained Bird-MAE backbone + its own feature_extractor into a
    standard classifier: raw waveform batch in, per-species logits out.

    Bird-MAE's feature_extractor computes its own mel spectrogram (with its
    own pretrained normalization stats) from a single 1D numpy array -- the
    published example only shows one clip at a time, not a batch. To stay
    correct without assuming undocumented batching support, this loops the
    feature_extractor per-sample and does ONE batched forward pass through
    the (expensive) backbone. If you confirm the feature_extractor accepts a
    list of arrays, that loop can be vectorized for a real speedup.

    NOTE: Bird-MAE-Huge is a large ViT-scale model. Running it frozen on CPU
    over a full BirdCLEF training set, per epoch, will be slow -- this is
    inherent to using a "Huge" foundation model on CPU, not something the
    code below can paper over. Consider Bird-MAE-Base if you don't have GPU
    access, or precomputing/caching embeddings once (since the backbone is
    frozen, its output for a given raw clip never changes across epochs).
    """

    def __init__(self, model_name="DBD-research-group/Bird-MAE-Huge",
                 head="linear", num_classes=None, sample_rate=32000, freeze_backbone=True,
                 hidden_dim=512, dropout=0.3):
        super().__init__()
        self.sample_rate = sample_rate
        self.freeze_backbone = freeze_backbone

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        # low_cpu_mem_usage=False: this checkpoint's custom modeling code
        # (modeling_bird_mae.py) calls .item() on tensors during __init__
        # (e.g. building the drop-path schedule), which isn't safe under
        # transformers' default fast-init path -- that path constructs the
        # model on the "meta" device first and materializes weights after,
        # and .item() cannot be called on a meta tensor ("Tensor.item()
        # cannot be called on meta tensors"). Forcing the classic init path
        # avoids that crash. Slightly slower to load, doesn't affect training.
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=False
        )

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        embed_dim = self._infer_embed_dim()
        # Simple MLP head on top of the (frozen) Bird-MAE embedding: one
        # hidden layer is enough here -- the backbone already does the heavy
        # representation learning, this just needs to linearly-ish separate
        # species in embedding space.
        match head:
            case "linear":
                self.classifier = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(p=dropout),
                    nn.Linear(embed_dim, num_classes),
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
        # "input_values" is a guess based on common HF audio-model convention.
        # Run `print(feature_extractor(dummy).keys())` once to confirm the
        # actual key and adjust here if it differs.
        if hasattr(feat, "keys"):
            key = "input_values" if "input_values" in feat else list(feat.keys())[0]
            return feat[key]
        return feat

    def train(self, mode=True):
        # model.train() at the top of each training epoch would otherwise
        # flip the frozen backbone's Dropout/BatchNorm-ish layers back into
        # train mode too -- keep it pinned to eval whenever it's frozen.
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def unfreeze_backbone(self):
        """
        Phase 1 -> Phase 2 transition: unfreeze every backbone parameter for
        full end-to-end fine-tuning. This only flips requires_grad flags and
        puts the backbone back in train() mode -- it does NOT touch any
        existing optimizer, which was built from the (previously frozen)
        parameter list and won't include these newly-trainable params. The
        caller must rebuild the optimizer (and typically the LR scheduler)
        immediately after calling this -- see the Phase 2 block in the main
        training loop below.
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()
        self.freeze_backbone = False
        print("  🔓 Backbone unfrozen -- entering Phase 2 (full fine-tuning).")

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
        # waveform_batch: tensor [B, 1, T], can be on CPU or CUDA
        device = waveform_batch.device
        specs = []
        for wf in waveform_batch:
            # feature_extractor needs a plain CPU numpy array regardless of
            # which device the batch/model live on -- .cpu() here, then move
            # the extracted spec back to `device` below before the backbone
            # forward pass.
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
# VALIDATION COVERAGE FIX
# ==========================================
def ensure_val_coverage(full_df, df_train, df_val,
                         label_col='primary_label', filename_col='filename'):
    """
    Guarantees every class that exists in the full dataset appears in the
    validation split with at least one example.

    Root cause: StratifiedGroupKFold groups by filename, so if a rare species
    has only 1-2 audio files, there is a real chance all of them land in
    training and none in validation. ROC-AUC then silently drops those classes
    from the macro average, inflating the reported metric relative to what the
    leaderboard actually scores (which includes those species).

    Fix: after the initial split, identify every class absent from val and move
    exactly one file per missing class from train to val, choosing the file with
    the most clips to maximise the chance of meaningful positive/negative coverage
    in the resulting val set. Group integrity is preserved -- all clips from any
    one audio file stay on the same side of the split.

    Note: this introduces a mild form of leakage for the moved examples (the
    model has trained on them). This is an acceptable trade-off when the
    alternative is a validation metric computed over a materially different set
    of classes than what the leaderboard scores against.
    """
    val_classes   = set(df_val[label_col].unique())
    all_classes   = set(full_df[label_col].unique())
    missing       = sorted(all_classes - val_classes)

    if not missing:
        print(f"✅ Val class coverage: {len(val_classes)}/{len(all_classes)} (100%) — no fix needed.")
        return df_train, df_val

    print(f"⚠️  Val class coverage before fix: {len(val_classes)}/{len(all_classes)}. "
          f"Moving one file per missing class for {len(missing)} class(es)...")

    indices_to_move = []
    still_missing   = []
    for cls in missing:
        candidate_files = df_train.loc[df_train[label_col] == cls, filename_col].unique()
        if len(candidate_files) == 0:
            still_missing.append(cls)
            continue
        # Pick the file with the most rows so the moved val subset is as
        # representative as possible for that class.
        file_clip_counts = {
            f: int((df_train[filename_col] == f).sum()) for f in candidate_files
        }
        chosen_file = max(file_clip_counts, key=file_clip_counts.get)
        indices_to_move.extend(df_train.index[df_train[filename_col] == chosen_file].tolist())

    if indices_to_move:
        move_mask  = df_train.index.isin(indices_to_move)
        rows_moved = df_train[move_mask].copy()
        df_train   = df_train[~move_mask].reset_index(drop=True)
        df_val     = pd.concat([df_val, rows_moved], ignore_index=True)

    val_classes_after = set(df_val[label_col].unique())
    print(f"✅ Val class coverage after fix: {len(val_classes_after)}/{len(all_classes)}. "
          f"Moved {len(indices_to_move)} clips across {len(missing) - len(still_missing)} file(s).")
    if still_missing:
        print(f"   ⚠️  {len(still_missing)} class(es) couldn't be moved (no training examples): {still_missing}")

    return df_train, df_val




# ==========================================
# SOUNDSCAPE DATA LOADING
# ==========================================
def load_soundscape_labels(labels_path, soundscape_audio_dir, label_to_idx,
                            clip_length=5.0):
    """
    Load train_soundscape_labels.csv and return a clip-level DataFrame that
    BirbSet can ingest directly via its soundscape_clips_df parameter.

    Handles column-naming variations across BirdCLEF years:
      - start_sec / end_sec  (BirdCLEF 2024+)
      - start_time / end_time

    If the file has multiple rows per time window (one row per detected
    species) or if species are packed together with semicolons, they are 
    processed so the first species becomes primary_label and the rest 
    become secondary_labels, matching BirbSet's soft-target format.
    """
    df = pd.read_csv(labels_path)

    rename = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ('start_sec', 'start_time', 'start'):
            rename[col] = 'start_time'
        elif lc in ('end_sec', 'end_time', 'end', 'seconds_end'):
            rename[col] = 'end_time'
        elif lc in ('primary_label', 'label', 'species', 'birds'):
            rename[col] = 'primary_label'
        elif lc in ('secondary_labels', 'secondary'):
            rename[col] = 'secondary_labels'
        elif lc in ('filename', 'soundscape_id', 'file_id'):
            rename[col] = 'filename'
    df = df.rename(columns=rename)

    required = {'filename', 'start_time', 'end_time', 'primary_label'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"train_soundscape_labels.csv is missing columns: {missing}. "
            f"Available: {list(df.columns)}\n"
            f"Adjust the rename map in load_soundscape_labels() if column names differ."
        )

    # Infer end_time if missing or all-zero
    if df['end_time'].isna().all() or (df['end_time'] == 0).all():
        df['end_time'] = df['start_time'] + clip_length

    if 'secondary_labels' not in df.columns:
        group_keys = ['filename', 'start_time', 'end_time']

        def aggregate_species(rows):
            # 1. Gather all species from this window, breaking up semicolon-separated strings
            all_species = []
            for val in rows['primary_label'].dropna():
                val_str = str(val).strip()
                if ';' in val_str:
                    tokens = [s.strip() for s in val_str.split(';') if s.strip()]
                    all_species.extend(tokens)
                elif val_str:
                    all_species.append(val_str)
            
            # 2. Deduplicate species while preserving sequence order
            unique_species = []
            for s in all_species:
                if s not in unique_species:
                    unique_species.append(s)

            if not unique_species:
                return None
                
            # 3. First bird is primary, the remainder become secondary
            r = rows.iloc[0].copy()
            r['primary_label']    = unique_species[0]
            r['secondary_labels'] = str(unique_species[1:]) if len(unique_species) > 1 else '[]'
            return r

        # Run aggregation
        df = (
            df.groupby(group_keys, sort=False)
              .apply(aggregate_species)
              .dropna()
        )
        
        # FIX: Version-agnostic restoration of grouping columns from index to columns
        for key in group_keys:
            if key in df.index.names and key not in df.columns:
                df = df.reset_index(level=key)
                
        # Clean up any remaining index levels safely
        df = df.reset_index(drop=True)
    else:
        # Fallback: If secondary_labels already exists, ensure packed strings 
        # in primary_label are still split out and appended cleanly.
        def unpack_row_with_secondary(row):
            val_str = str(row['primary_label']).strip()
            if ';' in val_str:
                tokens = [s.strip() for s in val_str.split(';') if s.strip()]
                if tokens:
                    row['primary_label'] = tokens[0]
                    
                    try:
                        existing = ast.literal_eval(str(row['secondary_labels'])) if pd.notna(row['secondary_labels']) else []
                        if not isinstance(existing, list): existing = [str(existing)]
                    except Exception:
                        existing = [str(row['secondary_labels'])] if pd.notna(row['secondary_labels']) and str(row['secondary_labels']).strip() != '[]' else []
                    
                    combined = []
                    for s in (tokens[1:] + existing):
                        if s not in combined:
                            combined.append(s)
                    row['secondary_labels'] = str(combined)
            return row

        df = df.apply(unpack_row_with_secondary, axis=1)
        df['secondary_labels'] = df['secondary_labels'].fillna('[]')

    df['audio_path'] = df['filename'].apply(
        lambda f: os.path.join(soundscape_audio_dir, os.path.basename(str(f)))
    )
    df = df[df['primary_label'].isin(label_to_idx)].copy()
    df['rating'] = np.nan

    exists_mask   = df['audio_path'].apply(os.path.exists)
    missing_files = (~exists_mask).sum()
    if missing_files > 0:
        print(f"  {missing_files} soundscape audio file(s) not found on disk -- skipped.")
    df = df[exists_mask].reset_index(drop=True)

    print(f"Loaded {len(df)} soundscape clips "
          f"({df['primary_label'].nunique()} unique species).")
    return df[['audio_path', 'start_time', 'end_time',
               'primary_label', 'secondary_labels', 'rating']]

# ==========================================
# COVERAGE-AWARE SPLIT
# ==========================================
def _count_all_appearances(df, primary_col='primary_label',
                            secondary_col='secondary_labels'):
    """
    Count every class occurrence across both primary AND secondary labels.

    Used by build_coverage_aware_split to determine train-only vs val-eligible
    thresholds. Without this, classes that only appear as secondary labels
    (e.g. background species in soundscape annotations) all get counted as 0
    primary occurrences, incorrectly forcing them into train-only status even
    when they appear dozens of times across the dataset.

    Secondary label strings are stored as Python list representations
    (e.g. "['asbfly', 'comsan']") -- the same format BirbSet writes.
    """
    counts = Counter(df[primary_col].dropna().tolist())
    if secondary_col not in df.columns:
        return counts
    for sec_val in df[secondary_col].dropna():
        sec_str = str(sec_val).strip()
        if sec_str in ('[]', '', 'nan', 'None'):
            continue
        try:
            for lbl in ast.literal_eval(sec_str):
                if lbl and lbl != 'nocall':
                    counts[lbl] += 1
        except (ValueError, SyntaxError):
            pass
    return counts


def build_coverage_aware_split(file_df, soundscape_clips_df=None,
                                label_col='primary_label', filename_col='filename',
                                n_splits=5, min_clips_for_val=5, random_state=42):
    """
    Split data to maximise TRAINING coverage rather than validation coverage.

    Strategy:
      1. Count total occurrences of every class across BOTH file_df (train.csv)
         and soundscape_clips_df, counting appearances as PRIMARY OR SECONDARY
         labels. This ensures classes that only appear in secondary labels are
         not incorrectly penalised by a primary-only count.

      2. Classes with < min_clips_for_val total occurrences are "train-only" --
         all their clips go to training. This guarantees the model trains on
         every class we know about, including soundscape-only and secondary-only
         classes that never appear as a primary label in train.csv.

      3. Classes with enough clips get a stratified split: StratifiedGroupKFold
         on file_df (grouped by filename to prevent leakage), plus a simple
         random split on soundscape clips (no grouping constraint since each
         window is an independent annotation).

    Returns
    -------
    file_train, file_val : subsets of file_df
    sc_train, sc_val     : subsets of soundscape_clips_df (None if not provided)
    train_only_classes   : set of class names intentionally excluded from val
    """
    file_counts = _count_all_appearances(file_df)
    sc_counts   = (_count_all_appearances(soundscape_clips_df)
                   if soundscape_clips_df is not None and len(soundscape_clips_df) > 0
                   else Counter())

    all_classes  = set(file_counts) | set(sc_counts)
    total_counts = {cls: file_counts.get(cls, 0) + sc_counts.get(cls, 0)
                    for cls in all_classes}

    train_only_classes = {cls for cls, n in total_counts.items() if n < min_clips_for_val}
    val_eligible       = all_classes - train_only_classes

    # Classes with 0 primary occurrences in file_df (only appear as secondary
    # or only in soundscapes)
    primary_only_file  = set(file_df[label_col].unique())
    secondary_only     = all_classes - primary_only_file
    soundscape_only    = set(sc_counts.keys()) - set(file_counts.keys())

    print(f"\nClass coverage analysis (min_clips_for_val={min_clips_for_val}):")
    print(f"  Total classes (primary + secondary, both sources): {len(all_classes)}")
    print(f"  Classes only in secondary labels                 : {len(secondary_only)}")
    print(f"  Classes only in soundscapes                      : {len(soundscape_only)}")
    print(f"  Train-only (too rare for val)                    : {len(train_only_classes)}")
    print(f"  Val-eligible                                     : {len(val_eligible)}")

    # --- Split file_df ---
    file_train_only = file_df[file_df[label_col].isin(train_only_classes)]
    file_val_elig   = file_df[file_df[label_col].isin(val_eligible)]

    n_unique = len(file_val_elig[label_col].unique())
    if len(file_val_elig) > 0 and n_unique >= n_splits:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=random_state)
        train_idx, val_idx = next(sgkf.split(
            X=file_val_elig,
            y=file_val_elig[label_col],
            groups=file_val_elig[filename_col]
        ))
        file_train = pd.concat(
            [file_train_only, file_val_elig.iloc[train_idx]]
        ).reset_index(drop=True)
        file_val = file_val_elig.iloc[val_idx].reset_index(drop=True)
    else:
        print("  Too few val-eligible classes to stratify -- all file rows go to training.")
        file_train = file_df.reset_index(drop=True)
        file_val   = pd.DataFrame(columns=file_df.columns)

    # --- Split soundscape clips ---
    sc_train = sc_val = None
    if soundscape_clips_df is not None and len(soundscape_clips_df) > 0:
        sc_train_only = soundscape_clips_df[
            soundscape_clips_df[label_col].isin(train_only_classes)
        ]
        sc_val_elig = soundscape_clips_df[
            soundscape_clips_df[label_col].isin(val_eligible)
        ]
        val_frac = 1.0 / n_splits
        if len(sc_val_elig) > 0:
            sc_val_part   = sc_val_elig.sample(frac=val_frac, random_state=random_state)
            sc_train_part = sc_val_elig.drop(sc_val_part.index)
        else:
            sc_val_part   = pd.DataFrame(columns=soundscape_clips_df.columns)
            sc_train_part = sc_val_elig

        sc_train = pd.concat([sc_train_only, sc_train_part]).reset_index(drop=True)
        sc_val   = sc_val_part.reset_index(drop=True)

    val_cls = (set(file_val[label_col].unique()) if len(file_val) > 0 else set()) |               (set(sc_val[label_col].unique()) if (sc_val is not None and len(sc_val) > 0) else set())
    print(f"\nSplit result:")
    print(f"  file_train : {len(file_train)} rows  |  file_val : {len(file_val)} rows")
    if soundscape_clips_df is not None:
        print(f"  sc_train   : {len(sc_train)} clips |  sc_val   : {len(sc_val)} clips")
    print(f"  Classes in val  : {len(val_cls)} / {len(all_classes)} "
          f"({len(train_only_classes)} intentionally train-only)\n")

    return file_train, file_val, sc_train, sc_val, train_only_classes



# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_size', required=True, choices=["Base", "Large", "Huge"], type=str, help="Bird-MAE size: Base, Large, Huge")
    parser.add_argument('--head', required=True, type=str, choices=["linear", "mlp"], help="Classification head: linear, mlp")
    parser.add_argument('--pink_noise', action='store_true')
    parser.add_argument('--white_noise', action='store_true')
    parser.add_argument('--pitch_shift', action='store_true')
    parser.add_argument('--mixcut', action='store_true')
    args = parser.parse_args()
    
    # --- Configuration ---
    root_path       = os.path.join("../", "birdclef-2026")
    CLIP_LENGTH_SEC = 5.0  
    MAX_EPOCHS      = 30
    PATIENCE        = 12
    WARMUP_EPOCHS   = 3     # linear LR ramp-up before cosine decay kicks in
    FOCAL_GAMMA     = 1.0   # focal loss concentration; 0=plain BCE, 2=original RetinaNet default
    # Spectrogram resolution -- 128 mels is the near-universal choice in top
    # BirdCLEF solutions; 64 is too coarse to reliably separate species with
    # similar call frequency ranges. n_fft=1024 pairs with default hop_length=512
    # (giving ~312 time frames per 5s clip vs ~400 at n_fft=800) for slightly
    # better frequency resolution at a small cost in time resolution.
    N_MELS          = 128
    N_FFT           = 1024
    # SWA: start averaging weights once the model has mostly converged. With
    # MAX_EPOCHS=30 and PATIENCE=12, the model typically stabilises around
    # epoch 15-20; averaging the last ~10 epochs' weights consistently adds
    # 1-3 AUC points in BirdCLEF solutions for essentially zero extra compute.
    # Set to MAX_EPOCHS + 1 to disable SWA entirely.
    SWA_START_EPOCH = 20
    # Minimum total clip count (across train.csv + soundscapes) for a class
    # to be eligible for the validation split. Classes below this threshold
    # are "train-only" -- we train on every clip we have rather than holding
    # any out for validation, accepting reduced val coverage for those classes.
    MIN_CLIPS_FOR_VAL = 5

    # --- BIRD-MAE CONFIG ---
    BIRDMAE_MODEL_SIZE = args.backbone_size # Base, Large, Huge
    BIRDMAE_MODEL_NAME = "DBD-research-group/Bird-MAE-" + BIRDMAE_MODEL_SIZE
    FREEZE_BACKBONE     = True    # Phase 1 always starts frozen -- Phase 2
                                   # kicks in automatically at UNFREEZE_AFTER_EPOCH,
                                   # see the transition block in the training loop.
    BACKBONE_LR         = 1e-5    # Phase 2: uniform LR across the WHOLE model
                                   # (encoder + head) once unfrozen -- deliberately
                                   # low so backprop through the pretrained encoder
                                   # doesn't wreck what it learned during SSL.
    HEAD_LR             = 1e-3    # Phase 1: head-only LR while encoder is frozen.

    # Phase 1 (epochs 1..UNFREEZE_AFTER_EPOCH): frozen encoder, head-only.
    # Phase 2 (epochs UNFREEZE_AFTER_EPOCH+1..MAX_EPOCHS): full fine-tune.
    UNFREEZE_AFTER_EPOCH = 5
    # Brief re-warmup right as the encoder becomes trainable -- same
    # "don't shock it on step one" rationale as the original WARMUP_EPOCHS,
    # applied again since a fresh optimizer has no momentum state yet.
    PHASE2_WARMUP_EPOCHS = 1

    # --- AUGMENTATION TOGGLES ---
    USE_PINK_NOISE   = args.pink_noise    # colored noise: 1/f pink spectrum
    USE_WHITE_NOISE  = args.white_noise   # colored noise: flat white spectrum
    USE_SPEC_AUGMENT = False    # N/A for Bird-MAE -- SpecAugment operates on OUR mel
                                 # spectrogram, which we no longer compute; BirbSet
                                 # forces this off internally when raw_waveform_output=True
    USE_PITCH_SHIFT  = args.pitch_shift   # ±3 semitone pitch shift via resampling trick
    USE_ESC50_NOISE  = False   # real environmental background noise from ESC-50
    ESC50_PATH       = os.path.join("..", "ESC-50-master")   # set to None to disable
    # Which ESC-50 sound categories to use as background noise.
    # None or [] means use all 50 categories (original behaviour).
    # Set to a list of category names from meta/esc50.csv to restrict.
    # Spaces, hyphens, and case are normalised automatically, so
    # "Sea Waves", "sea-waves", and "sea_waves" all match.
    # Full category list: https://github.com/karoldvl/ESC-50#license
    # Recommended environmental-only subset (excludes animals / indoor sounds):
    ESC50_CATEGORIES = [
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "water_drops",
        "wind",
        "pouring_water",
        "thunderstorm",
    ]
    USE_MIXCUT       = args.mixcut    # batch-level MixUp / CutMix
    MIXCUT_ALPHA     = 0.4     # Beta distribution shape param; lower = milder mixing
    MIXCUT_PROB      = 0.5     # probability of CutMix vs MixUp when mixcut fires

    tag_parts = [f'birdmae-{BIRDMAE_MODEL_SIZE.lower()}-{args.head}']
    #if FREEZE_BACKBONE:  tag_parts.append("frozen")
    if USE_PINK_NOISE:   tag_parts.append("pink")
    if USE_WHITE_NOISE:  tag_parts.append("white")
    if USE_PITCH_SHIFT:  tag_parts.append("pitch")
    if USE_ESC50_NOISE:  tag_parts.append("esc50")
    if USE_MIXCUT:       tag_parts.append("mixcut")
    run_tag = "_".join(tag_parts)

    run_name        = f"Training {run_tag}"
    artifact_name   = f"model_{run_tag}"
    checkpoint_path = f"best_model_{run_tag}.pth"

    # --- Setup Logging ---
    run = wandb.init(
        entity="axialmars-tu-dresden",
        project="BirdsArentReal",
        name=run_name,
        config={
            "head_learning_rate": HEAD_LR,
            "backbone_learning_rate": BACKBONE_LR,
            "architecture": BIRDMAE_MODEL_NAME,
            "freeze_backbone": FREEZE_BACKBONE,
            "unfreeze_after_epoch": UNFREEZE_AFTER_EPOCH,
            "phase2_warmup_epochs": PHASE2_WARMUP_EPOCHS,
            "dataset": "BirdClef+ 2026",
            "epochs": MAX_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "focal_gamma": FOCAL_GAMMA,
            "swa_start_epoch": SWA_START_EPOCH,
            "use_pink_noise": USE_PINK_NOISE,
            "use_white_noise": USE_WHITE_NOISE,
            "use_pitch_shift": USE_PITCH_SHIFT,
            "use_esc50_noise": USE_ESC50_NOISE,
            "esc50_categories": ESC50_CATEGORIES if USE_ESC50_NOISE else [],
            "use_mixcut": USE_MIXCUT,
            "mixcut_alpha": MIXCUT_ALPHA,
            "mixcut_prob": MIXCUT_PROB,
        },
    )

    # --- Load Data & Setup Splits ---
    full_df       = pd.read_csv(os.path.join(root_path, "train.csv"))
    unique_labels = pd.read_csv(os.path.join(root_path, "taxonomy.csv"))
    master_label_to_idx = {label: i for i, label in enumerate(unique_labels['primary_label'].unique())}
    num_classes         = len(master_label_to_idx)

    # --- Load soundscape labels (if present) ---
    soundscape_labels_path = os.path.join(root_path, "train_soundscapes_labels.csv")
    soundscape_audio_dir   = os.path.join(root_path, "train_soundscapes")
    sc_clips_all = None
    if os.path.exists(soundscape_labels_path):
        print(f"Loading soundscape labels from {soundscape_labels_path} ...")
        sc_clips_all = load_soundscape_labels(
            soundscape_labels_path, soundscape_audio_dir,
            master_label_to_idx, clip_length=CLIP_LENGTH_SEC
        )
    else:
        print("train_soundscapes_labels.csv not found -- proceeding with train.csv only.")

    # --- Coverage-aware split ---
    # Replaces the old StratifiedGroupKFold + ensure_val_coverage pair.
    # Priority: train on as many classes as possible. Soundscape-only classes
    # are included in training regardless. Classes with < MIN_CLIPS_FOR_VAL
    # total occurrences go entirely to training (no val holdout for them).
    df_train, df_val, sc_train, sc_val, train_only_cls = build_coverage_aware_split(
        file_df=full_df,
        soundscape_clips_df=sc_clips_all,
        min_clips_for_val=MIN_CLIPS_FOR_VAL,
    )

    # Log train_only classes count to W&B config after the fact
    run.config.update({
        "train_only_classes": len(train_only_cls),
        "min_clips_for_val": MIN_CLIPS_FOR_VAL,
        "use_soundscapes": sc_clips_all is not None,
    }, allow_val_change=True)

    # --- DataLoaders ---
    # Determined here (rather than only later, at "Engine Setup") so the
    # DataLoaders below can set pin_memory correctly for the actual device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    dset_train = BirbSet(
        df=df_train,
        root=os.path.join(root_path, 'train_audio'),
        clip_length=CLIP_LENGTH_SEC,
        label_to_idx=master_label_to_idx,
        is_train=True,
        use_pink_noise=USE_PINK_NOISE,
        use_white_noise=USE_WHITE_NOISE,
        use_spec_augment=USE_SPEC_AUGMENT,
        use_pitch_shift=USE_PITCH_SHIFT,
        use_esc50_noise=USE_ESC50_NOISE,
        esc50_path=ESC50_PATH if USE_ESC50_NOISE else None,
        esc50_categories=ESC50_CATEGORIES if USE_ESC50_NOISE else None,
        n_mels=N_MELS,
        n_fft=N_FFT,
        soundscape_clips_df=sc_train,
        raw_waveform_output=True,   # hand raw audio to Bird-MAE's own feature_extractor
    )

    # Create a dedicated, seeded generator for the sampler
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(42)

    # WeightedRandomSampler: inverse-frequency weights across ALL training clips
    # (both train.csv clips and soundscape clips), so rare species drawn from
    # soundscapes are up-sampled with the same logic as rare train.csv species.
    clip_label_counts = Counter(dset_train.labels)
    sample_weights = torch.tensor(
        [1.0 / clip_label_counts[lbl] for lbl in dset_train.labels],
        dtype=torch.float32
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True, generator=sampler_generator
    )

    loader = DataLoader(
        dset_train, batch_size=64, shuffle=False, sampler=sampler,
        pin_memory=(device.type == 'cuda'), num_workers=3, persistent_workers=True,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42)
    )

    dset_val = BirbSet(
        df=df_val,
        root=os.path.join(root_path, 'train_audio'),
        clip_length=CLIP_LENGTH_SEC,
        label_to_idx=master_label_to_idx,
        is_train=False,
        n_mels=N_MELS,
        n_fft=N_FFT,
        soundscape_clips_df=sc_val,
        raw_waveform_output=True,
    )
    loader_val = DataLoader(
        dset_val, batch_size=32, shuffle=False, pin_memory=(device.type == 'cuda'), num_workers=3,
        worker_init_fn=seed_worker, persistent_workers=True
    )


    # --- Engine Setup ---
    model = BirdMAEClassifier(
        model_name=BIRDMAE_MODEL_NAME,
        head=args.head,
        num_classes=num_classes,
        sample_rate=32000,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(device)

    # Only the classifier head is trained by default. If FREEZE_BACKBONE is
    # False, the (pretrained) backbone gets its own, much lower, LR so it
    # isn't blown away by the freshly-initialized head's larger updates.
    if FREEZE_BACKBONE:
        optimiser = torch.optim.AdamW(model.classifier.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    else:
        optimiser = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": HEAD_LR},
        ], weight_decay=1e-4)

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    # GradScaler is a no-op on CPU but still needs the right device string --
    # hardcoding 'cuda' here would crash on a CPU-only fallback.
    scaler = GradScaler(device.type)

    # SWA model: wraps the base model and maintains a running uniform average
    # of its weights starting from SWA_START_EPOCH. AveragedModel does not
    # participate in the forward/backward pass during training -- only the base
    # model is trained. We call swa_model.update_parameters(model) each epoch
    # after SWA_START_EPOCH, then run one BN update pass at the end.
    swa_model = AveragedModel(model)

    # Separate DataLoader for SWA BatchNorm update pass: no augmentation, no
    # sampler (we just need a stable forward pass over training data to
    # recompute running mean/var for the averaged weights). Smaller batch size
    # to avoid OOM during the extra pass.
    loader_bn = DataLoader(
        dset_train, batch_size=32, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=3,
        worker_init_fn=seed_worker
    )

    # --- LR Schedule: linear warmup for WARMUP_EPOCHS, then cosine decay.
    #     Warmup prevents the large pretrained weights from getting a destructive
    #     gradient kick on epoch 1 when the new head is still outputting noise.
    #     Cosine decay smoothly reduces LR rather than keeping it flat until
    #     early stopping, which typically recovers another small score bump in
    #     the final epochs. Used in virtually every top BirdCLEF solution. ---
    warmup_sched = LinearLR(
        optimiser, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = CosineAnnealingLR(
        optimiser, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimiser, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )

    # --- Training Loop ---
    patience_counter = 0
    best_val_auc = -1.0

    print("Starting training loop...")
    for epoch in range(1, MAX_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # --- Phase 1 -> Phase 2 transition ---
        # Cross from linear-probing (frozen encoder, head-only, HEAD_LR)
        # into full end-to-end fine-tuning (everything unfrozen, one
        # uniform, much lower BACKBONE_LR) exactly once, right at the
        # configured epoch boundary.
        if epoch == UNFREEZE_AFTER_EPOCH + 1 and model.freeze_backbone:
            print(f"\n{'='*50}")
            print(f"🔓 Epoch {epoch}: Phase 2 begins -- unfreezing backbone for full fine-tuning")
            print(f"{'='*50}")
            model.unfreeze_backbone()

            del optimiser
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Uniform LR across encoder + head -- deliberately NOT split into
            # per-group rates here, per the fine-tuning recipe: a single
            # conservative LR for the whole network during Phase 2.
            optimiser = torch.optim.AdamW(model.parameters(), lr=BACKBONE_LR, weight_decay=1e-4)

            # Fresh short warmup + cosine decay over the remaining epochs,
            # rather than reusing the Phase 1 scheduler (which was built for
            # a different optimizer and a 1e-3-scale peak LR).
            remaining_epochs = MAX_EPOCHS - epoch + 1
            phase2_warmup = min(PHASE2_WARMUP_EPOCHS, max(remaining_epochs - 1, 0))
            if phase2_warmup > 0:
                warmup_sched2 = LinearLR(
                    optimiser, start_factor=0.1, end_factor=1.0, total_iters=phase2_warmup
                )
                cosine_sched2 = CosineAnnealingLR(
                    optimiser, T_max=max(remaining_epochs - phase2_warmup, 1), eta_min=1e-7
                )
                scheduler = SequentialLR(
                    optimiser, schedulers=[warmup_sched2, cosine_sched2], milestones=[phase2_warmup]
                )
            else:
                scheduler = CosineAnnealingLR(optimiser, T_max=max(remaining_epochs, 1), eta_min=1e-7)

            run.config.update({"phase2_start_epoch": epoch}, allow_val_change=True)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params now: {trainable_params/1e6:.1f}M")

        epoch_start = time.time()
        train_loss, avg_grad_norm, nan_grad_batches = train_epoch(
            model, loader, optimiser, criterion, scaler, epoch,
            use_mixcut=USE_MIXCUT, mixcut_alpha=MIXCUT_ALPHA, mixcut_prob=MIXCUT_PROB,
            device=device
        )
        train_time = time.time() - epoch_start

        val_start = time.time()
        val_loss, val_auc, val_extra = validate_epoch(model, loader_val, criterion, epoch, device=device)
        val_time = time.time() - val_start

        epoch_time = time.time() - epoch_start
        current_lr = optimiser.param_groups[0]['lr']
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        train_throughput = len(dset_train) / train_time if train_time > 0 else 0.0

        if nan_grad_batches > 0:
            print(f"⚠️  {nan_grad_batches} batches had NaN/Inf gradients this epoch — GradScaler skipped those updates.")

        print(
            f"Epoch {epoch} Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f} | Val mAP: {val_extra['val_map']:.4f} | "
            f"Grad Norm: {avg_grad_norm:.3f} | NaN Batches: {nan_grad_batches} | "
            f"Classes Scored: {val_extra['num_valid_classes']}/{num_classes} | "
            f"Epoch Time: {epoch_time:.1f}s"
        )
        run.log({
            "Training Loss": train_loss,
            "Val Loss": val_loss,
            "Val ROC-AUC": val_auc,
            "Patience": patience_counter / PATIENCE,
            "Grad Norm (avg)": avg_grad_norm,
            "NaN Grad Batches": nan_grad_batches,
            "Learning Rate": current_lr,
            "Epoch Time (s)": epoch_time,
            "Train Time (s)": train_time,
            "Val Time (s)": val_time,
            "Train Throughput (samples_per_sec)": train_throughput,
            "GPU Peak Memory (MB)": peak_mem_mb,
            "Val Classes Scored": val_extra["num_valid_classes"],
            "Val Classes Skipped": val_extra["num_skipped_classes"],
            "Val AUC Min (per-class)": val_extra["auc_min"],
            "Val AUC Max (per-class)": val_extra["auc_max"],
            "Val AUC Std (per-class)": val_extra["auc_std"],
            "Val Mean Prob (positives)": val_extra["mean_prob_positive"],
            "Val Mean Prob (negatives)": val_extra["mean_prob_negative"],
            # mAP — logged alongside ROC-AUC so you can watch whether they
            # diverge. Large AUC–mAP gap = calibration or rare-class recall
            # problem rather than a ranking problem.
            "Val mAP": val_extra["val_map"],
            "Val mAP Classes Scored": val_extra["num_map_classes"],
            "Val AP Min (per-class)": val_extra["map_min"],
            "Val AP Max (per-class)": val_extra["map_max"],
            "Val AP Std (per-class)": val_extra["map_std"],
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Guard against saving a checkpoint whose weights contain NaN --
            # the GradScaler skips updates for NaN-gradient batches, but if
            # NaN somehow propagated into weights (e.g. from a corrupted audio
            # file producing NaN activations), the saved model would produce
            # NaN predictions at inference time, collapsing the LB score.
            nan_params = [n for n, p in model.named_parameters() if p.isnan().any()]
            if nan_params:
                print(f"⛔ NaN detected in {len(nan_params)} parameter tensor(s) — skipping checkpoint save.")
                print(f"   Affected: {nan_params[:5]}{'...' if len(nan_params) > 5 else ''}")
            else:
                print(f"--> 🔥 New Best Model Saved! (ROC-AUC: {best_val_auc:.4f})")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_val_auc,
                }, checkpoint_path)
                artifact = wandb.Artifact(name=artifact_name, type="model", metadata={"epoch": epoch})
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact)
        else:
            patience_counter += 1
            print(f"--> No improvement. Early stopping patience: {patience_counter}/{PATIENCE}")

        # Step scheduler every epoch regardless of improvement. SequentialLR
        # handles the warmup/cosine transition automatically at WARMUP_EPOCHS.
        scheduler.step()

        # SWA weight averaging: start accumulating after SWA_START_EPOCH.
        # This runs after scheduler.step() so the LR logged next epoch still
        # reflects the cosine schedule, not any SWA override.
        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            print(f"   📦 SWA updated (averaging since epoch {SWA_START_EPOCH})")

        print("-" * 50)

        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered at Epoch {epoch}. Model has plateaued.")
            break

    # ==========================================
    # SWA FINALISATION
    # ==========================================
    # AveragedModel's BatchNorm running statistics reflect whichever individual
    # checkpoint was loaded last, not the averaged weights. update_bn fixes this
    # by running one forward-only pass through the training data with the SWA
    # weights frozen, recomputing the running mean/var from scratch.
    swa_epochs_collected = max(0, epoch - SWA_START_EPOCH + 1)
    has_batchnorm = any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                         for m in model.modules())
    if swa_epochs_collected > 0 and has_batchnorm:
        print(f"\n🔁 Running SWA BatchNorm update pass ({swa_epochs_collected} epochs averaged)...")
        swa_model.to(device)
        # swa_update_bn expects (loader, model) and internally calls model.train()
        # then does forward passes; it does NOT call backward, so no gradients.
        swa_update_bn(loader_bn, swa_model, device=device)
    elif swa_epochs_collected > 0:
        # BirdMAEClassifier has no BatchNorm layers (frozen backbone + a plain
        # Linear head), so there's no running mean/var to recompute -- running
        # a full extra forward pass over the training set through the frozen
        # Bird-MAE backbone here would just burn CPU time for nothing.
        print(f"\nℹ️  No BatchNorm layers in model -- skipping BN update pass "
              f"({swa_epochs_collected} epochs averaged).")
        swa_model.to(device)

        print("🔍 Validating SWA model...")
        swa_val_loss, swa_val_auc, swa_val_extra = validate_epoch(
            swa_model, loader_val, criterion, epoch=0, device=device
        )
        print(f"SWA Model → Val Loss: {swa_val_loss:.4f} | Val ROC-AUC: {swa_val_auc:.4f} | "
              f"Classes Scored: {swa_val_extra['num_valid_classes']}/{num_classes}")
        run.log({"SWA Val ROC-AUC": swa_val_auc, "SWA Val Loss": swa_val_loss})

        # swa_model.module is the underlying BirdMAEClassifier with averaged weights.
        swa_checkpoint_path = checkpoint_path.replace(".pth", "_swa.pth")
        nan_params_swa = [n for n, p in swa_model.module.named_parameters() if p.isnan().any()]
        if nan_params_swa:
            print(f"⛔ NaN in SWA model — skipping SWA checkpoint save.")
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': swa_model.module.state_dict(),
                'best_val_auc': swa_val_auc,
                'swa_epochs': swa_epochs_collected,
            }, swa_checkpoint_path)
            print(f"💾 SWA checkpoint saved → {swa_checkpoint_path}")
            swa_artifact = wandb.Artifact(
                name=artifact_name + "_swa", type="model",
                metadata={"epoch": epoch, "swa_epochs": swa_epochs_collected}
            )
            swa_artifact.add_file(local_path=swa_checkpoint_path)
            run.log_artifact(swa_artifact)
    else:
        print(f"\nℹ️  SWA_START_EPOCH ({SWA_START_EPOCH}) not reached — no SWA model produced.")

    run.finish()


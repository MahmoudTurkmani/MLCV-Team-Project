import os
import ast
import glob
import time
import math
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
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm  # Changed from tqdm.notebook for terminal compatibility

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

class BirbSet(Dataset):
    def __init__(self, df, root, clip_length, label_to_idx, is_train=False,
                 use_pink_noise=False, use_white_noise=False, use_spec_augment=False,
                 use_pitch_shift=False, use_esc50_noise=False, esc50_path=None,
                 esc50_categories=None, n_mels=128, n_fft=1024):
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
            # Support absolute paths if provided by the dataframe, fallback to root + filename
            curr_audio_loc = entry.get('filepath', os.path.join(self.root, os.path.normpath(entry["filename"])))
            
            try:
                info = torchaudio.info(curr_audio_loc)
                duration = info.num_frames / self.sample_rate
            except Exception as e:
                print(f"Skipping metadata read error for {curr_audio_loc}: {e}")
                continue
            
            # If the dataframe row provides exact timestamps (Soundscape data)
            if not pd.isna(entry.get('start_sec')) and not pd.isna(entry.get('end_sec')):
                self.clips.append(curr_audio_loc)
                self.labels.append(self.label_to_idx[entry['primary_label']])
                self.ratings.append(entry.get('rating'))
                self.secondary_labels.append(entry.get('secondary_labels', '[]'))
                self.start_times.append(float(entry['start_sec']))
                self.end_times.append(float(entry['end_sec']))
            else:
                # Focal recording: chunk the entire duration
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


# ==========================================
# MODEL DEFINITION
# ==========================================
class EfficientBirbNN(nn.Module):
    def __init__(self, num_classes=234, pretrained=True):
        super().__init__()
        
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.base_model = efficientnet_b3(weights=weights)
        
        original_conv = self.base_model.features[0][0]
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels, \
            kernel_size=original_conv.kernel_size, \
            stride=original_conv.stride, \
            padding=original_conv.padding, \
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
# TRAINING & VALIDATION FUNCTIONS
# ==========================================
def train_epoch(model, dataloader, optimizer, criterion, scaler, epoch,
                 use_mixcut=False, mixcut_alpha=0.4, mixcut_prob=0.5):
    model.train()
    total_loss     = 0.0
    total_grad_norm = 0.0
    num_batches    = 0
    nan_grad_batches = 0   # batches where GradScaler detected NaN and skipped the update
    
    if len(dataloader) == 0:
        print("Warning: DataLoader has 0 batches. Check your dataset.")
        return 0.0, 0.0, 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)
    
    for spectrograms, targets in pbar:
        spectrograms = spectrograms.to('cuda', non_blocking=True)
        targets = targets.to('cuda', dtype=torch.float32, non_blocking=True)
        
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        if use_mixcut:
            spectrograms, targets = apply_mixcut(
                spectrograms, targets, alpha=mixcut_alpha, cutmix_prob=mixcut_prob
            )
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            logits = model(spectrograms)
            loss   = criterion(logits, targets)
            
        scaler.scale(loss).backward()

        # Unscale before clipping so max_norm applies to true gradient magnitude.
        # max_norm=1.0 actively clips rather than just measuring -- this is the
        # fix for the NaN gradient problem: large/NaN gradient contributions
        # from the focal weighting backward pass are clamped before they reach
        # the weight update, and the GradScaler will additionally skip the step
        # entirely for any batch where NaN gradients survive after clipping.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_val = float(grad_norm)

        # NaN-safe accumulation: a single NaN in the running sum would make
        # the entire epoch average NaN (the original bug). Track NaN batches
        # separately so the average reflects only the batches where a valid
        # gradient step actually occurred.
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
def validate_epoch(model, dataloader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", dynamic_ncols=True)
    
    for spectrograms, targets in pbar:
        spectrograms = spectrograms.to('cuda', non_blocking=True)
        targets = targets.to('cuda', dtype=torch.float32, non_blocking=True)
        
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
            
        with autocast('cuda'):
            logits = model(spectrograms)
            loss = criterion(logits, targets)
            
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    binary_targets = (all_targets > 0.0).astype(int)

    # > 0.0 instead of > 0.5: secondary labels from BirbSet are encoded at
    # confidence * 0.3 (max value 0.3), so the old > 0.5 threshold silently
    # treated every secondary-label positive as a negative -- any time the
    # model correctly identified a background species it was counted as a
    # false positive. > 0.0 includes them as ground-truth targets.
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
            # Per-class breakdown (not logged as a single number) -- used below
            # to report how uneven performance is across classes, since macro
            # AUC alone can hide a handful of classes the model is failing on.
            per_class_auc = roc_auc_score(
                binary_targets[:, valid_classes],
                all_preds[:, valid_classes],
                average=None
            )
        except ValueError as e:
            print(f"ROC-AUC calculation error: {e}")
            val_auc = 0.0

    # --- Extra diagnostics ---
    mean_prob_pos, mean_prob_neg = float('nan'), float('nan')
    if num_valid_classes > 0:
        scored_targets = binary_targets[:, valid_classes].astype(bool)
        scored_preds = all_preds[:, valid_classes]
        if scored_targets.any():
            mean_prob_pos = float(scored_preds[scored_targets].mean())
        if (~scored_targets).any():
            mean_prob_neg = float(scored_preds[~scored_targets].mean())

    # --- Post-hoc mAP ---
    # mAP (mean Average Precision) only requires at least one positive example
    # per class — it does not need negatives — so its valid_classes filter is
    # slightly more inclusive than ROC-AUC's. In practice with a large val set
    # almost every class that has positives also has negatives, so the
    # difference is usually small, but it's worth tracking both:
    #   ROC-AUC: measures rank-order discrimination (how well the model
    #            separates species from non-species globally).
    #   mAP:     measures precision-recall trade-off, which is more sensitive
    #            to rare-species recall and tends to be more pessimistic when
    #            the model is miscalibrated on low-frequency classes.
    # Both are logged to W&B so you can watch whether they diverge across runs
    # (large divergence = calibration or rare-class problem rather than a
    # general ranking problem).
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
        # mAP fields
        "val_map":          val_map,
        "num_map_classes":  num_map_classes,
        "map_min":  float(per_class_ap.min()) if per_class_ap is not None else float('nan'),
        "map_max":  float(per_class_ap.max()) if per_class_ap is not None else float('nan'),
        "map_std":  float(per_class_ap.std()) if per_class_ap is not None else float('nan'),
    }

    return total_loss / len(dataloader), val_auc, extra_metrics

# ==========================================
# VALIDATION COVERAGE FIX
# ==========================================
def prioritize_train_coverage(full_df, df_train, df_val, label_col='primary_label', filename_col='filename'):
    """
    Ensures that classes are prioritized for training. 
    1. Singleton classes (count == 1) are forced into the training set.
    2. Classes completely missing from training (but have >1 example) are rescued from validation.
    3. Classes completely missing from validation (but have >1 example) are rescued from training.
    """
    all_classes = set(full_df[label_col].unique())
    class_counts = full_df[label_col].value_counts()
    singletons = set(class_counts[class_counts == 1].index.tolist())
    
    # 1. Force singletons into the training set
    for cls in singletons:
        if cls in df_val[label_col].values:
            move_mask = df_val[label_col] == cls
            rows_moved = df_val[move_mask].copy()
            df_val = df_val[~move_mask].reset_index(drop=True)
            df_train = pd.concat([df_train, rows_moved], ignore_index=True)
            
    # 2. Rescue classes missing from training (count > 1)
    train_classes = set(df_train[label_col].unique())
    missing_in_train = sorted(all_classes - train_classes)
    
    for cls in missing_in_train:
        candidate_files = df_val.loc[df_val[label_col] == cls, filename_col].unique()
        if len(candidate_files) > 0:
            chosen_file = candidate_files[0]
            move_mask = df_val[filename_col] == chosen_file
            rows_moved = df_val[move_mask].copy()
            df_val = df_val[~move_mask].reset_index(drop=True)
            df_train = pd.concat([df_train, rows_moved], ignore_index=True)

    # 3. Rescue classes missing from validation (Only if count > 1)
    val_classes = set(df_val[label_col].unique())
    missing_in_val = sorted(all_classes - val_classes)
    
    for cls in missing_in_val:
        if cls in singletons:
            continue # Leave singletons in train
            
        candidate_files = df_train.loc[df_train[label_col] == cls, filename_col].unique()
        if len(candidate_files) > 0:
            chosen_file = candidate_files[0]
            move_mask = df_train[filename_col] == chosen_file
            rows_moved = df_train[move_mask].copy()
            df_train = df_train[~move_mask].reset_index(drop=True)
            df_val = pd.concat([df_val, rows_moved], ignore_index=True)

    final_train_classes = len(set(df_train[label_col].unique()))
    final_val_classes = len(set(df_val[label_col].unique()))
    print(f"✅ Coverage Check -> Train: {final_train_classes}/{len(all_classes)} classes | Val: {final_val_classes}/{len(all_classes)} classes.")
    
    return df_train, df_val

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    root_path       = os.path.join("..", "birdclef-2026")
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

    # --- AUGMENTATION TOGGLES ---
    USE_PINK_NOISE   = False    # colored noise: 1/f pink spectrum
    USE_WHITE_NOISE  = False   # colored noise: flat white spectrum
    USE_SPEC_AUGMENT = False    # frequency + time masking on the mel spectrogram
    USE_PITCH_SHIFT  = False   # ±3 semitone pitch shift via resampling trick
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
    USE_MIXCUT       = False    # batch-level MixUp / CutMix
    MIXCUT_ALPHA     = 0.4     # Beta distribution shape param; lower = milder mixing
    MIXCUT_PROB      = 0.5     # probability of CutMix vs MixUp when mixcut fires

    tag_parts = []
    if USE_PINK_NOISE:   tag_parts.append("pink")
    if USE_WHITE_NOISE:  tag_parts.append("white")
    if USE_SPEC_AUGMENT: tag_parts.append("spec")
    if USE_PITCH_SHIFT:  tag_parts.append("pitch")
    if USE_ESC50_NOISE:  tag_parts.append("esc50")
    if USE_MIXCUT:       tag_parts.append("mixcut")
    run_tag = "_".join(tag_parts) if tag_parts else "baseline"

    run_name        = f"Training {run_tag}"
    artifact_name   = f"cnn_bird_model_{run_tag}"
    checkpoint_path = f"best_efficientbirb_model_{run_tag}.pth"

    # --- Setup Logging ---
    run = wandb.init(
        entity="pumpkin_person-tu-dresden",
        project="CNN-Birds",
        name=run_name,
        config={
            "learning_rate": 0.0001,
            "architecture": "CNN",
            "dataset": "BirdClef+ 2026",
            "epochs": MAX_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "focal_gamma": FOCAL_GAMMA,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "swa_start_epoch": SWA_START_EPOCH,
            "use_pink_noise": USE_PINK_NOISE,
            "use_white_noise": USE_WHITE_NOISE,
            "use_spec_augment": USE_SPEC_AUGMENT,
            "use_pitch_shift": USE_PITCH_SHIFT,
            "use_esc50_noise": USE_ESC50_NOISE,
            "esc50_categories": ESC50_CATEGORIES if USE_ESC50_NOISE else [],
            "use_mixcut": USE_MIXCUT,
            "mixcut_alpha": MIXCUT_ALPHA,
            "mixcut_prob": MIXCUT_PROB,
        },
    )

    # --- Load Data & Setup Splits ---
    # 1. Prepare Focal Recordings
    focal_df = pd.read_csv(os.path.join(root_path, "train.csv"))
    focal_df['filepath'] = focal_df['filename'].apply(lambda x: os.path.join(root_path, "train_audio", os.path.normpath(x)))
    focal_df['start_sec'] = np.nan
    focal_df['end_sec'] = np.nan

    # 2. Prepare Soundscapes
    ss_labels_path = os.path.join(root_path, "train_soundscape_labels.csv")
    if os.path.exists(ss_labels_path):
        ss_raw = pd.read_csv(ss_labels_path)
        ss_rows = []
        for _, row in ss_raw.iterrows():
            row_id = row['row_id']
            birds_str = row['birds']
            
            # Extract filename and timestamps from row_id (e.g., soundscape_name_5)
            parts = str(row_id).rsplit('_', 1)
            filename = parts[0] + ".ogg"
            end_sec = float(parts[1])
            start_sec = end_sec - 5.0
            
            # Parse bird array
            if isinstance(birds_str, str):
                birds = ast.literal_eval(birds_str) if birds_str.startswith('[') else birds_str.split()
            else:
                birds = ['nocall']
                
            # Skip pure nocall segments to prevent swamping the dataset
            if not birds or birds == ['nocall']:
                continue 
                
            primary = birds[0]
            secondary = str(birds[1:])
            
            ss_rows.append({
                'filename': filename,
                'filepath': os.path.join(root_path, "train_soundscapes", os.path.normpath(filename)),
                'primary_label': primary,
                'secondary_labels': secondary,
                'rating': 5.0, # Expert annotated soundscapes are high confidence
                'start_sec': start_sec,
                'end_sec': end_sec
            })
            
        ss_df = pd.DataFrame(ss_rows)
        full_df = pd.concat([focal_df, ss_df], ignore_index=True)
        print(f"Merged Data: {len(focal_df)} focal files + {len(ss_df)} soundscape segments.")
    else:
        full_df = focal_df
        print("No train_soundscape_labels.csv found. Proceeding with focal data only.")

    unique_labels = pd.read_csv(os.path.join(root_path, "taxonomy.csv"))
    master_label_to_idx = {label: i for i, label in enumerate(unique_labels['primary_label'].unique())}
    num_classes         = len(master_label_to_idx)

    # 3. K-Fold Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = next(
        sgkf.split(X=full_df, y=full_df['primary_label'], groups=full_df['filename'])
    )

    df_train = full_df.iloc[train_indices].reset_index(drop=True)
    df_val   = full_df.iloc[val_indices].reset_index(drop=True)
    
    # 4. Apply new training-first priority logic
    df_train, df_val = prioritize_train_coverage(full_df, df_train, df_val)
    print(f"After coverage fix — Train: {len(df_train)} | Val: {len(df_val)}")

    # --- DataLoaders ---
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
    )

    # --- WeightedRandomSampler: give each clip a weight inversely proportional
    #     to its primary-label frequency across all training clips, then sample
    #     WITH replacement so that rare species appear in roughly equal proportion
    #     to common ones. This is one of the most impactful non-augmentation
    #     improvements in past BirdCLEF winning solutions -- without it the model
    #     sees e.g. 10x more clips of common species than rare ones per epoch,
    #     which is the primary reason rare classes score near chance on the LB.
    #
    #     Note: sampler and shuffle=True are mutually exclusive in PyTorch;
    #     the sampler takes over the shuffling role. ---
    clip_label_counts = Counter(dset_train.labels)
    sample_weights = torch.tensor(
        [1.0 / clip_label_counts[lbl] for lbl in dset_train.labels],
        dtype=torch.float32
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(
        dset_train, batch_size=64, shuffle=False, sampler=sampler,
        pin_memory=True, num_workers=3,
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
    )
    loader_val = DataLoader(
        dset_val, batch_size=32, shuffle=False, pin_memory=True, num_workers=3,
        worker_init_fn=seed_worker
    )

    # --- Engine Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientBirbNN(num_classes=num_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    scaler = GradScaler('cuda')

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
        dset_train, batch_size=32, shuffle=True, pin_memory=True, num_workers=3,
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

        epoch_start = time.time()
        train_loss, avg_grad_norm, nan_grad_batches = train_epoch(
            model, loader, optimiser, criterion, scaler, epoch,
            use_mixcut=USE_MIXCUT, mixcut_alpha=MIXCUT_ALPHA, mixcut_prob=MIXCUT_PROB
        )
        train_time = time.time() - epoch_start

        val_start = time.time()
        val_loss, val_auc, val_extra = validate_epoch(model, loader_val, criterion, epoch)
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
    if swa_epochs_collected > 0:
        print(f"\n🔁 Running SWA BatchNorm update pass ({swa_epochs_collected} epochs averaged)...")
        swa_model.to(device)
        # swa_update_bn expects (loader, model) and internally calls model.train()
        # then does forward passes; it does NOT call backward, so no gradients.
        swa_update_bn(loader_bn, swa_model, device=device)

        print("🔍 Validating SWA model...")
        swa_val_loss, swa_val_auc, swa_val_extra = validate_epoch(
            swa_model, loader_val, criterion, epoch=0
        )
        print(f"SWA Model → Val Loss: {swa_val_loss:.4f} | Val ROC-AUC: {swa_val_auc:.4f} | "
              f"Classes Scored: {swa_val_extra['num_valid_classes']}/{num_classes}")
        run.log({"SWA Val ROC-AUC": swa_val_auc, "SWA Val Loss": swa_val_loss})

        # swa_model.module is the underlying EfficientBirbNN with averaged weights.
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

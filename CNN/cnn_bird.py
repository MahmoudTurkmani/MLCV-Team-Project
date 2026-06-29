import os
import ast
import glob
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
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
    def __init__(self, df, root, clip_length, label_to_idx, is_train=False, use_pink_noise=False, use_spec_augment=False):
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
        self.use_spec_augment = use_spec_augment

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        self.mel_spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=800,
            n_mels=64
        )
        
        # Only initialize masking modules if we are training AND SpecAugment is enabled
        if self.is_train and self.use_spec_augment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=24)
        
        for _, entry in df.iterrows():
            curr_audio_loc = os.path.join(self.root, os.path.normpath(entry["filename"]))
            try:
                info = torchaudio.info(curr_audio_loc)
                duration = info.num_frames / self.sample_rate
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

            # --- Check Pink Noise Flag ---
            if self.is_train and self.use_pink_noise and np.random.rand() < 0.5:
                snr = np.random.uniform(10.0, 25.0)
                waveform = self._add_pink_noise(waveform, snr_db=snr)

            # Convert to dB
            spectrogram = self.mel_spect(waveform)
            spectrogram = self.amp_to_db(spectrogram)
            
            # Normalize
            mean, std   = spectrogram.mean(), spectrogram.std() + 1e-6
            spectrogram = (spectrogram - mean) / std

            # --- Check SpecAugment Flag ---
            if self.is_train and self.use_spec_augment:
                if np.random.rand() < 0.5:
                    spectrogram = self.freq_mask(spectrogram)
                if np.random.rand() < 0.5:
                    spectrogram = self.time_mask(spectrogram)

            target = torch.zeros(len(self.label_to_idx), dtype=torch.float32)
            primary = self.labels[idx]
            rating = self.ratings[idx]
            
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
    def __init__(self, audio_path, clip_length=5.0, sample_rate=32000):
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
            sample_rate=self.sample_rate, n_fft=800, n_mels=64
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
# TRAINING & VALIDATION FUNCTIONS
# ==========================================
def train_epoch(model, dataloader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss = 0.0
    
    if len(dataloader) == 0:
        print("Warning: DataLoader has 0 batches. Check your dataset.")
        return 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)
    
    for spectrograms, targets in pbar:
        spectrograms = spectrograms.to('cuda', non_blocking=True)
        targets = targets.to('cuda', dtype=torch.float32, non_blocking=True)
        
        if targets.ndim == 1:
            targets = targets.unsqueeze(1) 
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            logits = model(spectrograms)
            loss = criterion(logits, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix({'loss': f"{loss_val:.4f}"})
        
    return total_loss / len(dataloader)


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
    
    binary_targets = (all_targets > 0.5).astype(int)
    valid_classes = np.any(binary_targets == 1, axis=0)
    
    if not np.any(valid_classes):
        print("Warning: No valid classes found for mAP calculation in this split.")
        val_map = 0.0
    else:
        try:
            val_map = average_precision_score(
                binary_targets[:, valid_classes], 
                all_preds[:, valid_classes], 
                average='macro'
            )
        except ValueError as e:
            print(f"mAP calculation error: {e}")
            val_map = 0.0
            
    return total_loss / len(dataloader), val_map

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    root_path       = os.path.join("..", "birdclef-2026")
    CLIP_LENGTH_SEC = 5.0  
    MAX_EPOCHS      = 30
    PATIENCE        = 12  

    # --- AUGMENTATION TOGGLES ---
    USE_PINK_NOISE   = True   # Set to False to disable
    USE_SPEC_AUGMENT = True   # Set to False to disable

    # --- Setup Logging ---
    run = wandb.init(
        entity="pumpkin_person-tu-dresden",
        project="CNN-Birds",
        name="Training Pink Spec",
        config={
            "learning_rate": 0.0001,
            "architecture": "CNN",
            "dataset": "BirdClef+ 2026",
            "epochs": MAX_EPOCHS,
            "use_pink_noise": USE_PINK_NOISE,
            "use_spec_augment": USE_SPEC_AUGMENT
        },
    )

    # --- Load Data & Setup Splits ---
    full_df = pd.read_csv(os.path.join(root_path, "train.csv"))
    unique_labels = pd.read_csv(os.path.join(root_path, "taxonomy.csv"))
    master_label_to_idx = {label: i for i, label in enumerate(unique_labels['primary_label'].unique())}
    num_classes         = len(master_label_to_idx)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = next(
        sgkf.split(X=full_df, y=full_df['primary_label'], groups=full_df['filename'])
    )

    df_train = full_df.iloc[train_indices].reset_index(drop=True)
    df_val   = full_df.iloc[val_indices].reset_index(drop=True)
    print(f"Train samples: {len(df_train)} | Validation samples: {len(df_val)}")

    # --- DataLoaders ---
    dset_train = BirbSet(
        df=df_train, 
        root=os.path.join(root_path, 'train_audio'), 
        clip_length=CLIP_LENGTH_SEC,
        label_to_idx=master_label_to_idx, 
        is_train=True,
        use_pink_noise=USE_PINK_NOISE,
        use_spec_augment=USE_SPEC_AUGMENT
    )
    loader = DataLoader(dset_train, batch_size=32, shuffle=True, pin_memory=True, num_workers=3)

    dset_val = BirbSet(
        df=df_val, 
        root=os.path.join(root_path, 'train_audio'), 
        clip_length=CLIP_LENGTH_SEC,
        label_to_idx=master_label_to_idx, 
        is_train=False  
    )
    loader_val = DataLoader(dset_val, batch_size=32, shuffle=False, pin_memory=True, num_workers=3)

    # --- Engine Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientBirbNN(num_classes=num_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # --- Training Loop ---
    patience_counter = 0
    best_val_map = -1.0

    print("Starting training loop...")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, loader, optimiser, criterion, scaler, epoch)
        val_loss, val_map = validate_epoch(model, loader_val, criterion, epoch)

        print(f"Epoch {epoch} Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}")
        run.log({"Training Loss": train_loss, "Val Loss": val_loss, "Val mAP": val_map, "Patience": patience_counter/PATIENCE})

        if epoch % 2 == 0:
            if val_map > best_val_map:
                best_val_map = val_map
                patience_counter = 0  
            
                print(f"--> 🔥 New Best Model Saved! (mAP: {best_val_map:.4f})")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'best_val_map': best_val_map,
            }, "best_efficientbirb_model.pth")

            artifact = wandb.Artifact(name="cnn_bird_model_pink_spec", type="model", metadata={"epoch": epoch})
            artifact.add_file(local_path="best_efficientbirb_model.pth")
            run.log_artifact(artifact)

        else:
            patience_counter += 1
            print(f"--> No improvement. Early stopping patience: {patience_counter}/{PATIENCE}")

        print("-" * 50)

        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered at Epoch {epoch}. Model has plateaued.")
            break

    run.finish()

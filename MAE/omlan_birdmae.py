#!/usr/bin/env python3
"""
Bird-MAE-Base Fine-tuning — BirdCLEF+ 2026
Model  : DBD-research-group/Bird-MAE-Base (85.5M params)
Loss   : Focal Loss (gamma=2)
Sampler: WeightedRandomSampler (rare-species oversampling)
Phase 1 (epochs 1-5) : freeze encoder, train head only  (LR=1e-3)
Phase 2 (epochs 6-30): unfreeze all, full fine-tune      (LR=1e-5)
"""

import os, sys, time, json, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import librosa
import wandb

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("../", "birdclef-2026")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = 'DBD-research-group/Bird-MAE-Base'
SAMPLE_RATE    = 32000
TARGET_SAMPLES = 32000 * 5   # 160000 samples = 5 seconds
EMBED_DIM      = 768          # Bird-MAE-Base output dimension (confirmed)
BATCH_SIZE     = 32           # H100 has 80GB — comfortable at 32
NUM_EPOCHS     = 30
PHASE2_START   = 6            # epoch at which encoder is unfrozen
LR_HEAD        = 1e-3         # phase 1: only classifier head trained
LR_FINETUNE    = 1e-5         # phase 2: full model, low LR
FOCAL_GAMMA    = 2.0
NUM_WORKERS    = 4
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    (1 - p_t)^gamma suppresses easy examples, amplifies gradient
    from hard/rare-species predictions. Replaces BCE.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none')
        p_t   = torch.sigmoid(logits) * targets + \
                (1.0 - torch.sigmoid(logits)) * (1.0 - targets)
        loss  = (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


# ── Dataset ───────────────────────────────────────────────────────────────────
class BirdMAEDataset(Dataset):
    """
    Returns raw waveform processed through Bird-MAE feature extractor.
    The extractor computes mel internally → shape [1, 512, 128] per sample.
    This is different from our AST pipeline which uses librosa mels.
    """
    def __init__(self, df, label_to_idx, feature_extractor, augment=False):
        self.df            = df.reset_index(drop=True)
        self.label_to_idx  = label_to_idx
        self.fe            = feature_extractor
        self.augment       = augment
        self.num_classes   = len(label_to_idx)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            return np.zeros(TARGET_SAMPLES, dtype=np.float32)

        if len(audio) < TARGET_SAMPLES:
            # Pad with zeros if clip is shorter than 5 seconds
            audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
        elif len(audio) > TARGET_SAMPLES:
            if self.augment:
                # Random crop during training — data augmentation
                start = np.random.randint(0, len(audio) - TARGET_SAMPLES)
                audio = audio[start : start + TARGET_SAMPLES]
            else:
                # Center crop during validation — deterministic
                start = (len(audio) - TARGET_SAMPLES) // 2
                audio = audio[start : start + TARGET_SAMPLES]

        return audio.astype(np.float32)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(DATA_DIR, 'train_audio', row['filename'])
        audio = self._load_audio(path)

        # Bird-MAE feature extractor handles mel computation internally.
        # Output: [1, 1, 512, 128] → squeeze batch dim → [1, 512, 128]
        mel = self.fe(audio).squeeze(0)   # [1, 512, 128]

        # Multi-label target vector
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        species = row['primary_label']
        if species in self.label_to_idx:
            label[self.label_to_idx[species]] = 1.0

        return mel, label


# ── Model ─────────────────────────────────────────────────────────────────────
class BirdMAEClassifier(nn.Module):
    """
    Bird-MAE encoder + classification head.
    Encoder output is already globally pooled → [B, 768].
    No additional pooling needed (confirmed from probe).
    """
    def __init__(self, model_id: str, num_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM, num_classes)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 1, 512, 128]
        out       = self.encoder(mel)
        embedding = out.last_hidden_state  # [B, 768] — already pooled
        return self.head(embedding)        # [B, num_classes]

    def freeze_encoder(self):
        """Phase 1: only head trains."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        print('Encoder FROZEN — training head only')

    def unfreeze_encoder(self):
        """Phase 2: full model trains."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        print('Encoder UNFROZEN — full fine-tuning')


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_labels: np.ndarray, all_probs: np.ndarray):
    """Compute mAP and ROC-AUC. Skip classes with no positive val samples."""
    aps = []
    for c in range(all_labels.shape[1]):
        if all_labels[:, c].sum() > 0:
            aps.append(average_precision_score(all_labels[:, c], all_probs[:, c]))
    map_score = float(np.mean(aps)) if aps else 0.0

    try:
        roc = float(roc_auc_score(all_labels, all_probs, average='macro'))
    except Exception:
        roc = 0.0

    return map_score, roc


# ── WeightedRandomSampler ─────────────────────────────────────────────────────
def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Rare species (few samples) get higher sampling probability.
    Weight for each sample = 1 / count_of_its_species.
    """
    counts  = df['primary_label'].value_counts()
    weights = df['primary_label'].map(lambda x: 1.0 / counts.get(x, 1))
    return WeightedRandomSampler(
        weights    = torch.tensor(weights.values, dtype=torch.float32),
        num_samples = len(df),
        replacement = True
    )


# ── Train / Val ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(mel)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        logits = model(mel)
        total_loss += criterion(logits, labels).item()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    map_score, roc = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader), map_score, roc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Device : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # ── Data ─────────────────────────────────────────────────────────────────
    train_csv = pd.read_csv(f'{DATA_DIR}/train.csv')

    # Load label map from taxonomy.csv — guarantees identical 214-class
    # output space across ALL models in the paper. Do NOT rebuild from train.csv
    # (train.csv has only 206 species; 8 taxonomy species have zero training audio).
    taxonomy_df  = pd.read_csv(f'{DATA_DIR}/taxonomy.csv')
    label_to_idx = {label: i for i, label in enumerate(taxonomy_df['primary_label'].unique())}
    num_classes  = len(label_to_idx)
    print(f'Classes: {num_classes} (loaded from taxonomy.csv)')

    # Same train/val split as all other models (random_state=42, stratified)
    single_mask    = train_csv['primary_label'].map(
                        train_csv['primary_label'].value_counts() == 1)
    single_df      = train_csv[single_mask]
    multi_df       = train_csv[~single_mask]

    train_multi, val_df = train_test_split(
        multi_df, test_size=0.15,
        stratify=multi_df['primary_label'], random_state=42
    )
    train_df = pd.concat([train_multi, single_df]).reset_index(drop=True)
    print(f'Train  : {len(train_df)} | Val: {len(val_df)}')

    # ── Feature Extractor ─────────────────────────────────────────────────────
    print('Loading Bird-MAE feature extractor...')
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_dataset = BirdMAEDataset(train_df, label_to_idx, fe, augment=True)
    val_dataset   = BirdMAEDataset(val_df,   label_to_idx, fe, augment=False)

    sampler = make_sampler(train_df)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print('Loading Bird-MAE-Base encoder...')
    model     = BirdMAEClassifier(MODEL_ID, num_classes).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

    # Phase 1 — freeze encoder, train head only
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.init(
        project='birdclef-2026',
        name='bird_mae_base_v1',
        config={
            'model'        : MODEL_ID,
            'batch_size'   : BATCH_SIZE,
            'lr_head'      : LR_HEAD,
            'lr_finetune'  : LR_FINETUNE,
            'focal_gamma'  : FOCAL_GAMMA,
            'phase2_start' : PHASE2_START,
            'num_classes'  : num_classes,
            'num_epochs'   : NUM_EPOCHS,
        }
    )

    best_map = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Phase 2 switch ────────────────────────────────────────────────────
        if epoch == PHASE2_START:
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4
            )

        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_map, val_roc = val_epoch(model, val_loader, criterion, DEVICE)
        elapsed    = time.time() - t0

        phase = 1 if epoch < PHASE2_START else 2
        print(
            f'[P{phase}] Ep {epoch:02d}/{NUM_EPOCHS} | '
            f'Loss {train_loss:.4f}/{val_loss:.4f} | '
            f'mAP {val_map:.4f} | ROC {val_roc:.4f} | '
            f'{elapsed:.0f}s'
        )

        wandb.log({
            'epoch'      : epoch,
            'phase'      : phase,
            'train_loss' : train_loss,
            'val_loss'   : val_loss,
            'val_map'    : val_map,
            'val_roc_auc': val_roc,
        })

        # Save best checkpoint
        checkpoint_path = f'models/omlan_best_bird_mae_base_ep{epoch}.pth')
        if val_map > best_map:
            best_map = val_map
            torch.save({
                'epoch'       : epoch,
                'model'       : model.state_dict(),
                'best_map'    : best_map,
                'val_roc_auc' : val_roc,
                'num_classes' : num_classes,
                'label_to_idx': label_to_idx,
                'config'      : {
                    'model_id'   : MODEL_ID,
                    'embed_dim'  : EMBED_DIM,
                    'focal_gamma': FOCAL_GAMMA,
                }
            }, checkpoint_path)
            artifact = wandb.Artifact(name=f"omlan_bird_mae_base_ep{epoch}", type="model", metadata={"epoch": epoch})
            artifact.add_file(local_path=checkpoint_path)
            run.log_artifact(artifact)
            print(f'  ✓ New best mAP={best_map:.4f} — checkpoint saved')

    print(f'\nDone. Best mAP: {best_map:.4f}')
    wandb.finish()

    with open('models/omlan_bird_mae_base_results.json', 'w') as f:
        json.dump({'best_map': best_map, 'model': MODEL_ID,
                   'num_classes': num_classes}, f, indent=2)


if __name__ == '__main__':
    main()

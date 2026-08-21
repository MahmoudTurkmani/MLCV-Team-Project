# MLCV Team Project — BirdCLEF+ 2026

Bird (and amphibian, reptile, mammal, insect) sound classification for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition, developed as part of the MLCV team project at TU Dresden.

## Task Description

The BirdCLEF+ 2026 challenge asks participants to develop machine learning frameworks capable of identifying understudied species within continuous audio data from Brazil's Pantanal wetlands. Species span five taxonomic classes — **Aves, Amphibia, Reptilia, Mammalia, Insecta** — for a total of **234 target species**.

We explore three independent modeling approaches to this problem:

1. **CNN** — EfficientNet backbone (B0 / B3) fine-tuned on mel-spectrograms
2. **AST** — Audio Spectrogram Transformer with a frozen backbone + classification head
3. **Bird-MAE** — Domain-specific Masked Autoencoder pretrained on bird audio, fine-tuned end-to-end

## Members & Task Allocation

| Team Member | Task |
|---|---|
| Mahmoud Trkumani | CNN |
| Minjun Kim | AST |
| Marcel Riehme | Bird-MAE |

## Data

- **35,548** total training audio entries (`.ogg`, 3 seconds – 9 minutes long)
- Heavily class-imbalanced: **97.9%** of clips are Aves, with Amphibia (1.3%), Insecta, Mammalia, and Reptilia making up the rest
- Each clip contains at least one labeled species, and an optional quality rating from 1 (low) to 5 (high) — about a third of clips have no rating
- Many classes have fewer than 30 entries, and some classes appear *only* in unlabeled soundscape recordings, never in isolation — this made both training and evaluation harder for rare species

### Challenges

- Severe class imbalance across taxa and species
- Lack of training data for many rare classes
- No "clean" (single-species) audio available for some classes at all

### Scoring

The competition is scored primarily on:
- **ROC-AUC** — recall vs. (1 − specificity), how well positives/negatives are separated
- **mAP** — precision vs. recall, focused on correctly identifying positives

## Repository Structure

```
MLCV-Team-Project/
├── CNN/                    # EfficientNet B0/B3 training pipeline
├── MAE/                    # Bird-MAE fine-tuning pipeline
├── train_preproc_old.csv   # Preprocessed training manifest
└── README.md
```

*(AST training code is documented in the accompanying project report; see the `AST` section below for its architecture and results.)*

## Methodology

### 1. CNN (EfficientNet B0 / B3)

Audio is converted to a mel-spectrogram and passed through a convolutional backbone:

```
MelSpectrogram → (Modified) Conv2D [1 → C] → EfficientNet (B0/B3) Backbone → Dropout (p=0.4) → Linear (→ 234)
```

Both **EfficientNet B0** (~5M params) and **B3** (~12M params) were fine-tuned. B3 consistently scores higher, but B0 was kept because it fits within the competition's inference time budget — B3 alone is too slow to submit.

**Training setup:** Focal Loss (γ=1), max 30 epochs, patience 12, 5-second clips. The data split guarantees every class appears at least once in training, though not necessarily in validation.

**Augmentation versions tested:**

| Version | Pink Noise | Spec Augment | ESC-50 | Mixcut | Pitch Shift |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline | | | | | |
| V1 | ✅ | ✅ | | ✅ | |
| V2 | | ✅ | ✅ | ✅ | |
| V3 | ✅ | ✅ | ✅ | ✅ | |
| V4 | | ✅ | | | ✅ |
| V5 | | ✅ | | ✅ | |
| V6 | ✅ | | ✅ | ✅ |

**Competition results (Kaggle leaderboard):**

| Version | Model | Private Score | Public Score |
|---|---|---|---|
| Baseline | B3 | 77.56% | 80.32% |
| Baseline | B0 | 77.41% | 79.05% |
| V1 | B3 | 82.13% | 82.60% |
| V1 | B0 | 81.22% | 80.75% |
| V2 | B3 | 80.76% | 80.17% |
| V2 | B0 | 78.02% | 80.77% |
| V3 | B3 | 80.24% | 83.03% |
| V3 | B0 | 80.39% | 82.15% |
| V4 | B3 | 80.55% | 81.36% |
| V4 | B0 | 79.96% | 79.98% |
| V5 | B3 | 82.14% | 82.37% |
| V5 | B0 | 81.11% | 81.35% |
| V6 | B3 | 80.76% | 80.43% |
| V6 | B0 | 77.70% | 79.88% |

> B3 generally outperforms B0 by 1–2% on average, and both models lose roughly 9–10% of their local validation score once evaluated on the hidden leaderboard set, most likely due to sparsely-represented classes.

### 2. AST (Audio Spectrogram Transformer)

Uses [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), pretrained on AudioSet (2M clips), as a **frozen** feature extractor (768-dim embeddings), with a lightweight classification head trained on top:

```
5s clip → AST Feature Extractor → Log-Mel Spectrogram → Frozen AST Encoder → Cached Embedding → Head → 234 classes
```

Two heads were compared:
- **Linear head:** `Linear(768→512) → ReLU → Dropout(0.5) → Linear(512→234)`
- **MLP head:** same shape, consistently outperformed the plain linear head on both ROC-AUC and mAP

**Training setup:** BCEWithLogitsLoss vs. Focal Loss (BCE performed better here), AdamW, LR 1e-3, max 30 epochs, patience 8.

**Augmentation strategy:** rather than augmenting waveforms directly, augmented *embeddings* (Pink Noise, ESC-50, SpecAugment) are pre-computed and concatenated with clean embeddings into an expanded training set; MixUp is additionally applied at the batch/embedding level.

**Results:** the baseline (no augmentation) reached the highest validation ROC-AUC (~0.915), while embedding-level MixUp gave the best mAP gains, with ESC-50 + MixUp peaking at ~0.468 mAP. No single augmentation improved both metrics simultaneously.

### 3. Bird-MAE

[Bird-MAE](https://arxiv.org/abs/2504.12880) is a masked autoencoder pretrained *exclusively* on bird audio (BirdSet, XCL — 1.7M clips), as opposed to AST's general-purpose AudioSet pretraining. This specialization targets the low inter-class / high intra-class variation typical of closely related bird species.

```
Raw audio → Mel spectrogram → 16×16 non-overlapping patches → ViT Encoder → Classification Head → 234 classes
```

- **HuggingFace model IDs:** `DBD-research-group/Bird-MAE-{Base,Large,Huge}` (`trust_remote_code=True` required)
- **Model sizes:** Base (~86M, ViT-B/16), **Large (~307M, ViT-L/16 — reported "sweet spot")**, Huge (~632M, ViT-H/16)
- Bird-MAE-Large was used for the main experiments (embed_dim=1024, hidden_dim=512, ~645K trainable head parameters)

**Two-phase training strategy** (to avoid destroying pretrained representations):

| Phase | Epochs | Encoder | LR | Purpose |
|---|---|---|---|---|
| 1 — Head warmup | 1–5 | Frozen | 1e-3 | Let the head adapt before touching the encoder |
| 2 — Full fine-tuning | 6–30 | Unfrozen | 1e-5 | End-to-end fine-tuning for the target taxa |

Loss: Focal Loss (γ=1), Optimizer: AdamW, Batch size: 64.

Both a **Linear** head (`LayerNorm → Dropout(0.3) → Linear`) and an **MLP** head (`Linear → ReLU → Dropout(0.3) → Linear`) were implemented for comparison across model sizes.

## Results Summary

Full ROC-AUC / mAP curves, per-version breakdowns, and the CNN vs. AST vs. Bird-MAE comparison are in the [project report](https://github.com/MahmoudTurkmani/MLCV-Team-Project) slides. Headline takeaways:

- **CNN (B3)** achieved the best competition leaderboard scores overall, peaking around **83% public / 82% private** with the V3 augmentation stack (Pink Noise + SpecAugment + ESC-50 + Mixcut).
- **AST** with an MLP head and embedding-level MixUp gave the best validation mAP, though augmentation did not universally improve ROC-AUC.
- **Bird-MAE-Large**, fine-tuned end-to-end, is expected to benefit most from its bird-specific pretraining on fine-grained species discrimination; see the report for final numbers across Base/Large/Huge × Linear/MLP.

## References

- Competition: https://www.kaggle.com/competitions/birdclef-2026
- Bird-MAE paper: https://arxiv.org/abs/2504.12880
- ViT paper: https://arxiv.org/abs/2010.11929
- MAE paper: https://arxiv.org/abs/2111.06377
- AST model: [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

# MLCV Team Project - BirdClef 2026

## Task Description
This project is concerned with the annual BirdCLEF+ Challenge 2026 that aims at developing
machine learning frameworks capable of identifying understudied species within continuous audio
data from Brazil’s Pantanal wetlands. This includes the identification of individual bird, amphibian,
reptile, mammal, and insect species. 

## Members & Task Allocation
|Team Member      |Task | 
|    -            |  -  |
|Mahmoud Trkumani | CNN |
|Minjun Kim       | {Replace} |
|Marcel Reihme    | Bird-MAE |

## Data
We begin by having a look at the data from the competition first and understanding it.

### Classes
There are a total of 234 classes in the data that are split into 5 different groups:
- Aves
- Reptilia
- Amphibia
- Insecta
- Mammalia

### Training Audio
The compeition provides a folder that has training audio a folder for all species and then a number of audio recordings (with the least being one recording) containing recordings of that species and possibly others.

### Training manifest
The file `train.csv` contains a list of all recordings and their locations, name, primary and secondary labels, and a rating which is a rating of how reliable the recording is.

## Work
This section describes the work that has been done on each of the tasks and some information regarding the problem itself.

### CNN

To classify the data using a CNN, the audio was first loaded in, transformed into a MelSpectrogram, and then passed to a CNN that uses `EfficientNetB3` or `EfficientNetB0` as a backbone.
The reason for traning both the `B0` and the `B3` model is that the `B3` model yields better results but is too large for the competition. `B0` yields slightly worse results but can be submitted to the competition as it is abides to the time limit constraint. 
Note: `roc-auc` was chosen as the evaluation metric for the models as it is what the competition uses.

#### B3 Results
|Augmentation      |Epoch  | Local Score |
|---               |---    |---          |
|Vanilla           | 24    | 0.709       |
|SpecAugment       | 16    | 0.639       |
|SpecAug + Pink    | 24    | 0.7208      |

#### B0 Results 
|Augmentation      |Epochs | Private Score    | Public Score     |
|---               |---    |---               |---               |
|Vanilla           | 16    | 0.71118          | 0.72376          |
|SpecAugment       | 18    | 0.72930          | 0.70461          |
|SpecAug + Pink    | 1     | 0.66607          | 0.65888          |

### AST
TODO

### Bird-MAE

Bird-MAE (Rauch et al., 2025) is a masked autoencoder pretrained _exclusively_ on bird audio recordings. During pretraining it learned to reconstruct randomly masked patches of spectrograms — this forces the encoder to build detailed representations of bird vocalizations.

- **HuggingFace model IDs:** `DBD-research-group/Bird-MAE-{Base,Large,Huge}`
- **Pretraining data:** Bird audio recordings only (masked autoencoder objective)  
- **Parameters:** 86M (Base), 0.3B (Large), 0.6B (Huge)
- **Requires:** `trust_remote_code=True`

### Architecture (data flow)

```
Input: Raw waveform (no spectrogram — feature extractor handles it internally)
  │
  ↓ BirdMAEFeatureExtractor (internal)
       Takes raw waveform → produces (B, 1, 512, 128) internally
       NOTE: Do NOT pass sampling_rate argument — it is not accepted
  │
  ↓ Transformer Encoder layers (hidden dim = 768)
  │
  ↓ Already-pooled output: (B, 768)
       NOTE: output is last_hidden_state, already pooled — no manual pooling needed
  │
  ↓ Classification head (768 → 234)

Output: (B, 234) raw logits
```

### Classification Heads

#### Linear
```python
self.classifier = nn.Sequential(
    nn.LayerNorm(768),
    nn.Dropout(0.1),
    nn.Linear(768, 234)
)
```

#### MLP
```python
self.classifier = nn.Sequential(
	nn.Linear(768, 512), # hidden_dim 512
	nn.ReLU(inplace=True),
	nn.Dropout(p=0.3),
	nn.Linear(512, 234),
)
```

### Training Setup — Two Phases

Bird-MAE required a two-phase training strategy to avoid destroying pretrained representations:

**Phase 1 — Head Warmup (epochs 1–5)**

```
Encoder:  FROZEN (weights not updated)
Head:     TRAINED only
LR:       1e-3
Purpose:  Let the head adapt to our label space before touching the encoder
```

**Phase 2 — Full Fine-Tuning (epochs 6–30)**

```
Encoder:  UNFROZEN (all weights updated)
Head:     TRAINED
LR:       1e-5  (very low — avoid destroying pretrained weights)
Purpose:  Fine-tune everything end-to-end for our specific taxa
```

| Setting       | Value                                       |
| ------------- | ------------------------------------------- |
| Loss function | Focal Loss (γ=1)                            |
| Optimizer     | AdamW                                       |
| Epochs        | 30 (5 frozen + 25 full fine-tune)           |
| Batch size    | 64                                          |

### Results (ROC-AUC)

| Model | Linear | MLP |
| --- | --- | --- |
| Base | 0.0 | 0.0 |
| Large | 0.0 | 0.0 |
| Huge| 0.0 | 0.0 |

#### With Augmentations
TODO

---

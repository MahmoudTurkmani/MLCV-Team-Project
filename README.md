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
|Marcel Reihme    | {Replace} |

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

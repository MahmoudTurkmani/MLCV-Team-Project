#!/bin/bash

module load release/2026 GCCcore/14.3.0 Python/3.13.5 FFmpeg/7.1.2

cd /home/mari880e/MLCV-Team-Project/MAE

source ./venv/bin/activate

python train_birdmae.py "$@"

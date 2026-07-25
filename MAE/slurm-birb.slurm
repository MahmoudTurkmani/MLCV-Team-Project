#!/bin/bash

#SBATCH --job-name=birdmae
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=marcel.riehme@mailbox.tu-dresden.de
#SBATCH --time=70:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

#module load release/24.04 GCCcore/12.3.0 bzip2/1.0.8
#module load CUDA/12.1.1
#module load Python/3.11.3

module load release/2026 GCCcore/14.3.0 CUDA/13.0.0 Python/3.13.5 FFmpeg/7.1.2

source ./venv/bin/activate

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODENAME"
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"

WANDB_CACHE_DIR="/data/horse/ws/mari880e-mlcv-birdmae/cache/"
python train_birdmae.py "$@"

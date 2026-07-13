#!/bin/bash

#SBATCH --job-name=birdmae
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mari880e@mailbox.tu-dresden.de
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

module load release/24.04 GCCcore/12.3.0 bzip2/1.0.8
module load CUDA/12.1.1
module load Python/3.11.3

source ./venv/bin/activate

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODENAME"
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"

python train_birdmae.py "$@"

#!/bin/bash

# ==========================================
# Slurm Resource Allocation Settings
# ==========================================
#SBATCH --job-name=birdmae       # A descriptive name for your job
#SBATCH --output=logs/train_%j.out         # Standard output log (%j inserts the JobID)
#SBATCH --error=logs/train_%j.err          # Standard error log
#SBATCH --mail-type=END,FAIL               # Notifications for job completion or failure
#SBATCH --mail-user=mari880e@mailbox.tu-dresden.de # Your email address

# ==========================================
# Cluster Rules & Resource Management
# ==========================================
#SBATCH --time=08:00:00                     # Walltime limit (HH:MM:SS) - Adjust to your needs
#SBATCH --nodes=1                          # Number of nodes requested
#SBATCH --ntasks=1                         # Run a single task
#SBATCH --cpus-per-task=4                  # Number of CPU cores allocated for data loading/processing
#SBATCH --gres=gpu:1                       # Request 1 GPU (e.g., gpu:a100:1 if specific type is needed)

# ==========================================
# Environment Setup & Execution
# ==========================================

# 1. Clear any inherited modules for a clean slate
# module purge

# 2. Load your two required modules (e.g., CUDA and Python/Anaconda)
#    Replace these with the exact module names on your cluster
module load release/24.04 GCCcore/12.3.0 bzip2/1.0.8
module load CUDA/12.1.1
module load Python/3.11.3

# 3. Activate your Python virtual environment
#    Replace 'path/to/your/venv' with the actual path to your venv directory
source ./venv/bin/activate

# 4. Print some diagnostic info to the log file (highly recommended)
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODENAME"
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"

# 5. Run your Python script
#    Using 'python -u' ensures the output logs flush in real-time
python train_birdmae.py $@

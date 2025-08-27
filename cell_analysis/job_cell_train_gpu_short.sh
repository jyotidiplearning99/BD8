#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall          # <-- single-GPU friendly
#SBATCH --time=08:00:00               # short jobs backfill sooner
#SBATCH --job-name=cell-train-short
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@300

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # smaller footprint fits earlier
#SBATCH --gres=gpu:a100:1             # exactly 1 GPU

set -Eeuo pipefail
module purge
module load pytorch                   # GPU-enabled PyTorch on Mahti
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "==== Node & GPU check ===="; hostname
nvidia-smi || true
python - <<'PY' || true
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
echo "=========================="

# If using PyTorch Lightning:
TRAIN_CMD="python -u main.py --mode train --accelerator gpu --devices 1"

# Run
srun -u $TRAIN_CMD

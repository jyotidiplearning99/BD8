#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --time=08:00:00
#SBATCH --job-name=cell-train-short
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1

set -Eeuo pipefail
mkdir -p logs checkpoints

module purge
module load pytorch
source .venv-gpu/bin/activate          # <-- has --system-site-packages, sees module torch
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "==== Node & GPU check ===="; hostname
nvidia-smi || true
python - <<'PY' || true
import sys, torch, numpy
print("python =", sys.version.split()[0])
print("torch  =", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "CUDA:", getattr(torch.version, "cuda", None))
print("numpy  =", numpy.__version__)
PY
echo "=========================="

test -f main.py || { echo "main.py not found in \$PWD"; exit 2; }

( while true; do
    date "+%F %T"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
    sleep 30
  done ) >> logs/gpu-$SLURM_JOB_ID.log 2>&1 &

TRAIN_CMD="python -u main.py --mode train --accelerator gpu --devices 1"
srun -u $TRAIN_CMD

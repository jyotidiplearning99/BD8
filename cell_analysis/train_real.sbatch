#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -J bdS8_train
#SBATCH -o logs/%x-%j.out
#SBATCH --signal=B:TERM@300
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis

set -euo pipefail

module --force purge
module load pytorch/2.7

# Optional perf tweaks
python - <<'PY'
import torch
try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
PY

echo "Job ${SLURM_JOB_ID:-no_jobid} starting: $(date)"
python -u main.py --mode train
echo "Done: $(date)"

#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J bdS8_short
#SBATCH -o logs/%x-%j.out
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis

set -euo pipefail
module --force purge
module load pytorch/2.7
mkdir -p logs models lightning_logs

python - <<'PY'
import torch, os
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
print("CPUs per task:", os.environ.get("SLURM_CPUS_PER_TASK"))
PY

echo "Job ${SLURM_JOB_ID:-no_jobid} starting: $(date)"
python -u main.py --mode train
echo "Done: $(date)"

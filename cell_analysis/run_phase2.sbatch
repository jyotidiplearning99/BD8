#!/bin/bash
#SBATCH --job-name=phase2
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1          # <-- use GRES on this cluster
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis

set -euo pipefail
mkdir -p logs outputs/phase2

module purge || true
module load python-data/3.10-24.04
source ~/.venvs/amlphase2/bin/activate

# ensure src is a package and PYTHONPATH is safe
test -f src/__init__.py || touch src/__init__.py
export PROJECT_ROOT=/scratch/project_2010376/JDs_Project/cell_analysis
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# keep caches on scratch
export SCR_BASE=/scratch/project_2010376/homecache/$USER
mkdir -p "$SCR_BASE"/{pip-cache,tmp,torch-cache}
export PIP_CACHE_DIR="$SCR_BASE/pip-cache"
export TMPDIR="$SCR_BASE/tmp"
export TORCH_HOME="$SCR_BASE/torch-cache"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "===== GPU & Torch check ====="
nvidia-smi || true
python - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
echo "============================="

# run as a module so "import src.*" works
python -u -m src.phase2_fcs_gating

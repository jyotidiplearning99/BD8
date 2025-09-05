#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --job-name=flow2cp-v6
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1

set -Eeuo pipefail
set -x

# Load modules
module purge || true
module load pytorch

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories
mkdir -p logs .cache src outputs/flow2cp_analysis
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TORCH_HOME="$PWD/.cache/torch"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Virtual environment
PYENV="$PWD/.venv_flow2cp_fcs"
source "$PYENV/bin/activate"

echo "[`date`] Running Clinical Cell Painting Analysis V6"

# Run with CORRECT script name
srun -u python -u src/phase2_complete_v3.py \
    --image-dir /scratch/project_2010376/BDS8/BDS8_data /scratch/project_2010751/BDS8/BDS8_data \
    --fcs-dir /scratch/project_2010376/BDS8/BDS8_data /scratch/project_2010751/BDS8/BDS8_data \
    --output-dir outputs/flow2cp_analysis \
    --n-clusters 10 \
    --n-examples 20 \
    --batch-size 4 \
    --max-samples 5000 \
    --save-models \
    --load-existing-models \
    --analyze-fcs

RC=$?
echo "[`date`] Pipeline finished with exit code ${RC}"
exit ${RC}

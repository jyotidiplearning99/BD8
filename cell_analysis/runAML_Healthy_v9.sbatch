#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --job-name=flow2cpv9-integrated
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1

set -Eeuo pipefail
set -x

#-------------------------------
# Modules
#-------------------------------
module --force purge || true
module load pytorch

#-------------------------------
# Directories & caches
#-------------------------------
mkdir -p logs .cache src outputs/integrated_analysis
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TORCH_HOME="$PWD/.cache/torch"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR"

#-------------------------------
# Core env flags
#-------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

#-------------------------------
# Virtual environment (use system PyTorch via --system-site-packages)
#-------------------------------
PYENV="$PWD/.venv_flow2cp_fcs"
python -m venv --system-site-packages "$PYENV"
# shellcheck source=/dev/null
source "$PYENV/bin/activate"

# Ensure venv pip is used
python -m pip install --upgrade pip setuptools wheel
# Use headless OpenCV for HPC
python -m pip install \
  numpy pandas scikit-learn scipy matplotlib seaborn tifffile tqdm joblib \
  opencv-python-headless \
  fcsparser umap-learn hdbscan
# (Intentionally not installing torch; provided by module)

#-------------------------------
# Diagnostics (optional)
#-------------------------------
python -V
pip -V
nvidia-smi || true

echo "[$(date)] Running Integrated Clinical Cell Painting Analysis"

# Graceful shutdown on TERM (300s preemption notice)
trap 'echo "[$(date)] Caught SIGTERM, finishing up..."; sleep 5' TERM

#-------------------------------
# Run the integrated analysis
# NOTE: This call assumes your script accepts these CLI flags.
# If your actual script is the ClinicalPipeline v9.0 (no CLI),
# replace the srun line with: srun -u python -u src/ClinicalPipeline_v9.py
#-------------------------------
srun -u python -u src/AML_Healthy_v9.0_final.py \
  --image-dir /scratch/project_2010376/BDS8/BDS8_data /scratch/project_2010751/BDS8/BDS8_data \
  --fcs-dir   /scratch/project_2010376/BDS8/BDS8_data /scratch/project_2010751/BDS8/BDS8_data \
  --phase1-model models/last.ckpt \
  --output-dir outputs/integrated_analysis \
  --n-clusters 10 \
  --n-examples 20 \
  --batch-size 4 \
  --max-samples 5000 \
  --save-models \
  --analyze-fcs

RC=$?
echo "[$(date)] Pipeline finished with exit code ${RC}"
exit ${RC}

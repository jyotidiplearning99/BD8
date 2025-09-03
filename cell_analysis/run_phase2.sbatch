#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --time=08:00:00
#SBATCH --job-name=cell-train
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1

set -Eeuo pipefail

module purge || true
module load pytorch           # brings a CUDA-enabled torch + Python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

mkdir -p logs .cache
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"

# --- Python env that inherits the module's site-packages (so torch is visible) ---
PYENV="$PWD/.venv_amlphase2"
rm -rf "$PYENV"
python -m venv --system-site-packages "$PYENV"
source "$PYENV/bin/activate"
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel

# Light deps (NO flowkit). yaml is required by your script.
pip install --no-input pyyaml fcsparser tifffile opencv-python \
                      umap-learn scikit-learn matplotlib scipy pandas

# --- Sanity checks ---
echo "==== Node & GPU check ===="
hostname
nvidia-smi || true
python - <<'PY' || true
import torch, yaml
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("torch:", getattr(torch, "__version__", "?"))
print("pyyaml:", getattr(yaml, "__version__", "?"))
PY
echo "=========================="

echo "[`date`] Starting Phase 2…"
which python; python -V

# Run your phase 2 script
srun -u python -u src/phase2_complete_v3.py \
  --image-dir /scratch/project_2010376/BDS8/BDS8_data \
  --fcs-dir   /scratch/project_2010376/BDS8/BDS8_data \
  --model-path models/last.ckpt \
  --output-dir outputs/phase2 \
  --arcsinh-cofactor 150 \
  --max-cells-per-sample 1000 \
  --batch-size 4 \
  --max-samples 20

RC=$?
echo "[`date`] Phase 2 finished with exit code ${RC}"
exit ${RC}

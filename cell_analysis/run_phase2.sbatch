#!/bin/bash
#SBATCH -J phase2_aml
#SBATCH --partition=gputest            # try: gputest (15 min) → later: gpusmall/gpumedium
#SBATCH --time=00:15:00                # increase for gpusmall/gpumedium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Either of the following GPU requests usually works. Keep ONE and comment the other.
#SBATCH --gres=gpu:a100:1             # Mahti shows GRES=gpu:a100 in sinfo (safe choice)
# #SBATCH --gpus=1                     # Modern Slurm alternative (commented by default)

#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

module purge || true

# --- Activate Python environment ---
# Option A: Use site Miniconda module (preferred on CSC)
module avail 2>/dev/null | grep -i miniconda || true
module load Miniconda3  # adjust to the exact name shown by `module avail`

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amlphase2 || {
  echo "Conda env 'amlphase2' not found. Creating it..."
  conda create -y -n amlphase2 python=3.10
  conda activate amlphase2
  # CUDA-enabled PyTorch (CUDA 12.1 wheels are common on A100)
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  # Your analysis deps
  pip install fcsparser flowkit tifffile opencv-python umap-learn scikit-learn matplotlib scipy pandas
}

# --- Sanity checks ---
echo "===== GPU ====="
nvidia-smi || true
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# --- Run your analysis ---
python -u src/phase2_complete.py

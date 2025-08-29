#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -J bdS8_infer
#SBATCH -o logs/%x-%j.out
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis

set -euo pipefail
module --force purge
module load pytorch/2.7

# make sure output dir exists
mkdir -p logs

echo "Node: $(hostname)  |  CUDA: $(python - <<'PY'
import torch; print(torch.cuda.is_available())
PY
)"
echo "Start: $(date)"

python - <<'PY'
import torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.train import RealTIFFDataset
from src.models import BD_S8_Model

ckpt_path = Path("models/real_tiff_epoch=20_val_acc=0.995.ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = RealTIFFDataset(mode='test', skip_holdout=20000, train_count=30000, val_count=5000, test_count=5000)
dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

model = BD_S8_Model(num_classes=5).to(device).eval()
ck = torch.load(ckpt_path, map_state_dict=False, map_location=device)
state = ck.get("state_dict", ck)
clean = {k[6:]: v for k,v in state.items() if k.startswith("model.")}
missing = model.load_state_dict(clean, strict=False)
print("Loaded:", ckpt_path.name, "| missing/unexpected:", missing)

all_p, all_y = [], []
with torch.no_grad():
    for b in dl:
        x = b["image"].to(device, non_blocking=True)
        ys = torch.tensor([b["labels"]["extraction_method"][i] for i in range(len(b["labels"]["extraction_method"]))], device=device)
        out = model(x)["extraction"]
        p = out.argmax(1)
        all_p.append(p.cpu().numpy())
        all_y.append(ys.cpu().numpy())

all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
acc = (all_p == all_y).mean()
tp = ((all_p==1)&(all_y==1)).sum(); tn = ((all_p==0)&(all_y==0)).sum()
fp = ((all_p==1)&(all_y==0)).sum(); fn = ((all_p==0)&(all_y==1)).sum()
print(f"\n🎯 TEST (disjoint) — N={len(all_y)}")
print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
PY

echo "End: $(date)"

# src/phase2_complete.py
"""
Phase 2: Complete morpho-phenotypic analysis for AML vs Healthy
- No TIFF↔FCS per-cell mapping required
- Builds per-sample morphology feature caches
- Self-tuning blast gate from FCS (percentiles; CD45dim/SSClow fallback)
- Attention-based MIL (sample-level AML vs Healthy) + blast% regression
- Summary visualizations + attention inspection
"""

from __future__ import annotations

import os
import re
import cv2
import csv
import json
import yaml
import hashlib
import random
import warnings
import contextlib
import gc
from math import sqrt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile

from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance

# Optional UMAP
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

# Plotting (headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============ DETERMINISM ============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# FCS libs
try:
    import flowkit as fk
    HAVE_FLOWKIT = True
except Exception:
    HAVE_FLOWKIT = False

try:
    import fcsparser
    HAVE_FCSPARSER = True
except Exception:
    HAVE_FCSPARSER = False

if not HAVE_FLOWKIT and not HAVE_FCSPARSER:
    raise ImportError("Install either 'flowkit' or 'fcsparser' to read FCS files.")


# =========================
# FCS PROCESSOR WITH DYNAMIC MAPPING
# =========================
class FCSProcessor:
    """Process FCS with header-driven mapping and self-tuning blast gates."""
    def __init__(self, cofactor: float = 150, log_dir: Path = Path("outputs/phase2/logs")):
        self.cofactor = float(cofactor)
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_fcs = None

        self.marker_mappings = {
            'CD34' : ['CD34-BV421','CD34-V450','CD34-PerCP','CD34 BV785','CD34 BV786','CD34'],
            'CD117': ['CD117-PE','CD117-APC','cKit','c-Kit','KIT','CD117'],
            'CD38' : ['CD38-FITC','CD38-PE-Cy7','CD38'],
            'HLA-DR':['HLA-DR-APC-Cy7','HLA-DR-V500','HLADR','HLA-DR BV605','HLA-DR'],
            'CD45' : ['CD45-KO','CD45-V500','CD45 V500','CD45 BV510','CD45'],
            'CD13' : ['CD13-PE','CD13-FITC','CD13'],
            'CD33' : ['CD33-PE-Cy7','CD33-APC','CD33'],
            'CD14' : ['CD14-FITC','CD14-PerCP','CD14 BV650','CD14'],
            'CD15' : ['CD15-APC','CD15-FITC','CD15'],
            'CD16' : ['CD16-PE','CD16-APC-Cy7','CD16'],
            'CD11b': ['CD11b-PE','CD11b-APC','CD11b'],
            'CD56' : ['CD56-PE','CD56-APC','CD56'],
            'CD7'  : ['CD7-PE','CD7-APC','CD7'],
            'MPO'  : ['MPO PE','MPO PE-A','MPO'],
        }

    def load_fcs(self, fcs_path: Path) -> pd.DataFrame:
        """Read FCS and apply arcsinh to numeric channels (robust for both libs)."""
        fcs_path = Path(fcs_path)
        self.last_fcs = fcs_path.stem
        print(f"  📄 FCS file: {fcs_path}")

        meta = None
        df = None

        if HAVE_FLOWKIT:
            try:
                sample = fk.Sample(str(fcs_path))
                if getattr(sample, 'compensation', None) is not None:
                    sample.apply_compensation(sample.compensation)
                # Always pull raw and transform ourselves (avoids FlowKit transform pitfalls)
                df = sample.as_dataframe()
                num = df.select_dtypes(include=[np.number]).columns
                df[num] = np.arcsinh(df[num] / self.cofactor)
                print(f"  ↳ Using FlowKit; arcsinh(cofactor={self.cofactor}) applied to numeric channels")
            except Exception as e:
                print(f"  ⚠️ FlowKit read failed ({e}); falling back to fcsparser...")

        if df is None and HAVE_FCSPARSER:
            meta, data = fcsparser.parse(str(fcs_path))
            df = pd.DataFrame(data)
            num = df.select_dtypes(include=[np.number]).columns
            df[num] = np.arcsinh(df[num] / self.cofactor)
            print(f"  ↳ Using fcsparser; arcsinh(cofactor={self.cofactor}) applied to numeric channels")

        if df is None:
            raise RuntimeError(f"Failed to read FCS: {fcs_path.name}")

        # Standardize columns
        df, mapping = self._standardize_columns(df, meta)
        self._log_marker_mapping(df, fcs_path, mapping)
        return df

    def _standardize_columns(self, df: pd.DataFrame, meta: Optional[dict] = None) -> Tuple[pd.DataFrame, Dict]:
        new_columns, mapping = {}, {}
        search_texts = {}
        for i, col in enumerate(df.columns, start=1):
            pieces = [str(col)]
            if meta is not None:
                pnn = meta.get(f"$P{i}N"); pns = meta.get(f"$P{i}S")
                if pnn: pieces.append(str(pnn))
                if pns: pieces.append(str(pns))
            search_texts[col] = " ".join(pieces).upper()

        for col in df.columns:
            text = search_texts[col]
            matched = False
            for marker, variants in self.marker_mappings.items():
                for variant in variants + [marker]:
                    if variant.upper() in text:
                        new_columns[col] = marker
                        mapping[col] = marker
                        matched = True
                        break
                if matched: break
            if not matched and any(x in text for x in ['FSC','SSC','TIME','LIGHTLOSS','ZOMBIE']):
                new_columns[col] = col

        df = df.rename(columns=new_columns)
        df = df.loc[:, ~df.columns.duplicated()]
        return df, mapping

    def gate_blast_population_with_tuning(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Define blast gate (percentiles) with auto-tuning and CD45dim/SSClow fallback."""
        n = len(df)
        if n == 0:
            return np.zeros(0, dtype=bool), {"_empty": True}

        gate, thr = self._compute_blast_gate(df, {"CD34":75,"CD117":70,"CD38_low":20,"CD38_high":60,"HLA-DR":60})
        frac = gate.mean(); method = "primary"

        if frac < 0.005 or frac > 0.995:
            method = "auto_tuned"
            gate, thr = self._auto_tune_blast(df)
            frac = gate.mean()

        if frac < 0.005 or frac > 0.995:
            method = "fallback_cd45_ssc"
            gate, thr = self._fallback_cd45_ssc(df)
            frac = gate.mean()

        thr["method"] = method
        print(f"✓ Blast population: {int(gate.sum())}/{n} cells ({100*frac:.1f}%) [{method}]")
        self._save_thresholds(thr, gate, n)
        return gate, thr

    def _compute_blast_gate(self, df: pd.DataFrame, q: Dict) -> Tuple[np.ndarray, Dict]:
        n = len(df); gate = np.ones(n, dtype=bool); thr = {}
        if "CD34" in df.columns:
            t = float(np.nanpercentile(pd.to_numeric(df["CD34"], errors='coerce').dropna(), q["CD34"]))
            thr["CD34_thr"] = t; gate &= pd.to_numeric(df["CD34"], errors='coerce').fillna(-1e9) > t
        if "CD117" in df.columns:
            t = float(np.nanpercentile(pd.to_numeric(df["CD117"], errors='coerce').dropna(), q["CD117"]))
            thr["CD117_thr"] = t; gate &= pd.to_numeric(df["CD117"], errors='coerce').fillna(-1e9) > t
        if "CD38" in df.columns:
            v = pd.to_numeric(df["CD38"], errors='coerce').dropna()
            lo = float(np.nanpercentile(v, q["CD38_low"])); hi = float(np.nanpercentile(v, q["CD38_high"]))
            thr["CD38_low"] = lo; thr["CD38_high"] = hi
            x = pd.to_numeric(df["CD38"], errors='coerce').values
            gate &= (np.nan_to_num(x, nan=-1e9) > lo) & (np.nan_to_num(x, nan=1e9) < hi)
        if "HLA-DR" in df.columns:
            t = float(np.nanpercentile(pd.to_numeric(df["HLA-DR"], errors='coerce').dropna(), q["HLA-DR"]))
            thr["HLA-DR_thr"] = t; gate &= pd.to_numeric(df["HLA-DR"], errors='coerce').fillna(-1e9) > t
        return gate, thr

    def _auto_tune_blast(self, df: pd.DataFrame, max_iter: int = 10) -> Tuple[np.ndarray, Dict]:
        target_min, target_max = 0.005, 0.40
        q = {"CD34":60, "CD117":60, "CD38_low":15, "CD38_high":70, "HLA-DR":50}
        gate, thr = self._compute_blast_gate(df, q)
        for _ in range(max_iter):
            frac = gate.mean()
            if target_min <= frac <= target_max:
                return gate, thr
            if frac < target_min:
                q["CD34"]=max(40,q["CD34"]-5); q["CD117"]=max(40,q["CD117"]-5); q["HLA-DR"]=max(30,q["HLA-DR"]-5)
            else:
                q["CD34"]=min(90,q["CD34"]+5); q["CD117"]=min(90,q["CD117"]+5); q["HLA-DR"]=min(80,q["HLA-DR"]+5)
            gate, thr = self._compute_blast_gate(df, q)
        return gate, thr

    def _fallback_cd45_ssc(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        n = len(df); gate = np.ones(n, dtype=bool); thr = {}
        if "CD45" in df.columns:
            v = pd.to_numeric(df["CD45"], errors='coerce').values
            t = float(np.nanpercentile(v[np.isfinite(v)], 35)) if np.isfinite(v).any() else np.nan
            thr["CD45_dim"] = t; gate &= np.nan_to_num(v, nan=1e9) < t
        ssc_cols = [c for c in df.columns if "SSC" in str(c).upper()]
        if ssc_cols:
            v = pd.to_numeric(df[ssc_cols[0]], errors='coerce').values
            t = float(np.nanpercentile(v[np.isfinite(v)], 40)) if np.isfinite(v).any() else np.nan
            thr["SSC_low"] = t; gate &= np.nan_to_num(v, nan=1e9) < t
        return gate, thr

    def _save_thresholds(self, thr: Dict, gate: np.ndarray, n: int):
        out = {
            "file": self.last_fcs, "timestamp": datetime.now().isoformat(),
            "thresholds": thr, "gate_count": int(gate.sum()),
            "total_cells": int(n), "gate_percentage": float(gate.mean()*100.0)
        }
        p = self.log_dir / f"{self.last_fcs}_blast_thresholds.json"
        with open(p, "w") as f: json.dump(out, f, indent=2)

    def _log_marker_mapping(self, df: pd.DataFrame, fcs_path: Path, mapping: Dict):
        log = {
            "file": str(fcs_path), "timestamp": datetime.now().isoformat(),
            "mapping": mapping, "columns_final": list(df.columns),
            "cofactor": self.cofactor
        }
        p = self.log_dir / f"{fcs_path.stem}_marker_map.json"
        with open(p, "w") as f: json.dump(log, f, indent=2)


# =========================
# MEMORY-EFFICIENT MORPHOLOGICAL ANALYZER
# =========================
class MorphoPhenotypicAnalyzer:
    """Extract morphological features with per-image caching (O(1) RAM)."""
    def __init__(self, model_path: str = "models/last.ckpt", cache_dir: Path = Path("outputs/phase2/feat_cache")):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._load_model(model_path); self.model.eval()

    def _load_model(self, model_path: str) -> nn.Module:
        from src.models import BD_S8_Model
        model = BD_S8_Model(num_classes=5).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean = {}
        for k, v in state_dict.items():
            clean[k[6:]] = v if k.startswith('model.') else v
        model.load_state_dict(clean, strict=False)
        self.feature_extractor = model.encoder
        return model

    @torch.no_grad()
    def extract_morphological_features(self, image_paths: List[Path], batch_size: int = 4) -> np.ndarray:
        feats = []
        for pth in image_paths:
            rel = str(Path(pth).resolve()); hid = hashlib.sha1(rel.encode()).hexdigest()[:10]
            cache = self.cache_dir / f"{Path(pth).stem}_{hid}.npy"
            if cache.exists():
                try:
                    feats.append(np.load(cache)); continue
                except Exception:
                    pass
            # Single-image path (micro-batch = 1)
            t = self._load_and_preprocess_image(pth).unsqueeze(0).to(self.device)
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    z = self.feature_extractor(t).float().cpu().numpy()
            else:
                z = self.feature_extractor(t).cpu().numpy()
            np.save(cache, z[0]); feats.append(z[0])
            del t, z
            if self.device.type == 'cuda': torch.cuda.empty_cache()
        return np.vstack(feats) if feats else np.zeros((0, 1), dtype=np.float32)

    def _load_and_preprocess_image(self, path: Path) -> torch.Tensor:
        img = tifffile.imread(str(path))
        if img.ndim == 2: img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[0] <= 4: img = np.transpose(img[:3], (1, 2, 0))
        if img.shape[-1] != 3:
            h, w = img.shape[:2]; tmp = np.zeros((h, w, 3), dtype=np.float32)
            tmp[..., :min(3, img.shape[-1])] = img[..., :min(3, img.shape[-1])]
            img = tmp
        img = cv2.resize(img.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
        mx = float(img.max())
        if mx > 0:
            if mx > 255: img /= 65535.0
            elif mx > 1: img /= 255.0
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()


# =========================
# MIL MODEL (Attention)
# =========================
class AttentionMIL(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, n_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        self.blast_regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, return_attention: bool = False):
        # features: (B, N, D)
        A = self.attention(features)            # (B, N, 1)
        A = torch.softmax(A, dim=1)            # normalize across instances
        z = torch.sum(A * features, dim=1)     # (B, D)
        logits = self.classifier(z)            # (B, 2)
        blast_pred = self.blast_regressor(z)   # (B, 1)
        if return_attention:
            return logits, blast_pred, A.squeeze(-1)
        return logits, blast_pred


# =========================
# Dataset + Collate (pad bags per batch)
# =========================
class SampleLevelDataset(Dataset):
    def __init__(self, sample_features: Dict[str, Dict], max_cells: int = 1000, min_cells_pad: int = 1):
        self.samples = []
        for sid, data in sample_features.items():
            feats = data['features']
            if len(feats) == 0:    # skip empty
                continue
            feats = feats[:max_cells]
            self.samples.append({
                'id': sid,
                'features': feats.astype(np.float32),
                'label': int(data['label']),
                'blast_pct': float(data.get('blast_pct', 0.0))
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.from_numpy(s['features']),        # (N, D)
            'label': torch.tensor(s['label'], dtype=torch.long),
            'blast_pct': torch.tensor(s['blast_pct'], dtype=torch.float32),
            'sid': s['id'],
        }

def pad_collate(batch):
    # Find max instances in this batch
    max_n = max(b['features'].shape[0] for b in batch)
    D = batch[0]['features'].shape[1]
    B = len(batch)
    feats = torch.zeros(B, max_n, D, dtype=torch.float32)
    labels = torch.empty(B, dtype=torch.long)
    blast = torch.empty(B, 1, dtype=torch.float32)
    sids = []
    for i, b in enumerate(batch):
        n = b['features'].shape[0]
        feats[i, :n] = b['features']
        labels[i] = b['label']
        blast[i, 0] = b['blast_pct']
        sids.append(b['sid'])
    return {'features': feats, 'label': labels, 'blast_pct': blast, 'sid': sids}


# =========================
# MAIN PHASE 2 PIPELINE
# =========================
class Phase2Pipeline:
    def __init__(self,
                 image_dir: Path = Path('/scratch/project_2010376/BDS8/BDS8_data'),
                 fcs_dir: Path = Path('/scratch/project_2010376/BDS8/BDS8_data'),
                 model_path: str = 'models/last.ckpt',
                 output_dir: Path = Path('outputs/phase2'),
                 config: Optional[Dict] = None):
        self.image_dir = Path(image_dir)
        self.fcs_dir = Path(fcs_dir)
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.cofactor = float(self.config.get('arcsinh_cofactor', 150.))
        self.max_cells_per_sample = int(self.config.get('max_cells_per_sample', 1000))
        self.batch_size = int(self.config.get('batch_size', 4))
        self.fcs_processor = FCSProcessor(cofactor=self.cofactor, log_dir=self.output_dir / "logs")
        self.morpho_analyzer = MorphoPhenotypicAnalyzer(model_path=model_path, cache_dir=self.output_dir / "feat_cache")
        self.feature_cache: Dict[str, Dict] = {}
        self.mil_model: Optional[AttentionMIL] = None

    def match_samples(self) -> Dict[str, Dict]:
        matches = {}
        fcs_files = list(self.fcs_dir.rglob('*.fcs')) if self.fcs_dir else []
        print(f"\n📁 Found {len(fcs_files)} FCS files")
        for fcs_path in fcs_files:
            sample_id = fcs_path.stem
            patterns = [f"{sample_id}_images_*", f"{sample_id}_*.tif*", f"{sample_id}*.tif*"]
            image_paths = []
            for pat in patterns:
                for m in self.image_dir.rglob(pat):
                    if m.is_dir():
                        image_paths.extend(list(m.rglob("*.tif")))
                        image_paths.extend(list(m.rglob("*.tiff")))
                    elif m.suffix.lower() in {'.tif','.tiff'}:
                        image_paths.append(m)
            if image_paths:
                image_paths = sorted(set(image_paths))
                pstr = str(fcs_path)
                if '/AML/' in pstr or ' AML' in pstr:
                    sample_type = 'AML'
                elif '/Healthy' in pstr or 'Healthy BM' in pstr or 'BM_' in pstr:
                    sample_type = 'Healthy'
                else:
                    sample_type = 'Unknown'
                matches[sample_id] = {"fcs_path": fcs_path, "image_paths": image_paths, "sample_type": sample_type}
                print(f"  ✓ {sample_id}: {len(image_paths)} images, {sample_type}")
            else:
                print(f"  ✗ {sample_id}: no images found")
        return matches

    def build_feature_cache(self, matched_samples: Dict) -> Dict:
        print("\n📦 Building feature cache...")
        rng = np.random.RandomState(SEED)
        for sid, data in matched_samples.items():
            print(f"\nProcessing {sid}...")
            fcs_df = self.fcs_processor.load_fcs(data['fcs_path'])
            
            marker_cols = [c for c in ["CD34","CD117","CD38","HLA-DR","CD13","CD33","CD15","CD16","CD11b","CD56","CD7","MPO","CD45"]if c in fcs_df.columns]
            if marker_cols:
                        # robust median vector (nan-safe)
                phen_vec = fcs_df[marker_cols].median(numeric_only=True).astype(np.float32).values
            else:
                phen_vec = np.empty((0,), dtype=np.float32)

            blast_mask, _ = self.fcs_processor.gate_blast_population_with_tuning(fcs_df)
            blast_pct = float(blast_mask.mean())

            img_paths = data['image_paths']
            if len(img_paths) > self.max_cells_per_sample:
                idx = rng.choice(len(img_paths), size=self.max_cells_per_sample, replace=False)
                img_paths = [img_paths[i] for i in idx]

            feats = self.morpho_analyzer.extract_morphological_features(img_paths, batch_size=1)
            if feats.size == 0:
                print("  ⚠️ No features extracted; skipping sample.")
                continue
            #feats = self.morpho_analyzer.extract_morphological_features(img_paths, batch_size=1)

            # concatenate the sample-level phenotype vector to each cell feature
            if phen_vec.size:
                phen_tile = np.tile(phen_vec, (feats.shape[0], 1))
                feats = np.hstack([feats, phen_tile]).astype(np.float32)    
            self.feature_cache[sid] = {
                'features': feats,
                'label': 1 if data['sample_type'] == 'AML' else 0,
                'blast_pct': blast_pct,
                'sample_type': data['sample_type']
            }
            print(f"  ✓ Cached {len(feats)} cells, blast={blast_pct:.1%}")
        return self.feature_cache

    def train_mil_model(self, epochs: int = 50):
        print("\n🧠 Training MIL model...")
        sids = list(self.feature_cache.keys())
        if len(sids) < 4:
            print("  ⚠️ Not enough samples for MIL training (need ≥4). Skipping.")
            return None

        labels = [self.feature_cache[s]['label'] for s in sids]
        from sklearn.model_selection import train_test_split
        train_ids, val_ids = train_test_split(sids, test_size=0.2, stratify=labels, random_state=SEED)

        train_data = {sid: self.feature_cache[sid] for sid in train_ids}
        val_data = {sid: self.feature_cache[sid] for sid in val_ids}

        train_ds = SampleLevelDataset(train_data, max_cells=self.max_cells_per_sample)
        val_ds = SampleLevelDataset(val_data, max_cells=self.max_cells_per_sample)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=pad_collate)
        val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, collate_fn=pad_collate)

        feat_dim = next(iter(self.feature_cache.values()))['features'].shape[1]
        self.mil_model = AttentionMIL(feature_dim=feat_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        cls_crit = nn.CrossEntropyLoss()
        reg_crit = nn.MSELoss()
        opt = torch.optim.Adam(self.mil_model.parameters(), lr=1e-4)

        device = next(self.mil_model.parameters()).device
        best_val = 0.0

        for epoch in range(epochs):
            self.mil_model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch['features'].to(device)      # (B, N, D)
                y = batch['label'].to(device)         # (B,)
                b = batch['blast_pct'].to(device)     # (B, 1)

                logits, bpred = self.mil_model(x)
                loss = cls_crit(logits, y) + 0.1 * reg_crit(bpred, b)

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            # Val
            self.mil_model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['features'].to(device)
                    y = batch['label'].to(device)
                    logits, _ = self.mil_model(x)
                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            val_acc = (correct / total) if total else 0.0
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1:03d} | loss={epoch_loss/max(1,len(train_loader)):.3f} | val_acc={val_acc:.3f}")

            if val_acc > best_val:
                best_val = val_acc
                torch.save({
                    'model_state_dict': self.mil_model.state_dict(),
                    'feature_dim': feat_dim,
                    'val_acc': best_val,
                    'epoch': epoch+1
                }, self.output_dir / 'mil_model_best.pth')

        print(f"✓ Best validation accuracy: {best_val:.3f}")
        return self.mil_model

    def analyze_attention_patterns(self):
        if not self.mil_model:
            return {}
        print("\n🔍 Analyzing attention patterns...")
        device = next(self.mil_model.parameters()).device
        self.mil_model.eval()
        out = {}
        with torch.no_grad():
            for sid, data in self.feature_cache.items():
                x = torch.from_numpy(data['features'].astype(np.float32)).unsqueeze(0).to(device)
                logits, bpred, A = self.mil_model(x, return_attention=True)
                pred_cls = torch.argmax(logits, dim=1).item()   # FIXED: dim=1
                att = A.squeeze(0).detach().cpu().numpy()
                top_idx = np.argsort(att)[-10:]
                out[sid] = {
                    'prediction': int(pred_cls),
                    'blast_prediction': float(bpred.item()),
                    'actual_blast': float(data['blast_pct']),
                    'top_attention_indices': top_idx.tolist(),
                    'top_attention_weights': att[top_idx].tolist()
                }
        with open(self.output_dir / 'attention_analysis.json', 'w') as f:
            json.dump(out, f, indent=2)
        return out

    def compare_aml_vs_healthy(self, aml_ids: List[str], healthy_ids: List[str]):
        print("\n📊 Comparing AML vs Healthy...")
        aml_blasts = [self.feature_cache[s]['blast_pct'] for s in aml_ids]
        healthy_blasts = [self.feature_cache[s]['blast_pct'] for s in healthy_ids]
        from scipy.stats import mannwhitneyu
        if aml_blasts and healthy_blasts:
            _, p = mannwhitneyu(aml_blasts, healthy_blasts, alternative='greater')
        else:
            p = 1.0
        comp = {
            'blast_burden': {
                'aml_mean': float(np.mean(aml_blasts)) if aml_blasts else 0.0,
                'healthy_mean': float(np.mean(healthy_blasts)) if healthy_blasts else 0.0,
                'p_value': float(p),
                'significant': bool(p < 0.05)
            }
        }

        # Morphology divergence (fit PCA on combined to be fair)
        aml_feats = np.vstack([self.feature_cache[s]['features'][:1000] for s in aml_ids]) if aml_ids else np.empty((0,1))
        hlth_feats = np.vstack([self.feature_cache[s]['features'][:1000] for s in healthy_ids]) if healthy_ids else np.empty((0,1))
        if len(aml_feats) and len(hlth_feats):
            both = np.vstack([aml_feats, hlth_feats])
            pca = PCA(n_components=min(50, both.shape[1]), random_state=SEED)
            both_p = pca.fit_transform(both)
            aa = both_p[:len(aml_feats)]
            hh = both_p[len(aml_feats):]
            wd = [wasserstein_distance(aa[:, i], hh[:, i]) for i in range(both_p.shape[1])]
            comp['morphological_divergence'] = {'mean_wasserstein': float(np.mean(wd)), 'max_wasserstein': float(np.max(wd))}

        with open(self.output_dir / 'aml_vs_healthy_comparison.json', 'w') as f:
            json.dump(comp, f, indent=2)
        print(f"  Blast burden: AML {comp['blast_burden']['aml_mean']:.1%} vs Healthy {comp['blast_burden']['healthy_mean']:.1%} (p={comp['blast_burden']['p_value']:.3e})")
        return comp

    def create_summary_visualizations(self):
        print("\n📊 Creating visualizations...")
        sample_ids = list(self.feature_cache.keys())
        if not sample_ids:
            print("  (No samples to visualize.)")
            return
        blast = [self.feature_cache[s]['blast_pct'] for s in sample_ids]
        labels = [self.feature_cache[s]['label'] for s in sample_ids]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1) Blast distribution
        ax = axes[0,0]
        hb = [b for b,l in zip(blast, labels) if l==0]
        ab = [b for b,l in zip(blast, labels) if l==1]
        ax.hist(hb, bins=20, alpha=0.5, label='Healthy')
        ax.hist(ab, bins=20, alpha=0.5, label='AML')
        ax.set_xlabel('Blast fraction'); ax.set_ylabel('Samples'); ax.set_title('Blast Burden'); ax.legend()

        # 2) Sample counts
        ax = axes[0,1]
        ax.bar(['Healthy','AML'], [labels.count(0), labels.count(1)])
        ax.set_ylabel('N samples'); ax.set_title('Sample Distribution')

        # 3) Feature embedding (subset)
        ax = axes[1,0]
        stacks, labs = [], []
        for sid in sample_ids[:10]:
            feats = self.feature_cache[sid]['features'][:200]
            if len(feats): stacks.append(feats); labs += [self.feature_cache[sid]['label']]*len(feats)
        if stacks:
            X = np.vstack(stacks)
            if HAVE_UMAP and len(X) > 200:
                emb = umap.UMAP(n_components=2, random_state=SEED).fit_transform(X)
            else:
                emb = PCA(n_components=2, random_state=SEED).fit_transform(X)
            colors = ['blue' if l==0 else 'red' for l in labs]
            ax.scatter(emb[:,0], emb[:,1], c=colors, alpha=0.3, s=5)
            ax.set_title('Feature Space (subset)'); ax.set_xlabel('1'); ax.set_ylabel('2')

        # 4) Confusion matrix from attention_analysis (if present)
        ax = axes[1,1]
        att_path = self.output_dir / 'attention_analysis.json'
        if att_path.exists():
            data = json.load(open(att_path))
            actual, pred = [], []
            for sid in sample_ids:
                if sid in data:
                    actual.append(self.feature_cache[sid]['label'])
                    pred.append(int(data[sid]['prediction']))
            if actual:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(actual, pred, labels=[0,1])
                im = ax.imshow(cm, interpolation='nearest')
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(['Healthy','AML']); ax.set_yticklabels(['Healthy','AML'])
                ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('MIL Confusion')
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i,j]), ha='center', va='center')

        plt.suptitle('Phase 2: Morpho-Phenotypic Analysis Summary', fontsize=14)
        plt.tight_layout()
        out = self.output_dir / 'phase2_summary.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  ✓ Saved visualization to {out}")

    def run_complete_analysis(self, max_samples: Optional[int] = None):
        print("\n" + "="*60)
        print("🚀 PHASE 2: Complete Morpho-Phenotypic Analysis with MIL")
        print("="*60)

        matched = self.match_samples()
        if max_samples is not None:
            matched = dict(list(matched.items())[:max_samples])

        self.build_feature_cache(matched)

        # Train + analyze if enough samples
        if len(self.feature_cache) >= 5:
            self.train_mil_model(epochs=50)
            self.analyze_attention_patterns()

        # Group comparison if both present
        aml_ids = [sid for sid, d in self.feature_cache.items() if d['sample_type']=='AML']
        hlth_ids = [sid for sid, d in self.feature_cache.items() if d['sample_type']=='Healthy']
        if aml_ids and hlth_ids:
            self.compare_aml_vs_healthy(aml_ids, hlth_ids)

        self.create_summary_visualizations()

        print("\n✅ Phase 2 complete! Check outputs/phase2 for:")
        print("  - feat_cache/, logs/")
        print("  - mil_model_best.pth, attention_analysis.json")
        print("  - aml_vs_healthy_comparison.json")
        print("  - phase2_summary.png")
        return self.feature_cache


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    config = {
        'arcsinh_cofactor': 150,
        'max_cells_per_sample': 1000,
        'batch_size': 4,
    }

    pipeline = Phase2Pipeline(
        image_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        fcs_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        model_path='models/last.ckpt',
        output_dir=Path('outputs/phase2'),
        config=config
    )

    pipeline.run_complete_analysis(max_samples=20)

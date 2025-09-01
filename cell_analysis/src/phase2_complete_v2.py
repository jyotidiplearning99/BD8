# src/phase2_complete_v2.py
"""
Phase 2: Complete morpho-phenotypic analysis for AML vs Healthy
- No TIFF→FCS per-cell mapping required
- Builds per-sample morphology feature caches
- Self-tuning blast gate from FCS (percentiles; CD45dim/SSClow fallback)
- Attention-based MIL (sample-level AML vs Healthy) + blast% regression
- Summary visualizations + attention inspection
- CLUSTERING WITH EXAMPLE TIFFS for visual validation
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
import shutil
import joblib
from math import sqrt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from random import Random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
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
import seaborn as sns

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
        self.marker_order = [
            "CD34","CD117","CD38","HLA-DR","CD13","CD33","CD15","CD16",
            "CD11b","CD56","CD7","MPO","CD45"
        ]
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
                print(f"  → Using FlowKit; arcsinh(cofactor={self.cofactor}) applied to numeric channels")
            except Exception as e:
                print(f"  ⚠️ FlowKit read failed ({e}); falling back to fcsparser...")

        if df is None and HAVE_FCSPARSER:
            meta, data = fcsparser.parse(str(fcs_path))
            df = pd.DataFrame(data)
            num = df.select_dtypes(include=[np.number]).columns
            df[num] = np.arcsinh(df[num] / self.cofactor)
            print(f"  → Using fcsparser; arcsinh(cofactor={self.cofactor}) applied to numeric channels")

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
            if len(feats) == 0 or data['label'] is None:    # skip empty or Unknown
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
    # Max instances and max feature width in this batch
    max_n = max(b['features'].shape[0] for b in batch)
    Ds = [b['features'].shape[1] for b in batch]
    D = max(Ds)
    B = len(batch)

    feats = torch.zeros(B, max_n, D, dtype=torch.float32)
    labels = torch.empty(B, dtype=torch.long)
    blast = torch.empty(B, 1, dtype=torch.float32)
    sids = []

    for i, b in enumerate(batch):
        x = b['features']
        n, d = x.shape
        feats[i, :n, :d] = x  # right-pad feature width if needed
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
        self.matched_samples: Dict[str, Dict] = {}  # Store matched samples
        self.mil_model: Optional[AttentionMIL] = None

    def match_samples(self) -> Dict[str, Dict]:
        matches = {}
        fcs_files = list(self.fcs_dir.rglob('*.fcs')) if self.fcs_dir else []
        print(f"\n📊 Found {len(fcs_files)} FCS files")
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
        self.matched_samples = matches  # Store for later use
        return matches

    def _compute_phenotype_vector(self, fcs_df: pd.DataFrame) -> np.ndarray:
        vals = []
        for m in self.fcs_processor.marker_order: 
            if m in fcs_df.columns:
                v = pd.to_numeric(fcs_df[m], errors='coerce')
                vals.append(float(np.nanmedian(v)))
            else:
                vals.append(0.0)  # impute missing marker as 0 after arcsinh
        return np.asarray(vals, dtype=np.float32)

    def build_feature_cache(self, matched_samples: Dict) -> Dict:
        print("\n🔬 Building feature cache...")
        rng = np.random.RandomState(SEED)

        for sid, data in matched_samples.items():
            print(f"\nProcessing {sid}...")
            fcs_df = self.fcs_processor.load_fcs(data['fcs_path'])

            # fixed-length phenotype vector in a consistent order
            phen_vec = self._compute_phenotype_vector(fcs_df)

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

            # append the SAME phenotype vector to every cell (fixed dimension)
            phen_tile = np.tile(phen_vec, (feats.shape[0], 1))
            feats = np.hstack([feats, phen_tile]).astype(np.float32)

            # Proper label assignment: None for Unknown
            label = 1 if data['sample_type'] == 'AML' else (0 if data['sample_type'] == 'Healthy' else None)
            
            self.feature_cache[sid] = {
                'features': feats,
                'label': label,
                'blast_pct': blast_pct,
                'sample_type': data['sample_type'],
                'image_paths': img_paths  # Store the image paths for clustering
            }
            print(f"  ✓ Cached {len(feats)} cells, blast={blast_pct:.1%}, feature_dim={feats.shape[1]}")
        return self.feature_cache

    def perform_clustering_with_examples(self, 
                                        n_clusters: int = 8, 
                                        examples_per_cluster: int = 20,
                                        max_examples_per_sample: int = 3,
                                        method: str = 'kmeans'):
        """
        Cluster morphological features and save example TIFFs for validation.
        Helps identify true morphological differences vs artifacts (cell clumps/duplexes).
        """
        print("\n🔬 Performing clustering analysis with example images...")
        print(f"  Method: {method}, Clusters: {n_clusters}, Examples/cluster: {examples_per_cluster}")
        
        # Collect all features and track source images
        all_features = []
        image_paths = []
        sample_ids = []
        sample_types = []
        
        for sid, data in self.feature_cache.items():
            imgs = data.get('image_paths', [])
            
            for i, feat in enumerate(data['features'][:self.max_cells_per_sample]):
                if i < len(imgs):
                    all_features.append(feat)
                    image_paths.append(imgs[i])
                    sample_ids.append(sid)
                    sample_types.append(data['sample_type'])
        
        if not all_features:
            print("  ⚠️ No features available for clustering")
            return None, None
        
        print(f"  Total cells for clustering: {len(all_features)}")
        
        # Perform clustering (using only morphological features, excluding phenotype)
        X = np.vstack(all_features)
        
        # Determine morphological feature dimension (exclude phenotype markers)
        n_markers = len(self.fcs_processor.marker_order)
        morph_dim = X.shape[1] - n_markers
        X_morph = X[:, :morph_dim]
        
        print(f"  Morphological feature dimension: {morph_dim}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_morph)
        
        # Clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
            clusters = clusterer.fit_predict(X_scaled)
        elif method == 'dbscan':
            # DBSCAN for detecting outliers (potential artifacts)
            clusterer = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            clusters = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"  DBSCAN found {n_clusters} clusters + {(clusters == -1).sum()} outliers")
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Save the scaler and clusterer
        joblib.dump({'scaler': scaler, 'clusterer': clusterer, 'morph_dim': morph_dim},
                    self.output_dir / 'cluster_model.joblib')
        print(f"  ✓ Saved clustering model to cluster_model.joblib")
        
        # Create output directory for cluster examples
        cluster_dir = self.output_dir / "cluster_examples"
        if cluster_dir.exists():
            shutil.rmtree(cluster_dir)
        cluster_dir.mkdir(exist_ok=True)
        
        # Analyze and save examples for each cluster
        unique_clusters = sorted(set(clusters))
        cluster_info = {}
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                cluster_name = "outliers"
            else:
                cluster_name = f"cluster_{cluster_id:02d}"
            
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Create cluster subdirectory
            cluster_subdir = cluster_dir / cluster_name
            cluster_subdir.mkdir(exist_ok=True)
            
            # Save full membership CSV with proper quoting
            with open(cluster_subdir / "members.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "image_path", "sample_id", "sample_type"])
                for j in cluster_indices:
                    w.writerow([j, str(image_paths[j]), sample_ids[j], sample_types[j]])
            
            # Balanced example selection across samples
            n_examples = min(examples_per_cluster, len(cluster_indices))
            copied_files = []  # Initialize before conditional
            
            if n_examples > 0:
                # Group indices by sample_id to ensure diversity
                sample_to_indices = {}
                for idx in cluster_indices:
                    sid = sample_ids[idx]
                    if sid not in sample_to_indices:
                        sample_to_indices[sid] = []
                    sample_to_indices[sid].append(idx)
                
                # Select examples with balanced sampling
                example_indices = []
                samples_list = list(sample_to_indices.keys())
                
                # Round-robin selection from different samples
                sample_idx = 0
                while len(example_indices) < n_examples:
                    sid = samples_list[sample_idx % len(samples_list)]
                    sample_cells = sample_to_indices[sid]
                    
                    # Count how many we've already taken from this sample
                    taken_from_sample = sum(1 for idx in example_indices 
                                          if sample_ids[idx] == sid)
                    
                    # If we haven't exceeded the per-sample cap and have cells left
                    if taken_from_sample < max_examples_per_sample and len(sample_cells) > taken_from_sample:
                        # Take the next untaken cell from this sample
                        untaken = [idx for idx in sample_cells if idx not in example_indices]
                        if untaken:
                            example_indices.append(untaken[0])
                    
                    sample_idx += 1
                    
                    # Break if we've cycled through all samples without finding more
                    if sample_idx >= len(samples_list) * max_examples_per_sample:
                        break
                
                # Copy example TIFFs with metadata
                for i, idx in enumerate(example_indices):
                    src_path = Path(image_paths[idx])
                    sample_info = f"{sample_types[idx]}_{sample_ids[idx]}"
                    dst_name = f"example_{i:03d}_{sample_info}_{src_path.stem}.tiff"
                    dst_path = cluster_subdir / dst_name
                    
                    try:
                        # Check if same filesystem for hardlink
                        same_fs = (cluster_subdir.stat().st_dev == src_path.stat().st_dev)
                        if same_fs:
                            os.link(src_path, dst_path)
                        else:
                            shutil.copy2(src_path, dst_path)
                        copied_files.append(dst_name)
                    except OSError:
                        # Hardlink failed for any reason → copy as fallback
                        try:
                            shutil.copy2(src_path, dst_path)
                            copied_files.append(dst_name)
                        except Exception as ee:
                            print(f"    Warning: Could not copy {src_path.name}: {ee}")
                            continue
                    
                    # Create preview PNG (first 10 examples)
                    if i < 10:
                        try:
                            img = tifffile.imread(str(src_path))
                            # Handle both grayscale and RGB
                            if img.ndim == 2:
                                # Grayscale
                                img = img.astype(np.float32)
                                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                                img = (img * 255).astype(np.uint8)
                            elif img.ndim == 3:
                                # RGB or multi-channel
                                if img.shape[0] <= 4:
                                    img = np.transpose(img[:3], (1, 2, 0))
                                if img.shape[-1] > 3:
                                    img = img[..., :3]
                                img = img.astype(np.float32)
                                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                                img = (img * 255).astype(np.uint8)
                            
                            # Save as PNG for quick viewing
                            png_path = cluster_subdir / f"preview_{i:03d}.png"
                            cv2.imwrite(str(png_path), img)
                        except Exception:
                            pass
            
            # Calculate cluster statistics
            cluster_samples = [sample_types[i] for i in cluster_indices]
            aml_count = cluster_samples.count('AML')
            healthy_count = cluster_samples.count('Healthy')
            unknown_count = cluster_samples.count('Unknown')
            
            # Get morphological statistics
            cluster_morph = X_morph[cluster_indices]
            morph_mean = cluster_morph.mean(axis=0)
            morph_std = cluster_morph.std(axis=0)
            
            # Save cluster statistics
            cluster_stats = {
                "cluster_id": int(cluster_id) if cluster_id != -1 else "outliers",
                "n_cells": int(cluster_mask.sum()),
                "n_aml_cells": int(aml_count),
                "n_healthy_cells": int(healthy_count),
                "n_unknown_cells": int(unknown_count),
                "aml_ratio": float(aml_count / max(1, aml_count + healthy_count)),
                "example_files": copied_files,
                "morphology_stats": {
                    "mean_features": morph_mean.tolist()[:10],  # First 10 features
                    "std_features": morph_std.tolist()[:10],
                    "feature_variance": float(morph_std.mean())
                }
            }
            
            if cluster_id != -1 and method == 'kmeans':
                cluster_stats["centroid"] = clusterer.cluster_centers_[cluster_id].tolist()[:10]
            
            # Write cluster info
            with open(cluster_subdir / "cluster_info.json", "w") as f:
                json.dump(cluster_stats, f, indent=2)
            
            cluster_info[cluster_name] = cluster_stats
            
            print(f"  {cluster_name}: {cluster_stats['n_cells']} cells "
                  f"(AML: {aml_count}, Healthy: {healthy_count}, Unknown: {unknown_count}), "
                  f"{len(copied_files)} examples saved")
        
        # Create a summary visualization
        self._create_cluster_visualization(X_scaled, clusters, cluster_info, method)
        
        # Detect potential artifacts (cell clumps/duplexes)
        artifact_info = self._detect_potential_artifacts(X_morph, clusters, image_paths)
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "n_clusters": int(n_clusters),
            "total_cells": len(clusters),
            "cluster_info": cluster_info,
            "artifact_detection": artifact_info
        }
        
        with open(cluster_dir / "clustering_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Saved {n_clusters} clusters with example TIFFs to {cluster_dir}")
        print(f"  ✓ Check cluster_examples/ subdirectories for visual inspection")
        print(f"  ✓ Review clustering_summary.json for statistics")
        print(f"  ✓ members.csv in each cluster has full cell membership")
        
        return clusters, image_paths

    def _detect_potential_artifacts(self, X_morph: np.ndarray, clusters: np.ndarray, 
                                   image_paths: List[Path]) -> Dict:
        """
        Detect potential artifacts like cell clumps or duplexes based on morphological outliers.
        """
        print("\n  🔍 Detecting potential artifacts...")
        
        # Calculate outlier scores using distance to nearest cluster center
        from sklearn.neighbors import NearestNeighbors
        
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(X_morph)
        distances, _ = nn.kneighbors(X_morph)
        
        # Average distance to 10 nearest neighbors
        avg_distances = distances.mean(axis=1)
        
        # Identify outliers (top 5% distances)
        threshold = np.percentile(avg_distances, 95)
        outlier_mask = avg_distances > threshold
        outlier_indices = np.where(outlier_mask)[0]
        
        # Save examples of potential artifacts
        artifact_dir = self.output_dir / "cluster_examples" / "potential_artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        n_artifact_examples = min(20, len(outlier_indices))
        if n_artifact_examples > 0:
            selected_indices = np.random.choice(outlier_indices, n_artifact_examples, replace=False)
            
            for i, idx in enumerate(selected_indices):
                src_path = Path(image_paths[idx])
                dst_path = artifact_dir / f"artifact_{i:03d}_{src_path.stem}.tiff"
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception:
                    pass
        
        artifact_info = {
            "n_potential_artifacts": int(outlier_mask.sum()),
            "artifact_percentage": float(outlier_mask.mean() * 100),
            "outlier_threshold": float(threshold),
            "examples_saved": n_artifact_examples
        }
        
        print(f"  Found {artifact_info['n_potential_artifacts']} potential artifacts "
              f"({artifact_info['artifact_percentage']:.1f}% of cells)")
        
        return artifact_info

    def _create_cluster_visualization(self, X_scaled: np.ndarray, clusters: np.ndarray, 
                                     cluster_info: Dict, method: str):
        """Create visualization of clustering results."""
        print("\n  📊 Creating cluster visualization...")
        
        # Reduce dimensions for visualization
        if X_scaled.shape[0] > 5000:
            # Subsample for faster computation
            idx = np.random.choice(X_scaled.shape[0], 5000, replace=False)
            X_vis = X_scaled[idx]
            clusters_vis = clusters[idx]
        else:
            X_vis = X_scaled
            clusters_vis = clusters
        
        # Use UMAP or PCA for visualization
        if HAVE_UMAP and X_vis.shape[0] > 100:
            reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30)
            X_2d = reducer.fit_transform(X_vis)
        else:
            pca = PCA(n_components=2, random_state=SEED)
            X_2d = pca.fit_transform(X_vis)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster scatter plot
        ax = axes[0, 0]
        unique_clusters = sorted(set(clusters_vis))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters_vis == cluster_id
            label = f"Cluster {cluster_id}" if cluster_id != -1 else "Outliers"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], 
                      alpha=0.6, s=10, label=label)
        
        ax.set_title(f'{method.upper()} Clustering Results')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 2. Cluster size distribution
        ax = axes[0, 1]
        cluster_sizes = [info['n_cells'] for info in cluster_info.values()]
        cluster_names = list(cluster_info.keys())
        ax.bar(range(len(cluster_names)), cluster_sizes)
        ax.set_xticks(range(len(cluster_names)))
        ax.set_xticklabels(cluster_names, rotation=45, ha='right')
        ax.set_ylabel('Number of cells')
        ax.set_title('Cluster Size Distribution')
        
        # 3. AML vs Healthy distribution per cluster
        ax = axes[1, 0]
        aml_counts = [info['n_aml_cells'] for info in cluster_info.values()]
        healthy_counts = [info['n_healthy_cells'] for info in cluster_info.values()]
        
        x = np.arange(len(cluster_names))
        width = 0.35
        ax.bar(x - width/2, aml_counts, width, label='AML', color='red', alpha=0.7)
        ax.bar(x + width/2, healthy_counts, width, label='Healthy', color='blue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_names, rotation=45, ha='right')
        ax.set_ylabel('Number of cells')
        ax.set_title('Disease Distribution per Cluster')
        ax.legend()
        
        # 4. Feature variance per cluster
        ax = axes[1, 1]
        variances = [info['morphology_stats']['feature_variance'] 
                    for info in cluster_info.values()]
        ax.bar(range(len(cluster_names)), variances)
        ax.set_xticks(range(len(cluster_names)))
        ax.set_xticklabels(cluster_names, rotation=45, ha='right')
        ax.set_ylabel('Average Feature Variance')
        ax.set_title('Morphological Heterogeneity per Cluster')
        
        plt.suptitle(f'Morphological Clustering Analysis ({method.upper()})', fontsize=14)
        plt.tight_layout()
        
        output_path = self.output_dir / "cluster_examples" / "clustering_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved visualization to {output_path}")

    def train_mil_model(self, epochs: int = 50):
        print("\n🎯 Training MIL model...")
        
        # Keep only AML/Healthy for supervised training (exclude Unknown)
        sids_all = list(self.feature_cache.keys())
        sids = [s for s in sids_all if self.feature_cache[s]['sample_type'] in ('AML', 'Healthy')]
        
        # Report excluded samples
        n_excluded = len(sids_all) - len(sids)
        if n_excluded > 0:
            print(f"  ℹ️ Excluded {n_excluded} samples with Unknown/other labels from training")
        
        if len(sids) < 4:
            print("  ⚠️ Not enough labeled (AML/Healthy) samples for MIL training (need ≥4). Skipping.")
            return None
        
        # Count samples per class
        labels = [self.feature_cache[s]['label'] for s in sids]
        n_aml = sum(labels)
        n_healthy = len(labels) - n_aml
        
        print(f"  Dataset: {n_aml} AML, {n_healthy} Healthy samples")
        
        # Adjust validation strategy based on sample size
        from sklearn.model_selection import train_test_split
        
        if len(sids) < 10:
            # For small datasets, use shuffled manual split
            if n_aml >= 2 and n_healthy >= 2:
                # Shuffle before splitting to avoid bias
                rng = Random(SEED)
                
                aml_sids = [s for s in sids if self.feature_cache[s]['label'] == 1]
                healthy_sids = [s for s in sids if self.feature_cache[s]['label'] == 0]
                
                # Shuffle to avoid always taking the same samples
                rng.shuffle(aml_sids)
                rng.shuffle(healthy_sids)
                
                # Ensure at least 1 of each class in validation
                test_size = max(2, int(0.2 * len(sids)))
                print(f"  Small dataset: using {test_size} samples for validation")
                
                # Take at least 1 from each class for validation
                val_aml = aml_sids[:max(1, min(len(aml_sids), test_size // 2))]
                val_healthy = healthy_sids[:max(1, min(len(healthy_sids), test_size - len(val_aml)))]
                
                val_ids = val_aml + val_healthy
                train_ids = [s for s in sids if s not in val_ids]
            else:
                print("  ⚠️ Need at least 2 samples per class for training. Skipping.")
                return None
        else:
            # Standard stratified split for larger datasets
            test_size = 0.2
            train_ids, val_ids = train_test_split(sids, test_size=test_size, 
                                                 stratify=labels, random_state=SEED)
        
        print(f"  Training with {len(train_ids)} samples, validating with {len(val_ids)} samples")
        train_labels = [self.feature_cache[s]['label'] for s in train_ids]
        val_labels = [self.feature_cache[s]['label'] for s in val_ids]
        print(f"  Train: AML={sum(train_labels)}, Healthy={len(train_labels)-sum(train_labels)}")
        print(f"  Val: AML={sum(val_labels)}, Healthy={len(val_labels)-sum(val_labels)}")
        
        train_data = {sid: self.feature_cache[sid] for sid in train_ids}
        val_data = {sid: self.feature_cache[sid] for sid in val_ids}
        
        train_ds = SampleLevelDataset(train_data, max_cells=self.max_cells_per_sample)
        val_ds = SampleLevelDataset(val_data, max_cells=self.max_cells_per_sample)
        
        # Adjust batch size for small datasets
        batch_size = min(4, len(train_ids))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, batch_size=len(val_ids), shuffle=False, collate_fn=pad_collate)
        
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
                x = batch['features'].to(device)
                y = batch['label'].to(device)
                b = batch['blast_pct'].to(device)
                
                logits, bpred = self.mil_model(x)
                loss = cls_crit(logits, y) + 0.1 * reg_crit(bpred, b)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            
            # Validation
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
                    'epoch': epoch+1,
                    'n_train_samples': len(train_ids),
                    'n_val_samples': len(val_ids)
                }, self.output_dir / 'mil_model_best.pth')
        
        print(f"✓ Best validation accuracy: {best_val:.3f}")
        return self.mil_model

    def analyze_attention_patterns(self):
        if not self.mil_model:
            return {}
        print("\n🎯 Analyzing attention patterns...")
        device = next(self.mil_model.parameters()).device
        self.mil_model.eval()
        out = {}
        with torch.no_grad():
            for sid, data in self.feature_cache.items():
                x = torch.from_numpy(data['features'].astype(np.float32)).unsqueeze(0).to(device)
                logits, bpred, A = self.mil_model(x, return_attention=True)
                pred_cls = torch.argmax(logits, dim=1).item()
                att = A.squeeze(0).detach().cpu().numpy()
                top_idx = np.argsort(att)[-10:]
                out[sid] = {
                    'prediction': int(pred_cls) if data['label'] is not None else None,
                    'blast_prediction': float(bpred.item()),
                    'actual_blast': float(data['blast_pct']),
                    'top_attention_indices': top_idx.tolist(),
                    'top_attention_weights': att[top_idx].tolist(),
                    'sample_type': data['sample_type']
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
        sample_types = [self.feature_cache[s]['sample_type'] for s in sample_ids]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1) Blast distribution by sample type
        ax = axes[0,0]
        hb = [b for b,t in zip(blast, sample_types) if t=='Healthy']
        ab = [b for b,t in zip(blast, sample_types) if t=='AML']
        ub = [b for b,t in zip(blast, sample_types) if t=='Unknown']
        
        if hb: ax.hist(hb, bins=20, alpha=0.5, label=f'Healthy (n={len(hb)})')
        if ab: ax.hist(ab, bins=20, alpha=0.5, label=f'AML (n={len(ab)})')
        if ub: ax.hist(ub, bins=20, alpha=0.5, label=f'Unknown (n={len(ub)})')
        
        ax.set_xlabel('Blast fraction'); ax.set_ylabel('Samples'); ax.set_title('Blast Burden'); ax.legend()

        # 2) Sample counts
        ax = axes[0,1]
        counts = [sample_types.count('Healthy'), sample_types.count('AML'), sample_types.count('Unknown')]
        ax.bar(['Healthy','AML','Unknown'], counts)
        ax.set_ylabel('N samples'); ax.set_title('Sample Distribution')

        # 3) Feature embedding (subset)
        ax = axes[1,0]
        stacks, types = [], []
        for sid in sample_ids[:10]:
            feats = self.feature_cache[sid]['features'][:200]
            if len(feats): 
                stacks.append(feats)
                types += [self.feature_cache[sid]['sample_type']]*len(feats)
        if stacks:
            X = np.vstack(stacks)
            if HAVE_UMAP and len(X) > 200:
                emb = umap.UMAP(n_components=2, random_state=SEED).fit_transform(X)
            else:
                emb = PCA(n_components=2, random_state=SEED).fit_transform(X)
            
            color_map = {'Healthy': 'blue', 'AML': 'red', 'Unknown': 'gray'}
            colors = [color_map.get(t, 'gray') for t in types]
            ax.scatter(emb[:,0], emb[:,1], c=colors, alpha=0.3, s=5)
            ax.set_title('Feature Space (subset)'); ax.set_xlabel('1'); ax.set_ylabel('2')

        # 4) Confusion matrix from attention_analysis (if present)
        ax = axes[1,1]
        att_path = self.output_dir / 'attention_analysis.json'
        if att_path.exists():
            data = json.load(open(att_path))
            # Only consider AML/Healthy for confusion matrix
            actual, pred = [], []
            for sid in sample_ids:
                if sid in data and self.feature_cache[sid]['sample_type'] in ('AML', 'Healthy'):
                    if data[sid]['prediction'] is not None:
                        actual.append(self.feature_cache[sid]['label'])
                        pred.append(int(data[sid]['prediction']))
            if actual:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(actual, pred, labels=[0,1])
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(['Healthy','AML']); ax.set_yticklabels(['Healthy','AML'])
                ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('MIL Confusion (excluding Unknown)')
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i,j]), ha='center', va='center', 
                               color='white' if cm[i,j] > cm.max()/2 else 'black')

        plt.suptitle('Phase 2: Morpho-Phenotypic Analysis Summary', fontsize=14)
        plt.tight_layout()
        out = self.output_dir / 'phase2_summary.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  ✓ Saved visualization to {out}")

    def run_complete_analysis(self, max_samples: Optional[int] = None, 
                             perform_clustering: bool = True,
                             clustering_method: str = 'kmeans',
                             n_clusters: int = 8):
        print("\n" + "="*60)
        print("🚀 PHASE 2: Complete Morpho-Phenotypic Analysis with MIL")
        print("="*60)

        matched = self.match_samples()
        if max_samples is not None:
            matched = dict(list(matched.items())[:max_samples])

        self.build_feature_cache(matched)

        # Perform clustering analysis if requested
        if perform_clustering and len(self.feature_cache) > 0:
            self.perform_clustering_with_examples(
                n_clusters=n_clusters,
                examples_per_cluster=20,
                max_examples_per_sample=3,  # Balanced sampling
                method=clustering_method
            )

        # Count labeled samples for MIL training decision
        labeled_samples = [s for s in self.feature_cache if self.feature_cache[s]['sample_type'] in ('AML', 'Healthy')]
        
        # Train + analyze if enough labeled samples
        if len(labeled_samples) >= 4:
            self.train_mil_model(epochs=50)
            self.analyze_attention_patterns()
        else:
            print(f"\n  ℹ️ Skipping MIL training (only {len(labeled_samples)} labeled samples, need ≥4)")

        # Group comparison if both present
        aml_ids = [sid for sid, d in self.feature_cache.items() if d['sample_type']=='AML']
        hlth_ids = [sid for sid, d in self.feature_cache.items() if d['sample_type']=='Healthy']
        if aml_ids and hlth_ids:
            self.compare_aml_vs_healthy(aml_ids, hlth_ids)

        self.create_summary_visualizations()

        print("\n✅ Phase 2 complete! Check outputs/phase2 for:")
        print("  - feat_cache/, logs/")
        print("  - cluster_examples/ (with TIFF examples per cluster)")
        print("  - cluster_model.joblib (saved scaler + clusterer)")
        if len(labeled_samples) >= 4:
            print("  - mil_model_best.pth, attention_analysis.json")
        print("  - aml_vs_healthy_comparison.json")
        print("  - phase2_summary.png")
        print("\n📁 Cluster validation:")
        print("  - Review cluster_examples/cluster_XX/ for morphological patterns")
        print("  - Check cluster_examples/potential_artifacts/ for cell clumps/duplexes")
        print("  - See clustering_summary.json for statistics")
        print("  - members.csv in each cluster folder has full cell membership")
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

    # Run with clustering enabled
    pipeline.run_complete_analysis(
        max_samples=20,
        perform_clustering=True,
        clustering_method='kmeans',  # or 'dbscan' for outlier detection
        n_clusters=8
    )

# src/phase2_fcs_gating.py
"""
Phase 2: Production-ready morpho-phenotypic analysis for AML vs Healthy
Adds deeper myeloid/blast analysis, robust memory handling, and schema-safe I/O.
"""

import os
import io
import csv
import cv2
import json
import yaml
import time
import math
import tifffile
import hashlib
import random
import contextlib
import warnings
from math import sqrt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import umap

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

# ============ FCS backends ============
HAVE_FLOWKIT = False
HAVE_FCSPARSER = False
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
    raise RuntimeError("Neither flowkit nor fcsparser is available. Please install at least one reader.")

# ===============================
# Utils
# ===============================
def _safe_csv_writer(fh, fieldnames):
    """CSV writer with Excel-friendly dialect; avoids delimiter/quote issues."""
    return csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')


# ===============================
# FCS Processor
# ===============================

class FCSProcessor:
    """Process FCS files with dynamic channel mapping, gating, and logging"""

    def __init__(self, cofactor: float = 150, log_dir: Path = Path("outputs/phase2/logs")):
        self.cofactor = float(cofactor)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_fcs = None

        # Expanded marker dictionary for deeper myeloid analysis + common panel synonyms
        self.marker_mappings = {
            'CD34' : ['CD34','CD34-A','CD34 BV785','CD34 BV786','CD34-BV421','CD34-V450','CD34-PerCP'],
            'CD117': ['CD117','CD117-A','CD117-PE','CD117-APC','CKIT','KIT','cKit'],
            'CD38' : ['CD38','CD38-A','CD38-FITC','CD38-PE-CY7','CD38 PE-Fire700'],
            'HLA-DR':['HLA-DR','HLA-DR-A','HLA-DR-APC-CY7','HLA-DR-V500','HLA-DR BV605','HLADR'],
            'CD45' : ['CD45','CD45-A','CD45-KO','CD45-V500','CD45 V500','CD45 BV510','V500-A','BV510-A'],
            'CD13' : ['CD13','CD13-A','CD13-PE','CD13-FITC'],
            'CD33' : ['CD33','CD33-A','CD33-PE-CY7','CD33-APC'],
            'CD14' : ['CD14','CD14-A','CD14-FITC','CD14-PerCP','CD14 BV650','BV650-A'],
            'CD15' : ['CD15','CD15-A','CD15-APC','CD15-FITC'],
            'CD16' : ['CD16','CD16-A','CD16-PE','CD16-APC-CY7','BV570-A'],
            'CD11b': ['CD11b','CD11b-A','CD11B-PE','CD11B-APC'],
            'CD56' : ['CD56','CD56-A','CD56-PE','CD56-APC','PE-A'],
            'CD7'  : ['CD7','CD7-A','CD7-PE','CD7-APC'],
            'MPO'  : ['MPO','MPO-A','MPO PE','PE*-A'],
            # lymphoid/context markers that help phenotypic context
            'CD3'  : ['CD3','CD3-A','BV785*-A','CD3 BV785','CD3-BV785'],
            'CD4'  : ['CD4','CD4-A','BV421-A','Alexa Fluor 647-A','CD4 AF647-A'],
            'CD8'  : ['CD8','CD8-A','BV711-A'],
            'CD19' : ['CD19','CD19-A','BV605-A'],
            # Viability/dead-like channels used for pre-gate
            'DEAD' : ['DEAD','DEAD-A','ZOMBIE','ZOMBIE NIR*-A','HELIX','HELIX NP NIR*-A','VIABILITY','LIVE/DEAD'],
        }

    # -------- I/O --------

    def load_fcs(self, fcs_path: Path) -> pd.DataFrame:
        """Load FCS file with arcsinh transform and dynamic channel mapping."""
        fcs_path = Path(fcs_path)
        print(f"  📄 FCS file: {fcs_path}")
        self.last_fcs = fcs_path.stem
        meta = None
        df = None
        flowkit_err = None

        # Try FlowKit first
        if HAVE_FLOWKIT:
            try:
                sample = fk.Sample(str(fcs_path))
                if getattr(sample, 'compensation', None) is not None:
                    sample.apply_compensation(sample.compensation)
                df = sample.as_dataframe()
                print(f"  ↳ Using flowkit; arcsinh(cofactor={self.cofactor}) applied to numeric channels")
                num = df.select_dtypes(include=[np.number]).columns
                df[num] = np.arcsinh(df[num] / self.cofactor)
                try:
                    meta = getattr(sample, 'metadata', None)
                except Exception:
                    meta = None
            except Exception as e:
                flowkit_err = e

        # Fallback to fcsparser if needed
        if df is None and HAVE_FCSPARSER:
            try:
                meta, data = fcsparser.parse(str(fcs_path))
                df = pd.DataFrame(data)
                print(f"  ↳ Using fcsparser; arcsinh(cofactor={self.cofactor}) applied to numeric channels"
                      + (f" (FlowKit error: {flowkit_err})" if flowkit_err else ""))
                num = df.select_dtypes(include=[np.number]).columns
                df[num] = np.arcsinh(df[num] / self.cofactor)
            except Exception as e:
                raise RuntimeError(f"Both FlowKit and fcsparser failed on {fcs_path.name}. "
                                   f"FlowKit error: {flowkit_err}; fcsparser error: {e}")

        if df is None:
            raise RuntimeError(f"No FCS backend succeeded for {fcs_path.name}.")

        raw_cols = df.columns.tolist()
        print("  Raw FCS columns (first 20):", raw_cols[:20])

        # Channel map preview & save
        channel_rows = []
        try:
            for i, col in enumerate(raw_cols, start=1):
                pnn = meta.get(f"$P{i}N") if isinstance(meta, dict) else None
                pns = meta.get(f"$P{i}S") if isinstance(meta, dict) else None
                channel_rows.append({"index": i, "data_col": col, "$PnS": pns, "$PnN": pnn})
        except Exception:
            pass

        if channel_rows:
            print("  Channel → marker preview (first 30):")
            for r in channel_rows[:30]:
                print(f"{r['index']:>5}  {str(r['data_col'])[:28]:<28}  "
                      f"$PnS:{str(r['$PnS'])[:38]:<38}  $PnN:{str(r['$PnN'])[:38]}")

            try:
                map_csv = self.log_dir / f"{fcs_path.stem}_channel_map.csv"
                with open(map_csv, "w", newline="") as fh:
                    w = _safe_csv_writer(fh, ["index", "data_col", "$PnS", "$PnN"])
                    w.writeheader()
                    for row in channel_rows:
                        w.writerow(row)
                print(f"  ↳ Full channel map saved to: {map_csv}")
            except Exception as _e:
                print(f"  (Could not write channel map CSV: {_e})")

        # Standardize columns
        df = self._standardize_columns(df, meta=meta)
        print("  Standardized columns (first 20):", df.columns.tolist()[:20])

        # Log mapping
        self._log_marker_mapping(df, fcs_path)
        return df

    def _standardize_columns(self, df: pd.DataFrame, meta: Optional[dict] = None) -> pd.DataFrame:
        new_columns = {}
        search_texts = {}
        # Build a single uppercase search string from col, $PnS, $PnN
        for i, col in enumerate(df.columns, start=1):
            pieces = [str(col)]
            if isinstance(meta, dict):
                pnn = meta.get(f"$P{i}N")
                pns = meta.get(f"$P{i}S")
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
                        matched = True
                        break
                if matched:
                    break
            if not matched and any(x in text for x in ['FSC', 'SSC', 'TIME', 'LIGHTLOSS', 'ZOMBIE', 'VIABILITY']):
                new_columns[col] = col  # keep scatter/time/imaging names

        # Persist rename map
        try:
            rename_map_csv = self.log_dir / f"{self.last_fcs}_rename_map.csv"
            with open(rename_map_csv, "w", newline="") as fh:
                w = _safe_csv_writer(fh, ["original", "mapped"])
                w.writeheader()
                for k, v in new_columns.items():
                    w.writerow({"original": k, "mapped": v})
            print(f"  ↳ Canonical rename map saved to: {rename_map_csv}")
        except Exception as _e:
            print(f"  (Could not write rename map CSV: {_e})")

        df = df.rename(columns=new_columns)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _log_marker_mapping(self, df: pd.DataFrame, fcs_path: Path):
        used = {m: (m in df.columns) for m in self.marker_mappings.keys()}
        log_file = self.log_dir / f"{fcs_path.stem}_marker_map.json"
        with open(log_file, "w") as fh:
            json.dump({
                'file': str(fcs_path),
                'timestamp': datetime.now().isoformat(),
                'markers_found': used,
                'all_columns': list(df.columns),
                'cofactor': self.cofactor
            }, fh, indent=2)
        print(f"  ↳ Marker presence log saved to: {log_file}")

    # -------- Viability pre-gate --------
    def gate_viable(self, df: pd.DataFrame) -> np.ndarray:
        """Return mask of viable cells (exclude high 'Dead/Viability/Helix/Zombie' signals)."""
        cand = [c for c in df.columns
                if any(t in c.upper() for t in ['DEAD', 'VIABILITY', 'LIVE/DEAD', 'ZOMBIE', 'HELIX', 'NIR'])]
        if not cand:
            return np.ones(len(df), dtype=bool)
        v = pd.to_numeric(df[cand[0]], errors='coerce').values
        if not np.isfinite(v).any():
            return np.ones(len(df), dtype=bool)
        thr = float(np.nanpercentile(v[~np.isnan(v)], 50))  # bottom 50% as viable
        return np.nan_to_num(v, nan=np.inf) < thr

    # -------- Gating --------
    def gate_blast_population(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict, str]:
        """Define blast gate with thresholds and degeneracy check; returns (mask, thresholds, mode)."""
        n = len(df)
        thresholds = {}
        gates_applied = []
        mode = 'primary'
        gate = np.ones(n, dtype=bool)

        def pctl(col, q):
            colv = pd.to_numeric(df[col], errors='coerce')
            return float(np.nanpercentile(colv.dropna(), q)) if colv.notna().any() else 0.0

        primary_parts = 0
        if 'CD34' in df.columns:
            thr = pctl('CD34', 75)
            thresholds['CD34_P75'] = thr
            gate &= (pd.to_numeric(df['CD34'], errors='coerce').fillna(-1e9).values > thr)
            gates_applied.append('CD34+'); primary_parts += 1
        if 'CD117' in df.columns:
            thr = pctl('CD117', 70)
            thresholds['CD117_P70'] = thr
            gate &= (pd.to_numeric(df['CD117'], errors='coerce').fillna(-1e9).values > thr)
            gates_applied.append('CD117+'); primary_parts += 1
        if 'CD38' in df.columns:
            lo = pctl('CD38', 20); hi = pctl('CD38', 60)
            thresholds['CD38_P20'] = lo; thresholds['CD38_P60'] = hi
            v = pd.to_numeric(df['CD38'], errors='coerce').values
            gate &= (np.nan_to_num(v, nan=-1e9) > lo) & (np.nan_to_num(v, nan=1e9) < hi)
            gates_applied.append('CD38dim'); primary_parts += 1
        if 'HLA-DR' in df.columns:
            thr = pctl('HLA-DR', 60)
            thresholds['HLA-DR_P60'] = thr
            gate &= (pd.to_numeric(df['HLA-DR'], errors='coerce').fillna(-1e9).values > thr)
            gates_applied.append('HLA-DR+'); primary_parts += 1

        frac = float(np.mean(gate)) if n else 0.0

        # Fallback CD45dim & SSClow when primary not viable
        if (primary_parts < 2) or (frac < 0.005) or (frac > 0.995):
            mode = 'fallback_cd45_ssc'
            cd45 = pd.to_numeric(df['CD45'], errors='coerce').values if 'CD45' in df.columns else None
            ssc_candidates = [c for c in df.columns if 'SSC' in c and c.endswith('-A')]
            prefer = [c for c in ssc_candidates if c.upper() in ('SSC-A',) or 'IMAGING' in c.upper()]
            ssc_col = prefer[0] if prefer else (ssc_candidates[0] if ssc_candidates else None)
            ssc = pd.to_numeric(df[ssc_col], errors='coerce').values if ssc_col else None

            if cd45 is not None and np.isfinite(cd45).any():
                cd45_low = float(np.nanpercentile(cd45[~np.isnan(cd45)], 30))
                thresholds['CD45_P30'] = cd45_low
                mask_cd45 = np.nan_to_num(cd45, nan=1e9) < cd45_low
            else:
                mask_cd45 = np.ones(n, dtype=bool)

            if ssc is not None and np.isfinite(ssc).any():
                ssc_low = float(np.nanpercentile(ssc[~np.isnan(ssc)], 30))
                thresholds['SSC_P30'] = ssc_low
                mask_ssc = np.nan_to_num(ssc, nan=1e9) < ssc_low
            else:
                mask_ssc = np.ones(n, dtype=bool)

            gate = mask_cd45 & mask_ssc
            frac = float(np.mean(gate))

        print(f"✓ Blast population: {gate.sum()}/{n} cells ({100*frac:.1f}%)  [{mode}]")
        threshold_file = self.log_dir / f"{self.last_fcs}_blast_thresholds.json"
        with open(threshold_file, "w") as fh:
            json.dump({
                'file': self.last_fcs,
                'timestamp': datetime.now().isoformat(),
                'thresholds': thresholds,
                'gates_applied': gates_applied,
                'gate_count': int(gate.sum()),
                'total_cells': int(n),
                'gate_percentage': float(frac * 100),
                'mode': mode
            }, fh, indent=2)
        print(f"  ↳ Thresholds saved to: {threshold_file}")
        return gate, thresholds, mode

    # -------- FlowJo export --------
    def export_for_flowjo(self, fcs_df: pd.DataFrame, gates: Dict[str, np.ndarray], out_dir: Path, stem: str):
        """Export gated populations (FCS if fcswrite available, else CSV)."""
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        df_out = fcs_df.copy()
        for gname, gmask in gates.items():
            if isinstance(gmask, np.ndarray) and len(gmask) == len(df_out):
                df_out[f"GATE_{gname}"] = gmask.astype(np.int8)

        wrote_fcs = False
        try:
            import fcswrite
            arr = df_out.to_numpy(); chn = list(map(str, df_out.columns))
            fpath = out_dir / f"{stem}_gated.fcs"
            fcswrite.write_fcs(str(fpath), chn, arr)
            print(f"  ↳ FCS with gates saved to: {fpath}")
            wrote_fcs = True
        except Exception as e:
            print(f"  ⚠️ fcswrite not available or failed ({e}); writing CSV fallback.")

        if not wrote_fcs:
            csv_path = out_dir / f"{stem}_gated.csv"
            df_out.to_csv(csv_path, index=False)
            print(f"  ↳ CSV fallback saved to: {csv_path}")


# ===============================
# Morpho-Phenotypic Analyzer
# ===============================

class MorphoPhenotypicAnalyzer:
    """Extract and combine morphological and phenotypic features (memory-safe)."""

    def __init__(self, model_path: str = 'models/last.ckpt',
                 cache_dir: Path = Path("outputs/phase2/feat_cache"),
                 inference_image_size_gpu: int = 512,
                 inference_image_size_cpu: int = 256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.feature_extractor = self._load_model(model_path)
        self.model.eval()
        self.img_size_gpu = int(inference_image_size_gpu)
        self.img_size_cpu = int(inference_image_size_cpu)

    def _load_model(self, model_path: Path) -> Tuple[nn.Module, nn.Module]:
        """
        Try to load user's BD_S8 model; if unavailable, fall back to a tiny CNN encoder
        so the pipeline still runs (feature size ~32).
        """
        # Try custom model
        try:
            from src.models import BD_S8_Model
            model = BD_S8_Model(num_classes=5)
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            clean = {}
            for k, v in state_dict.items():
                clean[k[6:]] = v if k.startswith('model.') else v
            model.load_state_dict(clean, strict=False)
            model = model.to(self.device)
            feature_extractor = getattr(model, 'encoder', None)
            if feature_extractor is None:
                raise RuntimeError("BD_S8_Model missing .encoder attribute.")
            return model, feature_extractor
        except Exception as e:
            print(f"  ⚠️ Could not load BD_S8 model ({e}); using fallback tiny CNN encoder.")
            encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ).to(self.device)
            # Wrap in a simple container with .encoder for compatibility
            class _Wrap(nn.Module):
                def __init__(self, enc): super().__init__(); self.encoder = enc
                def forward(self, x): return self.encoder(x)
            model = _Wrap(encoder).to(self.device)
            return model, encoder

    def _resize_side(self):
        return self.img_size_gpu if self.device.type == 'cuda' else self.img_size_cpu

    def _load_and_preprocess_image(self, path: Path) -> torch.Tensor:
        img = tifffile.imread(str(path))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[0] <= 4:  # CHW -> HWC
            img = np.transpose(img[:3], (1, 2, 0))
        if img.shape[-1] != 3:
            h, w = img.shape[:2]
            tmp = np.zeros((h, w, 3), dtype=np.float32)
            tmp[..., :min(3, img.shape[-1])] = img[..., :min(3, img.shape[-1])]
            img = tmp
        side = self._resize_side()
        img = cv2.resize(img.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)
        mx = float(img.max())
        if mx > 0:
            if mx > 255: img = img / 65535.0
            elif mx > 1: img = img / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    @torch.no_grad()
    def extract_morphological_features(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        feats_list: List[np.ndarray] = []
        batch_imgs: List[torch.Tensor] = []
        batch_save_paths: List[Path] = []

        effective_bs = min(batch_size, 8) if self.device.type == 'cpu' else batch_size

        def flush_batch(images: List[torch.Tensor], save_paths: List[Path]):
            if not images: return

            def _run(tensors: List[torch.Tensor], paths: List[Path], micro_bs: int):
                start = 0
                while start < len(tensors):
                    end = min(start + micro_bs, len(tensors))
                    bt = torch.stack(tensors[start:end]).to(self.device, non_blocking=True)
                    ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                           if self.device.type == 'cuda' else contextlib.nullcontext())
                    try:
                        with ctx:
                            out = self.feature_extractor(bt)
                        out = out.float().detach().cpu().numpy()
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ('allocate memory' in msg) or ('out of memory' in msg):
                            del bt
                            if self.device.type == 'cuda':
                                torch.cuda.empty_cache()
                            if micro_bs > 1:
                                return _run(tensors, paths, max(1, micro_bs // 2))
                            else:
                                # last resort: process each image separately
                                for idx in range(start, end):
                                    b1 = torch.stack(tensors[idx:idx+1]).to(self.device, non_blocking=True)
                                    with ctx:
                                        arr = self.feature_extractor(b1).float().detach().cpu().numpy()
                                    np.save(paths[idx], arr[0]); feats_list.append(arr[0])
                                return
                        else:
                            raise
                    for k in range(out.shape[0]):
                        np.save(paths[start + k], out[k]); feats_list.append(out[k])
                    del bt
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    start = end

            _run(images, save_paths, micro_bs=max(1, len(images)))

        for path in image_paths:
            rel = str(Path(path).resolve())
            hid = hashlib.sha1(rel.encode()).hexdigest()[:10]
            cache_path = self.cache_dir / f"{Path(path).stem}_{hid}.npy"
            if cache_path.exists():
                try:
                    feats_list.append(np.load(cache_path))
                    continue
                except Exception:
                    pass
            batch_imgs.append(self._load_and_preprocess_image(path))
            batch_save_paths.append(cache_path)
            if len(batch_imgs) >= effective_bs:
                flush_batch(batch_imgs, batch_save_paths)
                batch_imgs.clear(); batch_save_paths.clear()
        if batch_imgs:
            flush_batch(batch_imgs, batch_save_paths)
            batch_imgs.clear(); batch_save_paths.clear()

        if not feats_list:
            return np.array([])
        try:
            return np.vstack(feats_list)
        except Exception:
            return np.array(feats_list)


# ===============================
# Phase 2 Pipeline
# ===============================

class Phase2Pipeline:
    """Main pipeline with gating, advanced myeloid analysis, embeddings, clustering, and robust I/O."""

    def __init__(self, image_dir: Path, fcs_dir: Path, model_path: str, output_dir: Path, config: Optional[Dict] = None):
        self.image_dir = Path(image_dir)
        self.fcs_dir = Path(fcs_dir)
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.cofactor = float(self.config.get('arcsinh_cofactor', 150.))
        self.umap_neighbors = int(self.config.get('umap_neighbors', 30))
        self.umap_min_dist = float(self.config.get('umap_min_dist', 0.3))
        self.max_cells_per_sample = int(self.config.get('max_cells_per_sample', 1000))
        self.max_tsne_samples = int(self.config.get('max_tsne_samples', 20000))
        self.dbscan_eps = float(self.config.get('dbscan_eps', 1.5))

        self.fcs_processor = FCSProcessor(cofactor=self.cofactor, log_dir=self.output_dir / "logs")
        self.morpho_analyzer = MorphoPhenotypicAnalyzer(
            model_path=model_path,
            cache_dir=self.output_dir / "feat_cache",
            inference_image_size_gpu=self.config.get('inference_image_size_gpu', 512),
            inference_image_size_cpu=self.config.get('inference_image_size_cpu', 256),
        )

    # -------- sample matching --------
    def match_samples(self) -> Dict[str, Dict]:
        matches = {}
        fcs_files = list(self.fcs_dir.rglob('*.fcs'))

        print(f"\n📁 Found {len(fcs_files)} FCS files")
        print(f"Searching images under: {self.image_dir}")

        for fcs_path in fcs_files:
            sample_id = fcs_path.stem
            image_paths: List[Path] = []

            for d in self.image_dir.rglob(f"{sample_id}_images_*"):
                if d.is_dir():
                    image_paths += list(d.rglob("*.tif"))
                    image_paths += list(d.rglob("*.tiff"))
            if not image_paths:
                image_paths += list(self.image_dir.rglob(f"{sample_id}_*.tif"))
                image_paths += list(self.image_dir.rglob(f"{sample_id}_*.tiff"))

            if not image_paths:
                def norm(s: str) -> str: return "".join(ch.lower() for ch in s if ch.isalnum())
                nstem = norm(sample_id)
                for tif in self.image_dir.rglob("*.tif"):
                    if norm(tif.stem).startswith(nstem): image_paths.append(tif)
                for tif in self.image_dir.rglob("*.tiff"):
                    if norm(tif.stem).startswith(nstem): image_paths.append(tif)

            if image_paths:
                image_paths = sorted(set(image_paths))
                pstr = str(fcs_path)
                if "/AML/" in pstr:
                    sample_type = "AML"; batch = "AML"
                elif "/Healthy BM/" in pstr:
                    sample_type = "Healthy"; batch = "Healthy BM"
                else:
                    sample_type = "Healthy" if "BM" in pstr else "AML" if "AML" in pstr else "Unknown"
                    batch = Path(self.image_dir).name
                matches[sample_id] = {"fcs_path": fcs_path, "image_paths": image_paths,
                                      "sample_type": sample_type, "batch": batch}
                print(f"  ✓ {sample_id}: {len(image_paths)} images, {sample_type}, batch={batch}")
            else:
                print(f"  ✗ {sample_id}: no images found")

        return matches

    # -------- embedding & clustering --------
    def _create_embedding_safe(self, features: np.ndarray, method: str = 'umap') -> np.ndarray:
        try:
            if method == 'umap':
                reducer = umap.UMAP(n_components=2, n_neighbors=self.umap_neighbors, min_dist=self.umap_min_dist,
                                    metric='euclidean', random_state=SEED)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, perplexity=30, early_exaggeration=12, random_state=SEED)
            else:
                reducer = PCA(n_components=2, random_state=SEED)
            return reducer.fit_transform(features)
        except Exception as e:
            print(f"⚠️ {method.upper()} failed ({e}); falling back to PCA.")
            return PCA(n_components=2, random_state=SEED).fit_transform(features)

    def _identify_subpopulations(self, embedding: np.ndarray, features: np.ndarray, labels: np.ndarray) -> Dict:
        results = {}
        if len(embedding) > 100:
            k_values = range(3, min(20, max(4, len(embedding)//50)))
            scores = []
            for k in k_values:
                km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
                clusters = km.fit_predict(embedding)
                try:
                    s = silhouette_score(embedding, clusters)
                except Exception:
                    s = -1
                scores.append(s)
            if scores:
                best_k = list(k_values)[int(np.argmax(scores))]
                kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
                results['kmeans_clusters'] = kmeans.fit_predict(embedding)
                print(f"  K-means: k={best_k}, silhouette={max(scores):.3f}")

        min_samples = max(10, int(sqrt(len(embedding))))
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=min_samples)
        results['dbscan_clusters'] = dbscan.fit_predict(embedding)
        n_clusters = len(set(results['dbscan_clusters'])) - (1 if -1 in results['dbscan_clusters'] else 0)
        print(f"  DBSCAN: {n_clusters} clusters (eps={self.dbscan_eps}, min_samples={min_samples})")

        n_components = min(8, max(2, len(embedding)//100))
        if n_components > 1:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=SEED)
            results['gmm_clusters'] = gmm.fit_predict(embedding)
            print(f"  GMM: {n_components} components")
        return results

    # -------- Myeloid differentiation scoring & helpers --------
    def _percentile_gate(self, series: pd.Series, p_low: float = 20, p_high: float = 80) -> Tuple[float, float]:
        s = pd.to_numeric(series, errors='coerce')
        if not s.notna().any():
            return (0.0, 0.0)
        return (float(np.nanpercentile(s, p_low)), float(np.nanpercentile(s, p_high)))

    def _token_to_mask(self, df: pd.DataFrame, token: str) -> np.ndarray:
        """
        Convert tokens like 'CD34+', 'CD34-', 'CD38dim', 'CD38-/dim', 'HLA-DR+' to boolean masks.
        Missing markers → neutral (all True) so they don't erroneously zero-out a stage.
        """
        token = token.strip()
        if '-/dim' in token:  # handle variants like CD38-/dim
            base = token.split('-/dim')[0]
            return self._token_to_mask(df, base + '-') | self._token_to_mask(df, base + 'dim')

        if token.endswith('+'):
            base = token[:-1]
            if base not in df.columns:
                return np.ones(len(df), dtype=bool)
            lo, _ = self._percentile_gate(df[base], 75, 95)  # >= P75 positive
            v = pd.to_numeric(df[base], errors='coerce').values
            return np.nan_to_num(v, nan=-1e9) >= lo
        elif token.endswith('-'):
            base = token[:-1]
            if base not in df.columns:
                return np.ones(len(df), dtype=bool)
            _, hi = self._percentile_gate(df[base], 5, 25)  # <= P25 negative
            v = pd.to_numeric(df[base], errors='coerce').values
            return np.nan_to_num(v, nan=1e9) <= hi
        elif token.lower().endswith('dim'):
            base = token.replace('dim', '').replace('-', '').strip()
            if base not in df.columns:
                return np.ones(len(df), dtype=bool)
            s = pd.to_numeric(df[base], errors='coerce')
            if not s.notna().any():
                return np.ones(len(df), dtype=bool)
            lo = float(np.nanpercentile(s.dropna(), 20))
            hi = float(np.nanpercentile(s.dropna(), 60))
            v = s.values
            vv = np.nan_to_num(v, nan=0.0)
            return (vv > lo) & (vv < hi)
        else:
            return self._token_to_mask(df, token + '+')  # bare marker -> '+'

    def _apply_stage_gates(self, df: pd.DataFrame, tokens: List[str]) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        for t in tokens:
            mask &= self._token_to_mask(df, t)
        return mask

    def score_myeloid_differentiation(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Score myeloid differentiation stages; returns (stage_scores, stage_masks).
        Require >=2 markers for a stage to consider it 'sufficient' to avoid over-calling on sparse panels.
        """
        stages = {
            'stem_progenitor': ['CD34+', 'CD38dim', 'CD117+'],
            'promyelocyte'  : ['CD34-', 'CD117+', 'CD13+', 'CD33+', 'MPO+'],
            'myelocyte'     : ['CD34-', 'CD13+', 'CD33+', 'CD15+'],
            'metamyelocyte' : ['CD13+', 'CD15+', 'CD16+'],
            'neutrophil'    : ['CD13+', 'CD16+', 'CD15+', 'CD11b+'],
        }
        stage_scores, stage_masks = {}, {}
        present = set(df.columns)
        for stage, tokens in stages.items():
            bases = {t.replace('+', '').replace('-', '').replace('dim', '').replace('-/', '') for t in tokens}
            n_present = sum(b in present for b in bases)
            if n_present < 2:
                m = np.zeros(len(df), dtype=bool); pct, cnt = 0.0, 0; suff = False
            else:
                m = self._apply_stage_gates(df, tokens)
                pct, cnt, suff = float(m.mean()*100.0), int(m.sum()), True
            stage_masks[stage] = m
            stage_scores[stage] = {'percentage': pct, 'count': cnt,
                                   'sufficient_markers': suff, 'markers_present': int(n_present)}
        # maturation block heuristic only if early stages had sufficient markers
        if stage_scores['promyelocyte']['sufficient_markers'] and stage_scores['promyelocyte']['percentage'] > 30:
            stage_scores['maturation_block'] = 'promyelocyte'
        elif stage_scores['stem_progenitor']['sufficient_markers'] and stage_scores['stem_progenitor']['percentage'] > 20:
            stage_scores['maturation_block'] = 'stem_progenitor'
        else:
            stage_scores['maturation_block'] = None
        return stage_scores, stage_masks

    # -------- Aberrant phenotypes --------
    def detect_aberrant_phenotypes(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Detect selected AML-associated aberrant phenotypes; returns (info, masks).
        """
        ab, masks = {}, {}

        def pos_mask(col, q=70):
            if col not in df.columns:
                return np.zeros(len(df), dtype=bool)
            s = pd.to_numeric(df[col], errors='coerce')
            if not s.notna().any():
                return np.zeros(len(df), dtype=bool)
            thr = float(np.nanpercentile(s.dropna(), q))
            v = s.values
            return np.nan_to_num(v, nan=-1e9) > thr

        # CD34+CD56+ (poor prognosis)
        if all(m in df.columns for m in ['CD34', 'CD56']):
            m = pos_mask('CD34', 75) & pos_mask('CD56', 70)
            masks['aberrant_CD34+CD56+'] = m
            ab['CD34+CD56+'] = {'percentage': float(m.mean()*100.0), 'count': int(m.sum()),
                                'clinical_significance': 'poor_prognosis'}

        # CD34+CD7+ (lineage infidelity)
        if all(m in df.columns for m in ['CD34', 'CD7']):
            m = pos_mask('CD34', 75) & pos_mask('CD7', 70)
            masks['aberrant_CD34+CD7+'] = m
            ab['CD34+CD7+'] = {'percentage': float(m.mean()*100.0), 'count': int(m.sum()),
                               'clinical_significance': 'lineage_infidelity'}

        # CD117+CD15+ (abnormal co-expression)
        if all(m in df.columns for m in ['CD117', 'CD15']):
            m = pos_mask('CD117', 70) & pos_mask('CD15', 70)
            masks['aberrant_CD117+CD15+'] = m
            ab['CD117+CD15+'] = {'percentage': float(m.mean()*100.0), 'count': int(m.sum()),
                                 'clinical_significance': 'abnormal_maturation'}

        return ab, masks

    # -------- Advanced morpho-phenotype integration --------
    def integrate_morpho_phenotype_advanced(self,
                                            morphological_features: np.ndarray,
                                            phenotypic_df: pd.DataFrame,
                                            gates: Dict[str, np.ndarray]) -> Dict:
        """
        Gate-specific morphological signatures (weighted by population fraction) and
        gate-specific phenotypic median profiles.
        Note: without per-cell image↔event linkage, morphology is weighted at sample level.
        """
        results = {}
        if morphological_features is None or len(morphological_features) == 0:
            return results
        morph_mean = morphological_features.mean(axis=0)
        for gate_name, gate_mask in gates.items():
            if not isinstance(gate_mask, np.ndarray) or gate_mask.sum() < 10:
                continue
            frac = float(gate_mask.mean())
            weighted_morph = morph_mean * frac
            if phenotypic_df is not None and len(phenotypic_df) == len(gate_mask) and phenotypic_df.shape[1] > 0:
                phenotype_summary = phenotypic_df.loc[gate_mask].median(numeric_only=True).astype('float32').values
            else:
                phenotype_summary = np.array([], dtype=np.float32)
            results[gate_name] = {
                'morphological_signature': weighted_morph,
                'phenotypic_profile': phenotype_summary,
                'population_fraction': frac
            }
        return results

    # -------- Save / visualize / QC --------
    def _append_summary_row(self, row: dict):
        csv_path = self.output_dir / "sample_summaries.csv"
        tmp_path = csv_path.with_suffix(".tmp")
        new_df = pd.DataFrame([row])
        if csv_path.exists():
            try:
                old = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
            except Exception:
                old = pd.DataFrame()
            all_cols = sorted(set(old.columns).union(new_df.columns))
            old = old.reindex(columns=all_cols)
            new_df = new_df.reindex(columns=all_cols)
            out = pd.concat([old, new_df], ignore_index=True)
        else:
            out = new_df
        out.to_csv(tmp_path, index=False); tmp_path.replace(csv_path)

    def _save_sample_results(self, results: Dict, fcs_df: pd.DataFrame):
        sample_id = results['sample_id']

        np.save(self.output_dir / f"{sample_id}_umap.npy", results['embeddings']['umap'])
        np.save(self.output_dir / f"{sample_id}_tsne.npy", results['embeddings']['tsne'])

        summary_dict = {
            "sample_id": sample_id,
            "type": results['sample_type'],
            "gate_fractions": results['gate_fractions'],
            "thresholds": results['thresholds'],
            "embeddings_meta": results['embeddings_meta'],
            "stage_scores": results.get('stage_scores', {}),
            "aberrant_phenotypes": results.get('aberrant_phenotypes', {}),
        }
        json_path = self.output_dir / f"{sample_id}_summary.json"
        with open(json_path, "w") as fh: json.dump(summary_dict, fh, indent=2)
        yaml_path = self.output_dir / f"{sample_id}_summary.yaml"
        with open(yaml_path, "w") as fh: yaml.safe_dump(summary_dict, fh, sort_keys=False)
        print(f"  ↳ Summary JSON: {json_path}")
        print(f"  ↳ Summary YAML: {yaml_path}")

        sample_summary = {
            "sample_id": sample_id,
            "sample_type": results['sample_type'],
            "n_fcs_cells": int(len(fcs_df)),
            "n_images_used": int(len(results['features']['morphological'])),
            "run_timestamp": datetime.now().isoformat(),
            "umap_neighbors": self.umap_neighbors,
            "umap_min_dist": self.umap_min_dist,
            "dbscan_eps": self.dbscan_eps,
            "tsne_is_subset": results['embeddings_meta']['tsne_is_subset'],
            "tsne_n": results['embeddings_meta']['tsne_n'],
            **{f"gate_{k}": v for k, v in results['gate_fractions'].items()}
        }
        self._append_summary_row(sample_summary)
        np.save(self.output_dir / f"{sample_id}_results.npy", results, allow_pickle=True)

    def visualize_results(self, results: Dict):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sample_id = results['sample_id']; sample_type = results['sample_type']
        fig = plt.figure(figsize=(16, 10))
        emb = results['embeddings']['umap']

        ax1 = plt.subplot(2, 3, 1)
        colors = ['red' if sample_type == 'AML' else 'blue'] * len(emb)
        ax1.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.5, s=1)
        ax1.set_title(f'UMAP - {sample_type}'); ax1.set_xlabel('UMAP 1'); ax1.set_ylabel('UMAP 2')

        ax2 = plt.subplot(2, 3, 2)
        if 'kmeans_clusters' in results['subpopulations']:
            cl = results['subpopulations']['kmeans_clusters']
            ax2.scatter(emb[:, 0], emb[:, 1], c=cl, cmap='tab20', alpha=0.5, s=1)
            ax2.set_title('K-means Clusters')
        ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2')

        ax3 = plt.subplot(2, 3, 3)
        if 'dbscan_clusters' in results['subpopulations']:
            cl = results['subpopulations']['dbscan_clusters']
            ax3.scatter(emb[:, 0], emb[:, 1], c=cl, cmap='tab20', alpha=0.5, s=1)
            ax3.set_title(f'DBSCAN (eps={self.dbscan_eps})')
        ax3.set_xlabel('UMAP 1'); ax3.set_ylabel('UMAP 2')

        ax4 = plt.subplot(2, 3, 4)
        tsne = results['embeddings']['tsne']; meta = results['embeddings_meta']
        title = f't-SNE ({meta["tsne_n"]} points' + (f', subset of {len(emb)})' if meta['tsne_is_subset'] else ')')
        ax4.set_title(title); ax4.scatter(tsne[:, 0], tsne[:, 1], c='gray', alpha=0.5, s=1)
        ax4.set_xlabel('t-SNE 1'); ax4.set_ylabel('t-SNE 2')

        ax5 = plt.subplot(2, 3, 5)
        try:
            from scipy.stats import gaussian_kde
            xy = emb.T; z = gaussian_kde(xy)(xy)
            ax5.scatter(emb[:, 0], emb[:, 1], c=z, cmap='hot', s=1); ax5.set_title('Cell Density')
        except Exception:
            ax5.set_title('Density (skipped)')
        ax5.set_xlabel('UMAP 1'); ax5.set_ylabel('UMAP 2')

        ax6 = plt.subplot(2, 3, 6)
        gate_fractions = results['gate_fractions']
        if gate_fractions:
            gates = list(gate_fractions.keys()); fracs = list(gate_fractions.values())
            y_pos = np.arange(len(gates)); bars = ax6.barh(y_pos, fracs)
            for bar, frac in zip(bars, fracs):
                bar.set_color('red' if frac > 0.5 else 'orange' if frac > 0.2 else 'green')
            ax6.set_yticks(y_pos); ax6.set_yticklabels(gates)
            ax6.set_xlabel('Fraction'); ax6.set_title('Gate Fractions (Sample-level)'); ax6.set_xlim([0, 1])

        plt.suptitle(f'Sample {sample_id} - Morpho-Phenotypic Analysis', fontsize=14)
        plt.tight_layout()
        save_path = self.output_dir / f'{sample_id}_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"✓ Saved visualization to {save_path}")
        return save_path

    # -------- AML vs Healthy comparison --------
    def compare_aml_healthy_populations(self, aml_results: List[Dict], healthy_results: List[Dict]) -> Dict:
        from scipy.stats import mannwhitneyu
        comparison = {}
        aml_blasts = [r.get('gate_fractions', {}).get('blast', 0.0) for r in aml_results]
        healthy_blasts = [r.get('gate_fractions', {}).get('blast', 0.0) for r in healthy_results]
        if aml_blasts and healthy_blasts:
            try:
                _, p_value = mannwhitneyu(aml_blasts, healthy_blasts, alternative='greater')
            except Exception:
                p_value = 1.0
            comparison['blast_burden'] = {
                'aml_mean': float(np.mean(aml_blasts)),
                'healthy_mean': float(np.mean(healthy_blasts)),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        return comparison

    # -------- Batch/QC helpers --------
    def detect_batch_effects(self, samples_results: List[Dict]) -> Dict:
        out = {"gate_blast_by_batch": {}, "outlier_samples": []}
        per_batch = {}
        for r in samples_results:
            b = r.get('batch', 'unknown')
            frac = r.get('gate_fractions', {}).get('blast')
            if frac is None:
                continue
            per_batch.setdefault(b, []).append(frac)
        for b, vals in per_batch.items():
            vals = np.array(vals, dtype=float)
            out["gate_blast_by_batch"][b] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": int(len(vals))}
        all_vals = [r.get('gate_fractions', {}).get('blast') for r in samples_results if r.get('gate_fractions', {}).get('blast') is not None]
        if all_vals:
            mu = float(np.mean(all_vals)); sd = float(np.std(all_vals)) + 1e-8
            for r in samples_results:
                frac = r.get('gate_fractions', {}).get('blast')
                if frac is None:
                    continue
                if abs(frac - mu) > 3 * sd:
                    out["outlier_samples"].append({"sample_id": r.get('sample_id'), "blast": float(frac)})
        return out

    def cross_sample_qc(self, samples_results: List[Dict]) -> Dict:
        qc = {}
        gates = set()
        for r in samples_results:
            gates.update(r.get('gate_fractions', {}).keys())
        for g in gates:
            vals = [r['gate_fractions'][g] for r in samples_results if g in r.get('gate_fractions', {})]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            qc[g] = {"mean": float(arr.mean()), "std": float(arr.std()),
                     "cv": float(arr.std()/(arr.mean()+1e-8)), "n": int(len(arr))}
        return qc

    # -------- main per-sample --------
    def process_sample(self, sample_id: str, sample_data: Dict) -> Dict:
        print(f"\n{'='*60}\nProcessing sample: {sample_id}\n{'='*60}")

        results = {'sample_id': sample_id, 'sample_type': sample_data['sample_type']}

        # 1. FCS
        print("\n📊 Loading FCS data...")
        fcs_df = self.fcs_processor.load_fcs(sample_data['fcs_path'])
        print(f"  Loaded {len(fcs_df)} events with {len(fcs_df.columns)} channels")

        # 1b. Viability pre-gate (optional but recommended)
        viable_mask = self.fcs_processor.gate_viable(fcs_df)
        if viable_mask.mean() < 0.99:
            print(f"  Viability pre-gate: kept {viable_mask.sum()}/{len(viable_mask)} events ({100*viable_mask.mean():.1f}%)")
        fcs_df = fcs_df.loc[viable_mask].reset_index(drop=True)

        # 2. Blast gate
        print("\n🔬 Applying gates (blast + myeloid hierarchy)...")
        blast_gate, blast_thresholds, blast_mode = self.fcs_processor.gate_blast_population(fcs_df)
        results['gates'] = {'blast': blast_gate}
        results['thresholds'] = {'blast': blast_thresholds, 'blast_mode': blast_mode}

        # 3. Phenotypic summary table (only mapped markers, not scatter/viability)
        skip_cols = set(['DEAD'])
        mapped = [m for m in self.fcs_processor.marker_mappings.keys() if m in fcs_df.columns and m not in skip_cols]
        marker_cols = [c for c in fcs_df.columns if c in mapped]
        if marker_cols:
            phenotypic_df = fcs_df[marker_cols].copy()
            spec_summary_vec = phenotypic_df.median(numeric_only=True).astype('float32').values
        else:
            print("⚠️ No mapped markers found; continuing with morphology only.")
            phenotypic_df = pd.DataFrame(index=fcs_df.index)  # empty
            spec_summary_vec = np.zeros(1, dtype=np.float32)
        print(f"  Phenotypic summary: {len(spec_summary_vec)} features (sample-level median)")

        # 4. Myeloid stage scores + masks
        stage_scores, stage_masks = self.score_myeloid_differentiation(fcs_df)
        results['stage_scores'] = stage_scores
        for name, m in stage_masks.items():
            results['gates'][f'stage_{name}'] = m

        # 5. Aberrant phenotypes
        aberrant_info, aberrant_masks = self.detect_aberrant_phenotypes(fcs_df)
        results['aberrant_phenotypes'] = aberrant_info
        for name, m in aberrant_masks.items():
            results['gates'][name] = m

        # 6. Images → morphology
        image_paths = sample_data['image_paths']
        if len(image_paths) > self.max_cells_per_sample:
            random.seed(SEED)
            image_paths = random.sample(image_paths, self.max_cells_per_sample)
        print(f"\n🖼️ Extracting features from {len(image_paths)} images...")
        if image_paths:
            print(f"  First image parent dir: {Path(image_paths[0]).parent}")
        morphological_features = self.morpho_analyzer.extract_morphological_features(image_paths, batch_size=32)
        n_cells = len(morphological_features)
        if n_cells == 0:
            print("⚠️ No images found/usable after sampling; saving minimal summary and skipping rest.")
            self._append_summary_row({
                "sample_id": sample_id, "sample_type": results['sample_type'],
                "n_fcs_cells": int(len(fcs_df)), "n_images_used": 0,
                "run_timestamp": datetime.now().isoformat(),
                "umap_neighbors": self.umap_neighbors, "umap_min_dist": self.umap_min_dist,
                "dbscan_eps": self.dbscan_eps, "tsne_is_subset": False, "tsne_n": 0,
                "gate_blast": float(np.mean(results['gates']['blast'])) if len(fcs_df) else 0.0
            })
            return results

        # 7. Gate fractions (always include blast; drop degenerate others)
        gate_fractions = {}
        for gname, gmask in results['gates'].items():
            if not isinstance(gmask, np.ndarray):
                continue
            frac = float(np.mean(gmask)) if len(gmask) else 0.0
            if gname == 'blast' or (0.005 <= frac <= 0.995):
                gate_fractions[gname] = frac
        results['gate_fractions'] = gate_fractions
        pretty = {k: round(float(v), 5) for k, v in gate_fractions.items()}
        print(f"\n📊 Gate fractions (sample-level): {pretty}")

        # 8. Combine features
        print("\n🎯 Combining morphological and phenotypic features...")
        scaler_morph = StandardScaler(); morph_norm = scaler_morph.fit_transform(morphological_features)
        phen_mat = np.repeat(spec_summary_vec[None, :], n_cells, axis=0)
        scaler_phen = StandardScaler(); phen_norm = scaler_phen.fit_transform(phen_mat)
        alpha = 0.6
        combined = np.hstack([alpha*morph_norm, (1-alpha)*phen_norm])
        print(f"  Combined shape: {combined.shape} (morph {morph_norm.shape[1]} + phen {phen_norm.shape[1]})")

        results['features'] = {'morphological': morphological_features,
                               'phenotypic_summary': spec_summary_vec, 'combined': combined}

        # 9. Embeddings
        print("\n📈 Creating embeddings...")
        umap_embedding = self._create_embedding_safe(combined, method='umap')
        tsne_input = combined; tsne_is_subset = False
        if len(tsne_input) > self.max_tsne_samples:
            idx = np.random.RandomState(SEED).choice(len(tsne_input), self.max_tsne_samples, replace=False)
            tsne_input = tsne_input[idx]; tsne_is_subset = True
        tsne_embedding = self._create_embedding_safe(tsne_input, method='tsne')
        results['embeddings'] = {'umap': umap_embedding, 'tsne': tsne_embedding}
        results['embeddings_meta'] = {'tsne_is_subset': tsne_is_subset, 'tsne_n': int(len(tsne_input))}

        # 10. Clustering
        labels = np.ones(n_cells) if sample_data['sample_type'] == 'AML' else np.zeros(n_cells)
        subpop_results = self._identify_subpopulations(umap_embedding, combined, labels=labels)
        results['subpopulations'] = subpop_results

        # 11. Advanced integration
        results['integration_advanced'] = self.integrate_morpho_phenotype_advanced(
            morphological_features=morphological_features,
            phenotypic_df=phenotypic_df if len(phenotypic_df) else None,
            gates={'blast': blast_gate, **stage_masks, **aberrant_masks}
        )

        # 12. Save + FlowJo export
        self._save_sample_results(results, fcs_df)
        try:
            print("🧾 Exporting gated events for FlowJo...")
            self.fcs_processor.export_for_flowjo(fcs_df=fcs_df,
                                                 gates=results['gates'],
                                                 out_dir=self.output_dir / "flowjo_exports",
                                                 stem=sample_id)
        except Exception as e:
            print(f"  ⚠️ FlowJo export skipped: {e}")

        print(f"✓ {sample_id} ({sample_data['sample_type']}): images={n_cells}, FCS={len(fcs_df)}, blast={100.0*gate_fractions.get('blast',0.0):.1f}%")
        return results

    # -------- run many --------
    def run_analysis(self, max_samples: int = 5):
        print("\n" + "="*60)
        print("🚀 PHASE 2: Morpho-Phenotypic Analysis")
        print("    Building on Phase 1's 79% accurate model")
        print("="*60)
        print(f"📂 FCS dir   : {self.fcs_dir}")
        print(f"🖼️ Images dir: {self.image_dir}")
        print(f"📦 Output dir: {self.output_dir}\n")

        matched = self.match_samples()
        if not matched:
            print("❌ No matched samples found!")
            return []

        all_results = []
        for i, (sample_id, sample_data) in enumerate(matched.items()):
            if i >= max_samples:
                break
            try:
                sample_data['batch'] = sample_data.get('batch', Path(self.image_dir).name)
                r = self.process_sample(sample_id, sample_data)
                r['batch'] = sample_data['batch']
                all_results.append(r)
                self.visualize_results(r)
            except Exception as e:
                print(f"❌ Error processing {sample_id}: {e}")
                import traceback; traceback.print_exc()
                continue

        # Print summary (tolerant read)
        csv_path = self.output_dir / "sample_summaries.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
                cols = [c for c in ['sample_id', 'sample_type', 'n_images_used', 'gate_blast'] if c in df.columns]
                if cols:
                    print("\n📊 Summary of processed samples:")
                    print(df[cols].to_string(index=False))
            except Exception as e:
                print(f"⚠️ Could not parse sample_summaries.csv ({e}); file may be partially written.")

        print("\n✅ Phase 2 analysis complete!")
        return all_results


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    # Config tuned for myeloid/blast analysis
    config = {
        'arcsinh_cofactor': 150,
        'umap_neighbors': 30,
        'umap_min_dist': 0.3,
        'max_cells_per_sample': 2000,   # higher per-sample image cap for stats
        'max_tsne_samples': 20000,
        'dbscan_eps': 1.5,
        'inference_image_size_gpu': 512,
        'inference_image_size_cpu': 256,
    }

    pipeline = Phase2Pipeline(
        image_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        fcs_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        model_path='models/last.ckpt',
        output_dir=Path('outputs/phase2_myeloid'),
        config=config
    )

    results = pipeline.run_analysis(max_samples=20)

    # Optional post-hoc analyses:
    aml = [r for r in results if r.get('sample_type') == 'AML']
    healthy = [r for r in results if r.get('sample_type') == 'Healthy']
    if aml and healthy:
        comp = pipeline.compare_aml_healthy_populations(aml, healthy)
        with open(pipeline.output_dir / "aml_vs_healthy_comparison.json", "w") as fh:
            json.dump(comp, fh, indent=2)
        print("🧪 AML vs Healthy comparison saved.")

    batch_effects = pipeline.detect_batch_effects(results)
    with open(pipeline.output_dir / "batch_effects.json", "w") as fh:
        json.dump(batch_effects, fh, indent=2)

    qc_metrics = pipeline.cross_sample_qc(results)
    with open(pipeline.output_dir / "cross_sample_qc.json", "w") as fh:
        json.dump(qc_metrics, fh, indent=2)

    print("\n✅ Phase 2 complete! Check outputs for:")
    print("  - sample_summaries.csv (aggregate metrics; schema-safe)")
    print("  - *_summary.json/yaml (per-sample metadata incl. stage scores & aberrants)")
    print("  - aml_vs_healthy_comparison.json (if both groups present)")
    print("  - batch_effects.json, cross_sample_qc.json")
    print("  - *_umap.npy, *_tsne.npy (embeddings)")
    print("  - *_analysis.png (visualizations)")
    print("  - logs/, flowjo_exports/, feat_cache/")

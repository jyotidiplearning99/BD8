# src/phase2_complete_v3.py
"""
Phase 2: Complete morpho-phenotypic analysis with clustering and TIFF examples
Fixed version with proper feature dimension handling and all improvements
"""

from __future__ import annotations

import os
import csv
import json
import yaml
import hashlib
import random
import warnings
import shutil
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile
import cv2

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# Determinism
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Optional imports
try:
    import umap
    HAVE_UMAP = True
except:
    HAVE_UMAP = False

try:
    import fcsparser
    HAVE_FCSPARSER = True
except:
    HAVE_FCSPARSER = False

try:
    import flowkit as fk
    HAVE_FLOWKIT = True
except:
    HAVE_FLOWKIT = False

if not HAVE_FLOWKIT and not HAVE_FCSPARSER:
    raise ImportError("Install either 'flowkit' or 'fcsparser' to read FCS files.")


# =========================
# FCS PROCESSOR
# =========================
class FCSProcessor:
    """Process FCS with PBMC-specific markers."""
    def __init__(self, cofactor: float = 150, log_dir: Path = Path("outputs/phase2/logs"), 
                 dead_cell_percentile: float = 5.0):
        self.cofactor = float(cofactor)
        self.dead_cell_percentile = dead_cell_percentile  # Configurable, not hardcoded
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # PBMC panel markers
        self.marker_order = ["CD3","CD4","CD8","CD19","CD14","CD16","CD45","CD56"]
        
        self.marker_mappings = {
            'CD3': ['CD3-BV785','CD3 BV785','CD3'],
            'CD4': ['CD4-BV421','CD4 BV421','CD4'],
            'CD8': ['CD8a-BV711','CD8-BV711','CD8a','CD8'],
            'CD19': ['CD19-BV605','CD19 BV605','CD19'],
            'CD14': ['CD14-BV650','CD14 BV650','CD14'],
            'CD16': ['CD16-BV570','CD16 BV570','CD16'],
            'CD45': ['CD45-BV510','CD45 BV510','CD45'],
            'CD56': ['NK-PE','CD56-PE','CD56','NK'],
            'HELIX': ['Helix NIR','HELIX','Dead cells']
        }

    def load_fcs(self, fcs_path: Path) -> pd.DataFrame:
        """Load FCS and apply arcsinh transformation."""
        fcs_path = Path(fcs_path)
        print(f"  📄 FCS file: {fcs_path}")
        
        df = None
        meta = None
        
        if HAVE_FLOWKIT:
            try:
                sample = fk.Sample(str(fcs_path))
                df = sample.as_dataframe()
                # Get metadata if available
                meta = getattr(sample, 'metadata', None)
                num = df.select_dtypes(include=[np.number]).columns
                df[num] = np.arcsinh(df[num] / self.cofactor)
                print(f"  → Using FlowKit; arcsinh(cofactor={self.cofactor}) applied")
            except Exception as e:
                print(f"  ⚠️ FlowKit failed ({e}); trying fcsparser...")
        
        if df is None and HAVE_FCSPARSER:
            meta, data = fcsparser.parse(str(fcs_path))
            df = pd.DataFrame(data)
            num = df.select_dtypes(include=[np.number]).columns
            df[num] = np.arcsinh(df[num] / self.cofactor)
            print(f"  → Using fcsparser; arcsinh(cofactor={self.cofactor}) applied")
        
        if df is None:
            raise RuntimeError(f"Failed to read FCS: {fcs_path.name}")
        
        # Standardize columns with header metadata
        df = self._standardize_columns(df, meta)
        
        # Filter dead cells (configurable percentage)
        df = self._drop_dead(df, top_pct=self.dead_cell_percentile)
        
        return df

    def _standardize_columns(self, df: pd.DataFrame, meta: Optional[dict] = None) -> pd.DataFrame:
        """Map channel names to standard markers using header metadata when available."""
        new_columns = {}
        
        for i, col in enumerate(df.columns, start=1):
            # Build search text from column name and metadata
            search_parts = [str(col).upper()]
            if meta is not None:
                pnn = meta.get(f"$P{i}N")
                pns = meta.get(f"$P{i}S")
                if pnn: search_parts.append(str(pnn).upper())
                if pns: search_parts.append(str(pns).upper())
            search_text = " ".join(search_parts)
            
            matched = False
            for marker, variants in self.marker_mappings.items():
                for variant in variants:
                    if variant.upper() in search_text:
                        new_columns[col] = marker
                        matched = True
                        break
                if matched:
                    break
            
            if not matched and any(x in search_text for x in ['FSC','SSC','TIME','HELIX']):
                new_columns[col] = col
        
        df = df.rename(columns=new_columns)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _drop_dead(self, df: pd.DataFrame, viability_col: str = 'HELIX', top_pct: float = None) -> pd.DataFrame:
        """Filter out dead cells based on viability marker."""
        if top_pct is None:
            top_pct = self.dead_cell_percentile
            
        if viability_col in df.columns:
            v = pd.to_numeric(df[viability_col], errors='coerce')
            threshold = np.nanpercentile(v, 100 - top_pct)
            mask = v < threshold
            print(f"  → Filtered {(~mask).sum()} dead cells (top {top_pct}% of {viability_col})")
            return df[mask]
        return df

    def identify_cell_type(self, df: pd.DataFrame) -> str:
        """Identify cell type based on marker expression using fraction of positive events."""
        def frac_pos(col, q=0.7):
            """Calculate fraction of events above percentile threshold."""
            if col not in df.columns:
                return 0.0
            v = pd.to_numeric(df[col], errors='coerce')
            thr = np.nanpercentile(v, q * 100)
            return float(np.mean(v > thr))
        
        # Calculate fractions
        cd3p = frac_pos('CD3', 0.7)
        cd4p = frac_pos('CD4', 0.6)
        cd8p = frac_pos('CD8', 0.6)
        cd56p = frac_pos('CD56', 0.6)
        cd19p = frac_pos('CD19', 0.7)
        cd14p = frac_pos('CD14', 0.7)
        
        # Decision rules
        if cd3p > 0.3 and cd4p > 0.3 and cd8p < 0.2:
            return 'CD4_T'
        if cd3p > 0.3 and cd8p > 0.3 and cd4p < 0.2:
            return 'CD8_T'
        if cd3p > 0.2 and cd56p > 0.2:
            return 'NKT'
        if cd19p > 0.2:
            return 'B_cell'
        if cd14p > 0.2:
            return 'Monocyte'
        if cd56p > 0.3 and cd3p < 0.1:
            return 'NK'
        return 'Unknown'


# =========================
# MORPHOLOGICAL ANALYZER
# =========================
class MorphoPhenotypicAnalyzer:
    """Extract features with ResNet fallback and consistent dimensions."""
    def __init__(self, model_path: str = "models/last.ckpt", cache_dir: Path = Path("outputs/phase2/feat_cache")):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear old cache if model changes
        self.cache_version_file = self.cache_dir / "version.txt"
        self.model_identifier = self._get_model_identifier(model_path)
        self._validate_cache()
        
        self.feature_extractor, self.feature_dim = self._load_feature_extractor(model_path)
    
    def _get_model_identifier(self, model_path: str) -> str:
        """Get unique identifier for model to detect changes."""
        if Path(model_path).exists():
            return hashlib.md5((str(model_path) + str(Path(model_path).stat().st_mtime)).encode()).hexdigest()
        return "resnet50_fallback"
    
    def _validate_cache(self):
        """Clear cache if model has changed."""
        if self.cache_version_file.exists():
            with open(self.cache_version_file, 'r') as f:
                cached_version = f.read().strip()
            if cached_version != self.model_identifier:
                print("  ⚠️ Model changed, clearing feature cache...")
                for cache_file in self.cache_dir.glob("*.npy"):
                    cache_file.unlink()
        
        # Write current version
        with open(self.cache_version_file, 'w') as f:
            f.write(self.model_identifier)
    
    def _load_feature_extractor(self, model_path: str) -> Tuple[nn.Module, int]:
        """Load feature extractor with safe ResNet fallback."""
        try:
            # Try custom model
            from src.models import BD_S8_Model
            model = BD_S8_Model(num_classes=5).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            clean = {}
            for k, v in state_dict.items():
                clean[k[6:] if k.startswith('model.') else k] = v
            model.load_state_dict(clean, strict=False)
            model.eval()
            
            # Get feature dimension
            dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
            with torch.no_grad():
                feat = model.encoder(dummy_input)
                feature_dim = feat.shape[1] if feat.ndim == 2 else feat.view(1, -1).shape[1]
            
            print(f"  ✓ Loaded custom BD_S8_Model (feature_dim={feature_dim})")
            return model.encoder, feature_dim
            
        except Exception as e:
            print(f"  ⚠️ Could not load custom model ({e}). Using ResNet50 fallback...")
            
            # Safe ResNet loading (avoid downloads on cluster)
            try:
                import torchvision.models as models
                from torchvision.models import resnet50, ResNet50_Weights
                
                # Try with weights
                try:
                    weights = ResNet50_Weights.DEFAULT
                    resnet = resnet50(weights=weights).to(self.device)
                    print("  ✓ Loaded pretrained ResNet50")
                except:
                    # Load without weights
                    resnet = resnet50(weights=None).to(self.device)
                    
                    # Try to load cached weights
                    cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
                    weight_files = list(cache_dir.glob("resnet50*.pth"))
                    if weight_files:
                        resnet.load_state_dict(torch.load(weight_files[0], map_location=self.device))
                        print("  ✓ Loaded cached ResNet50 weights")
                    else:
                        print("  ⚠️ Using untrained ResNet50")
            except:
                import torchvision.models as models
                resnet = models.resnet50(pretrained=False).to(self.device)
                print("  ⚠️ Using untrained ResNet50")
            
            # Remove final layer
            feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            feature_extractor.eval()
            return feature_extractor, 2048  # ResNet50 outputs 2048 features

    def _load_and_preprocess_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess 5-channel TIFF using BF + fluorescence."""
        img = tifffile.imread(str(path))
        
        # Handle 5-channel images [B,G,Y,R,BF]
        if img.ndim == 3 and img.shape[0] == 5:
            B, G, Y, R, BF = img
            # Use BF as luminance, add Y and B for texture
            rgb = np.stack([BF, Y, B], axis=-1)
        elif img.ndim == 3 and img.shape[0] <= 5:
            rgb = np.transpose(img[:3], (1, 2, 0))
        elif img.ndim == 2:
            rgb = np.stack([img]*3, axis=-1)
        else:
            rgb = img[..., :3] if img.shape[-1] >= 3 else img
        
        # Ensure 3 channels
        if rgb.shape[-1] != 3:
            h, w = rgb.shape[:2]
            tmp = np.zeros((h, w, 3), dtype=np.float32)
            tmp[..., :min(3, rgb.shape[-1])] = rgb[..., :min(3, rgb.shape[-1])]
            rgb = tmp
        
        # Resize and normalize
        rgb = cv2.resize(rgb.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
        mx = float(rgb.max())
        if mx > 0:
            if mx > 255:
                rgb /= 65535.0
            elif mx > 1:
                rgb /= 255.0
        
        return torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).contiguous()

    @torch.no_grad()
    def extract_morphological_features(self, image_paths: List[Path]) -> np.ndarray:
        """Extract features with caching and consistent dimensions."""
        feats = []
        for pth in image_paths:
            # Create unique cache key
            rel = str(Path(pth).resolve())
            hid = hashlib.sha1(rel.encode()).hexdigest()[:10]
            cache = self.cache_dir / f"{Path(pth).stem}_{hid}.npy"
            
            if cache.exists():
                try:
                    feat = np.load(cache)
                    # Validate dimension
                    if feat.shape[0] == self.feature_dim:
                        feats.append(feat)
                        continue
                    else:
                        # Clear invalid cache
                        cache.unlink()
                except:
                    pass
            
            # Extract features
            img_tensor = self._load_and_preprocess_image(pth).unsqueeze(0).to(self.device)
            
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    z = self.feature_extractor(img_tensor).float().cpu().numpy()
            else:
                z = self.feature_extractor(img_tensor).cpu().numpy()
            
            # Flatten if needed
            if z.ndim > 2:
                z = z.reshape(z.shape[0], -1)
            
            # Ensure correct dimension
            if z.shape[1] != self.feature_dim:
                print(f"  Warning: Expected {self.feature_dim} features, got {z.shape[1]}")
                # Pad or truncate to match expected dimension
                if z.shape[1] < self.feature_dim:
                    z = np.pad(z, ((0, 0), (0, self.feature_dim - z.shape[1])), 'constant')
                else:
                    z = z[:, :self.feature_dim]
            
            np.save(cache, z[0])
            feats.append(z[0])
            
            del img_tensor, z
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        if not feats:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        
        # Stack and validate dimensions
        result = np.vstack(feats)
        assert result.shape[1] == self.feature_dim, f"Feature dimension mismatch: {result.shape[1]} vs {self.feature_dim}"
        return result


# =========================
# MAIN PIPELINE
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Load metadata if available
        self.meta = {}
        metadata_path = Path("metadata/experiments.yaml")
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.meta = yaml.safe_load(f)
        self.sample_to_experiment = self.meta.get("sample_to_experiment", {})
        
        # Initialize components
        dead_cell_pct = self.config.get('dead_cell_percentile', 5.0)  # Configurable
        self.fcs_processor = FCSProcessor(log_dir=self.output_dir / "logs", 
                                         dead_cell_percentile=dead_cell_pct)
        self.morpho_analyzer = MorphoPhenotypicAnalyzer(model_path=model_path, 
                                                       cache_dir=self.output_dir / "feat_cache")
        self.feature_cache = {}
        self.max_cells_per_sample = int(self.config.get('max_cells_per_sample', 1000))

    def identify_experiment(self, sample_name: str) -> str:
        """Identify experiment with metadata fallback."""
        # First check metadata
        if sample_name in self.sample_to_experiment:
            return self.sample_to_experiment[sample_name]
        
        # Fallback to heuristics
        name_lower = sample_name.lower()
        if any(x in name_lower for x in ['12.3', '12_3', 'exp_5', 'experiment_5']):
            return 'R5'
        elif any(x in name_lower for x in ['8.4', '8_4', 'exp_7', 'experiment_7']):
            return 'R7'
        elif any(x in name_lower for x in ['14.5', '14_5', 'exp_8', 'experiment_8']):
            return 'R8'
        
        return 'Unknown'

    def match_samples(self) -> Dict[str, Dict]:
        """Match FCS files with images."""
        matches = {}
        fcs_files = list(self.fcs_dir.rglob('*.fcs'))
        print(f"\n📊 Found {len(fcs_files)} FCS files")
        
        for fcs_path in fcs_files:
            sample_id = fcs_path.stem
            
            # Get experiment from metadata or heuristic
            experiment = self.sample_to_experiment.get(sample_id, self.identify_experiment(sample_id))
            
            # Find matching images
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
                
                # Determine sample type
                pstr = str(fcs_path)
                if '/AML/' in pstr or 'AML' in pstr:
                    sample_type = 'AML'
                elif any(x in pstr for x in ['/Healthy', 'Healthy BM', 'PBMC']):
                    sample_type = 'Healthy'
                else:
                    sample_type = 'Unknown'
                
                matches[sample_id] = {
                    "fcs_path": fcs_path,
                    "image_paths": image_paths,
                    "sample_type": sample_type,
                    "experiment": experiment
                }
                print(f"  ✓ {sample_id}: {len(image_paths)} images, {sample_type}, Exp: {experiment}")
            else:
                print(f"  ✗ {sample_id}: no images found")
        
        return matches

    def build_feature_cache(self, matched_samples: Dict) -> Dict:
        """Build feature cache."""
        print("\n🔬 Building feature cache...")
        rng = np.random.RandomState(SEED)
        
        for sid, data in matched_samples.items():
            print(f"\nProcessing {sid}...")
            
            # Load FCS
            fcs_df = self.fcs_processor.load_fcs(data['fcs_path'])
            
            # Compute phenotype vector
            phen_vec = []
            for m in self.fcs_processor.marker_order:
                if m in fcs_df.columns:
                    v = pd.to_numeric(fcs_df[m], errors='coerce')
                    phen_vec.append(float(np.nanmedian(v)))
                else:
                    phen_vec.append(0.0)
            phen_vec = np.asarray(phen_vec, dtype=np.float32)
            
            # Identify cell type
            cell_type = self.fcs_processor.identify_cell_type(fcs_df)
            
            # Sample images
            img_paths = data['image_paths']
            if len(img_paths) > self.max_cells_per_sample:
                idx = rng.choice(len(img_paths), size=self.max_cells_per_sample, replace=False)
                img_paths = [img_paths[i] for i in idx]
            
            # Extract features
            feats = self.morpho_analyzer.extract_morphological_features(img_paths)
            if feats.size == 0:
                print("  ⚠️ No features extracted; skipping sample.")
                continue
            
            # Combine morphological and phenotypic features
            phen_tile = np.tile(phen_vec, (feats.shape[0], 1))
            feats = np.hstack([feats, phen_tile]).astype(np.float32)
            
            # Proper label assignment: None for Unknown
            label = 1 if data['sample_type'] == 'AML' else (0 if data['sample_type'] == 'Healthy' else None)
            
            self.feature_cache[sid] = {
                'features': feats,
                'label': label,
                'sample_type': data['sample_type'],
                'experiment': data['experiment'],
                'cell_type': cell_type,
                'image_paths': img_paths
            }
            print(f"  ✓ Cached {len(feats)} cells, type={cell_type}, exp={data['experiment']}")
        
        return self.feature_cache

    def _detect_and_save_artifacts(self, X_morph: np.ndarray, image_paths: List[Path], out_dir: Path):
        """Detect and save potential artifacts (cell clumps/duplexes)."""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(X_morph)
        dists, _ = nn.kneighbors(X_morph)
        avg = dists.mean(axis=1)
        thr = np.percentile(avg, 95)
        out_idx = np.where(avg > thr)[0]
        
        art_dir = out_dir / "potential_artifacts"
        art_dir.mkdir(exist_ok=True, parents=True)
        
        for i, idx in enumerate(out_idx[:20]):  # Save up to 20 examples
            src = Path(image_paths[idx])
            dst = art_dir / f"artifact_{i:03d}_{src.stem}.tiff"
            try:
                # Hardlink when possible
                if art_dir.stat().st_dev == src.stat().st_dev:
                    os.link(src, dst)
                else:
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f"    Artifact copy/link failed for {src.name}: {e}")
        
        print(f"  Found {len(out_idx)} potential artifacts (~{100*len(out_idx)/len(X_morph):.1f}%)")
        return {"n_artifacts": int(len(out_idx)), "percentage": float(100*len(out_idx)/len(X_morph))}

    def perform_clustering_with_examples(self,
                                        n_clusters: int = 8,
                                        examples_per_cluster: int = 20,
                                        method: str = 'kmeans'):
        """Perform clustering and save example TIFFs."""
        print("\n🔬 Performing clustering analysis with example images...")
        print(f"  Method: {method}, Clusters: {n_clusters}")
        
        # Collect all features with metadata
        all_features = []
        image_paths = []
        sample_ids = []
        experiments = []
        cell_types = []
        sample_types = []
        
        for sid, data in self.feature_cache.items():
            imgs = data.get('image_paths', [])
            
            for i, feat in enumerate(data['features'][:self.max_cells_per_sample]):
                if i < len(imgs):
                    all_features.append(feat)
                    image_paths.append(imgs[i])
                    sample_ids.append(sid)
                    experiments.append(data['experiment'])
                    cell_types.append(data['cell_type'])
                    sample_types.append(data['sample_type'])
        
        if not all_features:
            print("  ⚠️ No features available for clustering")
            return None
        
        print(f"  Total cells for clustering: {len(all_features)}")
        
        # Extract morphological features only
        X = np.vstack(all_features)
        n_markers = len(self.fcs_processor.marker_order)
        X_morph = X[:, :-n_markers]  # Exclude phenotype markers
        
        print(f"  Morphological feature dimension: {X_morph.shape[1]}")
        
        # Standardize and cluster
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_morph)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
            clusters = clusterer.fit_predict(X_scaled)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            clusters = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"  DBSCAN found {n_clusters} clusters + {(clusters == -1).sum()} outliers")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save model
        joblib.dump({
            'scaler': scaler, 
            'clusterer': clusterer,
            'morph_dim': X_morph.shape[1]  # Save dimension info
        }, self.output_dir / 'cluster_model.joblib')
        
        # Create output directory
        cluster_dir = self.output_dir / "cluster_examples"
        if cluster_dir.exists():
            shutil.rmtree(cluster_dir)
        cluster_dir.mkdir(exist_ok=True)
        
        # Detect and save artifacts
        artifact_info = self._detect_and_save_artifacts(X_morph, image_paths, cluster_dir)
        
        # Process each cluster
        unique_clusters = sorted(set(clusters))
        cluster_info = {}
        
        for cluster_id in unique_clusters:
            cluster_name = "outliers" if cluster_id == -1 else f"cluster_{cluster_id:02d}"
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Create cluster subdirectory
            cluster_subdir = cluster_dir / cluster_name
            cluster_subdir.mkdir(exist_ok=True)
            
            # Save membership CSV
            with open(cluster_subdir / "members.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "image_path", "sample_id", "experiment", "cell_type", "sample_type"])
                for j in cluster_indices:
                    w.writerow([j, str(image_paths[j]), sample_ids[j], 
                               experiments[j], cell_types[j], sample_types[j]])
            
            # Select diverse examples
            n_examples = min(examples_per_cluster, len(cluster_indices))
            if n_examples > 0:
                # Group by experiment and cell type
                grouped = {}
                for idx in cluster_indices:
                    key = (experiments[idx], cell_types[idx])
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(idx)
                
                # Select balanced examples
                example_indices = []
                for (exp, ctype), indices in grouped.items():
                    n_from_group = min(3, len(indices))  # Max 3 per group
                    selected = np.random.choice(indices, n_from_group, replace=False)
                    example_indices.extend(selected)
                    if len(example_indices) >= n_examples:
                        break
                example_indices = example_indices[:n_examples]
                
                # Copy or hardlink TIFFs
                for i, idx in enumerate(example_indices):
                    src_path = Path(image_paths[idx])
                    metadata = f"{experiments[idx]}_{cell_types[idx]}_{sample_types[idx]}"
                    dst_name = f"example_{i:03d}_{metadata}_{src_path.stem}.tiff"
                    dst_path = cluster_subdir / dst_name
                    
                    try:
                        # Use hardlink if same filesystem
                        same_fs = (cluster_subdir.stat().st_dev == src_path.stat().st_dev)
                        if same_fs:
                            os.link(src_path, dst_path)
                        else:
                            shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        print(f"    Warning: Could not copy {src_path.name}: {e}")
            
            # Calculate cluster statistics
            cluster_exps = [experiments[i] for i in cluster_indices]
            cluster_types = [cell_types[i] for i in cluster_indices]
            cluster_samples = [sample_types[i] for i in cluster_indices]
            
            # Save cluster info
            cluster_stats = {
                "cluster_id": int(cluster_id) if cluster_id != -1 else "outliers",
                "n_cells": int(cluster_mask.sum()),
                "experiment_distribution": dict(Counter(cluster_exps)),
                "cell_type_distribution": dict(Counter(cluster_types)),
                "sample_type_distribution": dict(Counter(cluster_samples)),
            }
            
            with open(cluster_subdir / "cluster_info.json", "w") as f:
                json.dump(cluster_stats, f, indent=2)
            
            cluster_info[cluster_name] = cluster_stats
            
            print(f"  {cluster_name}: {cluster_stats['n_cells']} cells")
        
        # Save summary with artifact info
        summary = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "n_clusters": len(unique_clusters),
            "total_cells": len(clusters),
            "cluster_info": cluster_info,
            "artifact_detection": artifact_info
        }
        
        with open(cluster_dir / "clustering_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Saved {len(unique_clusters)} clusters with example TIFFs")
        print(f"  ✓ Detected {artifact_info['n_artifacts']} potential artifacts")
        print(f"  ✓ Check cluster_examples/ for visual validation")
        
        return clusters

    def run_complete_analysis(self, max_samples: Optional[int] = None,
                             perform_clustering: bool = True,
                             clustering_method: str = 'kmeans',
                             n_clusters: int = 8):
        """Run complete Phase 2 analysis."""
        print("\n" + "="*60)
        print("🚀 PHASE 2: Morpho-Phenotypic Analysis with Clustering")
        print("="*60)
        
        # Match samples
        matched = self.match_samples()
        if max_samples is not None:
            matched = dict(list(matched.items())[:max_samples])
        
        # Build feature cache
        self.build_feature_cache(matched)
        
        # Perform clustering
        if perform_clustering and len(self.feature_cache) > 0:
            self.perform_clustering_with_examples(
                n_clusters=n_clusters,
                examples_per_cluster=20,
                method=clustering_method
            )
        
        print("\n✅ Phase 2 complete! Check outputs/phase2 for:")
        print("  - feat_cache/ (cached features)")
        print("  - cluster_examples/ (TIFF examples per cluster)")
        print("  - cluster_examples/potential_artifacts/ (suspected artifacts)")
        print("  - clustering_summary.json")
        
        return self.feature_cache


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Create metadata directory and example YAML
    metadata_dir = Path("metadata")
    metadata_dir.mkdir(exist_ok=True)
    
    # Create example experiments.yaml if it doesn't exist
    yaml_path = metadata_dir / "experiments.yaml"
    if not yaml_path.exists():
        example_metadata = {
            "sample_to_experiment": {
                "Pre-sort 12.3.2025": "R5",
                "Pre-sort 8.4.2025": "R7",
                "Pre-sort 14.5.2025 Exp 8": "R8",
                "Pre-sort 12.2.2025": "R5",  # Corrected mapping
                "Pre-sort 8.5.2025 exp 7": "R7",
            }
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(example_metadata, f, default_flow_style=False)
    
    config = {
        'arcsinh_cofactor': 150,
        'max_cells_per_sample': 1000,
        'batch_size': 4,
        'dead_cell_percentile': 5.0  # Configurable, not hardcoded
    }
    
    pipeline = Phase2Pipeline(
        image_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        fcs_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        model_path='models/last.ckpt',
        output_dir=Path('outputs/phase2'),
        config=config
    )
    
    pipeline.run_complete_analysis(
        max_samples=20,
        perform_clustering=True,
        clustering_method='kmeans',
        n_clusters=8
    )

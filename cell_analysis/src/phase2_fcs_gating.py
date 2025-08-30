# src/phase2_fcs_gating.py
"""
Phase 2: Production-ready morpho-phenotypic analysis for AML vs Healthy
Building on Phase 1's 79% accurate model for myeloid/blast compartment analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import tifffile
import cv2
import json
import csv
import yaml
import contextlib
import hashlib
from datetime import datetime
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Headless
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============ DETERMINISM ============
import random, os
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# FCS processing
try:
    import flowkit as fk
except ImportError:
    print("Warning: flowkit not installed. Using fcsparser.")
    import fcsparser


class FCSProcessor:
    """Process FCS files with proper logging and gate transparency"""
    
    def __init__(self, 
                 cofactor: float = 150,
                 log_dir: Path = Path("outputs/phase2/logs")):
        
        self.cofactor = cofactor
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.marker_mappings = {
            'CD34': ['CD34-BV421', 'CD34-V450', 'CD34-PerCP', 'CD34'],
            'CD117': ['CD117-PE', 'CD117-APC', 'cKit', 'CD117'],
            'CD38': ['CD38-FITC', 'CD38-PE-Cy7', 'CD38'],
            'HLA-DR': ['HLA-DR-APC-Cy7', 'HLA-DR-V500', 'HLADR', 'HLA-DR'],
            'CD45': ['CD45-KO', 'CD45-V500', 'CD45'],
            'CD13': ['CD13-PE', 'CD13-FITC', 'CD13'],
            'CD33': ['CD33-PE-Cy7', 'CD33-APC', 'CD33'],
            'CD14': ['CD14-FITC', 'CD14-PerCP', 'CD14'],
            'CD15': ['CD15-APC', 'CD15-FITC', 'CD15'],
            'CD16': ['CD16-PE', 'CD16-APC-Cy7', 'CD16']
        }
        
        self.last_fcs = None
        
    def load_fcs(self, fcs_path: Path) -> pd.DataFrame:
        """Load FCS file with proper marker resolution"""
        self.last_fcs = fcs_path.stem
        
        try:
            sample = fk.Sample(str(fcs_path))
            if sample.compensation is not None:
                sample.apply_compensation(sample.compensation)
            sample.apply_transform('arcsinh', channels=None, cofactor=self.cofactor)
            df = sample.as_dataframe()
        except Exception as e:
            print(f"Fallback to fcsparser: {e}")
            meta, data = fcsparser.parse(str(fcs_path))
            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = np.arcsinh(df[numeric_cols] / self.cofactor)
        
        df = self._standardize_columns(df)
        self._log_marker_mapping(df, fcs_path)
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map channels to standard marker names"""
        new_columns = {}
        
        for col in df.columns:
            col_upper = col.upper()
            
            for marker, variants in self.marker_mappings.items():
                for variant in variants:
                    if variant.upper() in col_upper:
                        new_columns[col] = marker
                        break
                if col in new_columns:
                    break
            
            if col not in new_columns:
                if any(x in col_upper for x in ['FSC', 'SSC', 'TIME']):
                    new_columns[col] = col
        
        df = df.rename(columns=new_columns)
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _log_marker_mapping(self, df: pd.DataFrame, fcs_path: Path):
        """Log exact marker resolution"""
        used = {}
        for marker in self.marker_mappings.keys():
            hit = marker if marker in df.columns else None
            used[marker] = hit
        
        log_file = self.log_dir / f"{fcs_path.stem}_marker_map.json"
        with open(log_file, "w") as f:
            json.dump({
                'file': str(fcs_path),
                'timestamp': datetime.now().isoformat(),
                'markers_found': used,
                'all_columns': list(df.columns),
                'cofactor': self.cofactor
            }, f, indent=2)
    
    def gate_blast_population(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Define blast gate with threshold logging and degeneracy check"""
        n = len(df)
        gate = np.ones(n, dtype=bool)
        thresholds = {}
        gates_applied = []
        
        if 'CD34' in df.columns:
            threshold = np.nanpercentile(df['CD34'].dropna(), 75)
            thresholds['CD34_P75'] = float(threshold)
            gate &= df['CD34'].fillna(-np.inf) > threshold
            gates_applied.append('CD34+')
            print(f"  CD34+ gate: {gate.sum()}/{n} cells")
        
        if 'CD117' in df.columns:
            threshold = np.nanpercentile(df['CD117'].dropna(), 70)
            thresholds['CD117_P70'] = float(threshold)
            gate &= df['CD117'].fillna(-np.inf) > threshold
            gates_applied.append('CD117+')
            print(f"  CD34+CD117+ gate: {gate.sum()}/{n} cells")
        
        if 'CD38' in df.columns:
            low = np.nanpercentile(df['CD38'].dropna(), 20)
            high = np.nanpercentile(df['CD38'].dropna(), 60)
            thresholds['CD38_P20'] = float(low)
            thresholds['CD38_P60'] = float(high)
            gate &= (df['CD38'].fillna(-np.inf) > low) & (df['CD38'].fillna(np.inf) < high)
            gates_applied.append('CD38dim')
            print(f"  CD34+CD117+CD38dim gate: {gate.sum()}/{n} cells")
        
        if 'HLA-DR' in df.columns:
            threshold = np.nanpercentile(df['HLA-DR'].dropna(), 60)
            thresholds['HLA-DR_P60'] = float(threshold)
            gate &= df['HLA-DR'].fillna(-np.inf) > threshold
            gates_applied.append('HLA-DR+')
            print(f"  Full blast gate: {gate.sum()}/{n} cells")
        
        # Check for degenerate gates
        frac = float(gate.mean())
        if frac < 0.005 or frac > 0.995:
            print(f"  ⚠️ Blast gate is degenerate ({frac:.2%}); may need adjustment")
            thresholds['_degenerate'] = True
        
        print(f"✓ Blast population: {gate.sum()}/{n} cells ({100*frac:.1f}%)")
        
        # Save thresholds
        threshold_file = self.log_dir / f"{self.last_fcs}_blast_thresholds.json"
        with open(threshold_file, "w") as f:
            json.dump({
                'file': self.last_fcs,
                'timestamp': datetime.now().isoformat(),
                'thresholds': thresholds,
                'gates_applied': gates_applied,
                'gate_count': int(gate.sum()),
                'total_cells': int(n),
                'gate_percentage': float(frac * 100)
            }, f, indent=2)
        
        return gate, thresholds


class MorphoPhenotypicAnalyzer:
    """Extract and combine morphological and phenotypic features"""
    
    def __init__(self, 
                 model_path: str = 'models/last.ckpt',
                 cache_dir: Path = Path("outputs/phase2/feat_cache")):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: Path) -> nn.Module:
        """Load trained model for feature extraction"""
        from src.models import BD_S8_Model
        
        model = BD_S8_Model(num_classes=5).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        clean_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_dict[k[6:]] = v
            else:
                clean_dict[k] = v
        
        model.load_state_dict(clean_dict, strict=False)
        self.feature_extractor = model.encoder
        
        return model
    
    def extract_morphological_features(self, 
                                      image_paths: List[Path],
                                      batch_size: int = 32) -> np.ndarray:
        """Extract CNN features with collision-proof caching and mixed precision"""
        
        features_list = []
        batch_images = []
        to_save_paths = []
        
        with torch.no_grad():
            for path in image_paths:
                # Collision-proof cache keys
                rel = str(Path(path).resolve())
                hid = hashlib.sha1(rel.encode()).hexdigest()[:10]
                cache_path = self.cache_dir / f"{Path(path).stem}_{hid}.npy"
                
                if cache_path.exists():
                    features_list.append(np.load(cache_path))
                else:
                    img = self._load_and_preprocess_image(path)
                    batch_images.append(img)
                    to_save_paths.append(cache_path)
                    
                    if len(batch_images) == batch_size:
                        batch_tensor = torch.stack(batch_images).to(self.device)
                        
                        # Mixed precision for A100s
                        ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                               if torch.cuda.is_available() else contextlib.nullcontext())
                        with ctx:
                            features = self.feature_extractor(batch_tensor).float().cpu().numpy()
                        
                        for feat, save_path in zip(features, to_save_paths):
                            np.save(save_path, feat)
                        
                        features_list.extend(list(features))
                        batch_images = []
                        to_save_paths = []
            
            # Process remainder
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                       if torch.cuda.is_available() else contextlib.nullcontext())
                with ctx:
                    features = self.feature_extractor(batch_tensor).float().cpu().numpy()
                
                for feat, save_path in zip(features, to_save_paths):
                    np.save(save_path, feat)
                
                features_list.extend(list(features))
        
        return np.vstack(features_list) if features_list else np.array([])
    
    def _load_and_preprocess_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess consistent with training"""
        img = tifffile.imread(path)
        
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[0] <= 4:
            img = np.transpose(img[:3], (1, 2, 0))
        
        if img.shape[-1] != 3:
            h, w = img.shape[:2]
            new_img = np.zeros((h, w, 3), dtype=np.float32)
            new_img[..., :min(3, img.shape[-1])] = img[..., :min(3, img.shape[-1])]
            img = new_img
        
        img = cv2.resize(img.astype(np.float32), (512, 512))
        img = img.astype(np.float32)
        mx = float(img.max())
        if mx > 0:
            if mx > 255:
                img /= 65535.0
            elif mx > 1:
                img /= 255.0
        
        return torch.from_numpy(img).permute(2, 0, 1)


class Phase2Pipeline:
    """Main pipeline with all corrections"""
    
    def __init__(self,
                 image_dir: Path = Path('/scratch/project_2010376/BDS8/BDS8_data'),
                 fcs_dir: Path = Path('/scratch/project_2010376/BDS8/FCS_data'),
                 model_path: str = 'models/last.ckpt',
                 output_dir: Path = Path('outputs/phase2'),
                 config: Optional[Dict] = None):
        
        self.image_dir = Path(image_dir)
        self.fcs_dir = Path(fcs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Config with all knobs exposed
        self.config = config or {}
        self.cofactor = self.config.get('arcsinh_cofactor', 150)
        self.umap_neighbors = self.config.get('umap_neighbors', 30)
        self.umap_min_dist = self.config.get('umap_min_dist', 0.3)
        self.max_cells_per_sample = self.config.get('max_cells_per_sample', 1000)
        self.max_tsne_samples = self.config.get('max_tsne_samples', 20000)
        self.dbscan_eps = self.config.get('dbscan_eps', 1.5)
        
        self.fcs_processor = FCSProcessor(
            cofactor=self.cofactor,
            log_dir=self.output_dir / "logs"
        )
        self.morpho_analyzer = MorphoPhenotypicAnalyzer(
            model_path=model_path,
            cache_dir=self.output_dir / "feat_cache"
        )
    
    def process_sample(self, 
                      sample_id: str,
                      sample_data: Dict) -> Dict:
        """Process single sample with all corrections and safety checks"""
        
        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_id}")
        print(f"{'='*60}")
        
        results = {
            'sample_id': sample_id,
            'sample_type': sample_data['sample_type']
        }
        
        # 1. Load FCS
        print("\n📊 Loading FCS data...")
        fcs_df = self.fcs_processor.load_fcs(sample_data['fcs_path'])
        print(f"  Loaded {len(fcs_df)} cells with {len(fcs_df.columns)} markers")
        
        # 2. Apply gates
        print("\n🔬 Applying gates...")
        blast_gate, blast_thresholds = self.fcs_processor.gate_blast_population(fcs_df)
        
        results['gates'] = {'blast': blast_gate}
        results['thresholds'] = {'blast': blast_thresholds}
        
        # Handle no mapped markers gracefully
        marker_cols = [c for c in fcs_df.columns if c in self.fcs_processor.marker_mappings]
        if not marker_cols:
            print("⚠️ No mapped markers found; continuing with morphology only.")
            spec_summary_vec = np.zeros(1, dtype=np.float32)
        else:
            phenotypic_df = fcs_df[marker_cols].copy()
            spec_summary_vec = phenotypic_df.median(numeric_only=True).values
        
        print(f"  Phenotypic summary: {len(spec_summary_vec)} features (sample-level median)")
        
        # 4. Process images
        image_paths = sample_data['image_paths']
        if len(image_paths) > self.max_cells_per_sample:
            random.seed(SEED)
            image_paths = random.sample(image_paths, self.max_cells_per_sample)
        
        print(f"\n🖼️ Extracting features from {len(image_paths)} images...")
        morphological_features = self.morpho_analyzer.extract_morphological_features(
            image_paths, batch_size=32
        )
        
        # FIX 2: Handle zero usable images gracefully
        n_cells = len(morphological_features)
        if n_cells == 0:
            print("⚠️ No images found/usable after sampling; skipping sample.")
            # Still persist minimal summary
            self._append_summary_row({
                "sample_id": sample_id,
                "sample_type": results['sample_type'],
                "n_fcs_cells": int(len(fcs_df)),
                "n_images_used": 0,
                "run_timestamp": datetime.now().isoformat(),
                "umap_neighbors": self.umap_neighbors,
                "umap_min_dist": self.umap_min_dist,
                "dbscan_eps": self.dbscan_eps,
                "tsne_is_subset": False,
                "tsne_n": 0
            })
            return results
        
        # 5. Broadcast summary to image count
        phenotypic_features = np.repeat(spec_summary_vec[None, :], n_cells, axis=0)
        
        # Gate fractions (sample-level)
        gate_fractions = {}
        for gate_name, gate_mask in results['gates'].items():
            frac = float(gate_mask.mean())
            if frac < 0.005 or frac > 0.995:
                print(f"  ⚠️ Gate '{gate_name}' is degenerate ({frac:.2%}); excluding from summaries.")
            else:
                gate_fractions[gate_name] = frac
        
        results['gate_fractions'] = gate_fractions
        print(f"\n📊 Gate fractions (sample-level): {gate_fractions}")
        
        # 6. Combine features
        print("\n🎯 Combining morphological and phenotypic features...")
        from sklearn.preprocessing import StandardScaler
        
        scaler_morph = StandardScaler()
        morph_norm = scaler_morph.fit_transform(morphological_features)
        
        scaler_phen = StandardScaler()
        phen_norm = scaler_phen.fit_transform(phenotypic_features)
        
        alpha = 0.6
        combined_features = np.hstack([
            alpha * morph_norm ,
            (1 - alpha) * phen_norm])
        
        print(f"  Combined shape: {combined_features.shape} "
            f"(morph {morph_norm.shape[1]} + phen {phen_norm.shape[1]})")
        
        results['features'] = {
            'morphological': morphological_features,
            'phenotypic_summary': spec_summary_vec,
            'combined': combined_features
        }
        
        # 7. Create embeddings with fail-soft
        print("\n📈 Creating embeddings...")
        
        # UMAP (full data)
        umap_embedding = self._create_embedding_safe(
            combined_features,
            method='umap'
        )
        
        # t-SNE (with safety cap)
        tsne_input = combined_features
        tsne_is_subset = False
        if len(tsne_input) > self.max_tsne_samples:
            idx = np.random.RandomState(SEED).choice(
                len(tsne_input), self.max_tsne_samples, replace=False
            )
            tsne_input = tsne_input[idx]
            tsne_is_subset = True
        
        tsne_embedding = self._create_embedding_safe(
            tsne_input,
            method='tsne'
        )
        
        results['embeddings'] = {
            'umap': umap_embedding,
            'tsne': tsne_embedding
        }
        
        results['embeddings_meta'] = {
            'tsne_is_subset': tsne_is_subset,
            'tsne_n': int(len(tsne_input))
        }
        
        # 8. Identify subpopulations
        labels = np.ones(n_cells) if sample_data['sample_type'] == 'AML' else np.zeros(n_cells)
        
        subpop_results = self._identify_subpopulations(
            umap_embedding,
            combined_features,
            labels=labels
        )
        
        results['subpopulations'] = subpop_results
        
        # 9. Save results
        self._save_sample_results(results, fcs_df)
        
        # Slack-friendly one-liner
        print(f"✓ {sample_id} ({sample_data['sample_type']}): "
              f"images={n_cells}, FCS={len(fcs_df)}, "
              f"blast={results['gate_fractions'].get('blast', 0.0):.1%}")
        
        return results
    
    def _create_embedding_safe(self, features: np.ndarray, method: str = 'umap') -> np.ndarray:
        """FIX 3: Create embedding with fail-soft on memory issues"""
        try:
            if method == 'umap':
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=self.umap_neighbors,
                    min_dist=self.umap_min_dist,
                    metric='euclidean',
                    random_state=SEED
                )
            elif method == 'tsne':
                reducer = TSNE(
                    n_components=2,
                    perplexity=30,
                    early_exaggeration=12,
                    random_state=SEED
                )
            else:  # PCA
                reducer = PCA(n_components=2, random_state=SEED)
            
            return reducer.fit_transform(features)
            
        except Exception as e:
            print(f"⚠️ {method.upper()} failed ({e}); falling back to PCA.")
            reducer = PCA(n_components=2, random_state=SEED)
            return reducer.fit_transform(features)
    
    def _identify_subpopulations(self, 
                                embedding: np.ndarray,
                                features: np.ndarray,
                                labels: np.ndarray) -> Dict:
        """Identify subpopulations with improved clustering"""
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score
        
        results = {}
        
        # Adaptive K-means
        if len(embedding) > 100:
            k_values = range(3, min(20, max(4, len(embedding)//50)))
            scores = []
            
            for k in k_values:
                km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
                clusters = km.fit_predict(embedding)
                score = silhouette_score(embedding, clusters)
                scores.append(score)
            
            if scores:
                best_k = list(k_values)[np.argmax(scores)]
                kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
                results['kmeans_clusters'] = kmeans.fit_predict(embedding)
                print(f"  K-means: k={best_k}, silhouette={max(scores):.3f}")
        
        # Adaptive DBSCAN
        min_samples = max(10, int(sqrt(len(embedding))))
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=min_samples)
        results['dbscan_clusters'] = dbscan.fit_predict(embedding)
        n_clusters = len(set(results['dbscan_clusters'])) - (1 if -1 in results['dbscan_clusters'] else 0)
        print(f"  DBSCAN: {n_clusters} clusters (eps={self.dbscan_eps}, min_samples={min_samples})")
        
        # GMM
        n_components = min(8, max(2, len(embedding)//100))
        if n_components > 1:
            gmm = GaussianMixture(n_components=n_components, 
                                 covariance_type='full',
                                 random_state=SEED)
            results['gmm_clusters'] = gmm.fit_predict(embedding)
            print(f"  GMM: {n_components} components")
        
        return results
    
    def _save_sample_results(self, results: Dict, fcs_df: pd.DataFrame):
        """Save comprehensive results with all formats"""
        
        sample_id = results['sample_id']
        
        # Save embeddings explicitly
        np.save(self.output_dir / f"{sample_id}_umap.npy", 
                results['embeddings']['umap'])
        np.save(self.output_dir / f"{sample_id}_tsne.npy", 
                results['embeddings']['tsne'])
        
        # Save compact JSON
        json_path = self.output_dir / f"{sample_id}_summary.json"
        summary_dict = {
            "sample_id": sample_id,
            "type": results['sample_type'],
            "gate_fractions": results['gate_fractions'],
            "thresholds": results['thresholds'],
            "embeddings_meta": results['embeddings_meta']
        }
        with open(json_path, "w") as f:
            json.dump(summary_dict, f, indent=2)
        
        # FIX 4: Save YAML for easier review
        yaml_path = self.output_dir / f"{sample_id}_summary.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(summary_dict, f, sort_keys=False)
        
        # Enhanced CSV with run metadata
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
        
        # Save full results
        np.save(self.output_dir / f"{sample_id}_results.npy", 
                results, allow_pickle=True)
    
    def _append_summary_row(self, row: dict):
        """Append row to CSV summary"""
        csv_path = self.output_dir / "sample_summaries.csv"
        
        write_header = not csv_path.exists()
        
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)
    
    def visualize_results(self, results: Dict):
        """Create visualizations with proper cleanup"""
        
        sample_id = results['sample_id']
        sample_type = results['sample_type']
        
        fig = plt.figure(figsize=(16, 10))
        
        # UMAP colored by sample type
        ax1 = plt.subplot(2, 3, 1)
        embedding = results['embeddings']['umap']
        colors = ['red' if sample_type == 'AML' else 'blue'] * len(embedding)
        ax1.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.5, s=1)
        ax1.set_title(f'UMAP - {sample_type}')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # K-means clusters
        ax2 = plt.subplot(2, 3, 2)
        if 'kmeans_clusters' in results['subpopulations']:
            clusters = results['subpopulations']['kmeans_clusters']
            ax2.scatter(embedding[:, 0], embedding[:, 1], 
                       c=clusters, cmap='tab20', alpha=0.5, s=1)
            ax2.set_title('K-means Clusters')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        
        # DBSCAN
        ax3 = plt.subplot(2, 3, 3)
        if 'dbscan_clusters' in results['subpopulations']:
            clusters = results['subpopulations']['dbscan_clusters']
            ax3.scatter(embedding[:, 0], embedding[:, 1], 
                       c=clusters, cmap='tab20', alpha=0.5, s=1)
            ax3.set_title(f'DBSCAN (eps={self.dbscan_eps})')
        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
        
        # t-SNE (with subset info)
        ax4 = plt.subplot(2, 3, 4)
        tsne_embedding = results['embeddings']['tsne']
        meta = results['embeddings_meta']
        title = f't-SNE ({meta["tsne_n"]} points'
        if meta['tsne_is_subset']:
            title += f', subset of {len(embedding)})'
        else:
            title += ')'
        ax4.set_title(title)
        ax4.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                   c='gray', alpha=0.5, s=1)
        ax4.set_xlabel('t-SNE 1')
        ax4.set_ylabel('t-SNE 2')
        
        # Density
        ax5 = plt.subplot(2, 3, 5)
        from scipy.stats import gaussian_kde
        try:
            xy = embedding.T
            z = gaussian_kde(xy)(xy)
            ax5.scatter(embedding[:, 0], embedding[:, 1], c=z, cmap='hot', s=1)
            ax5.set_title('Cell Density')
        except:
            ax5.set_title('Density (skipped)')
        ax5.set_xlabel('UMAP 1')
        ax5.set_ylabel('UMAP 2')
        
        # Gate fractions
        ax6 = plt.subplot(2, 3, 6)
        gate_fractions = results['gate_fractions']
        if gate_fractions:
            gates = list(gate_fractions.keys())
            fractions = list(gate_fractions.values())
            y_pos = np.arange(len(gates))
            bars = ax6.barh(y_pos, fractions)
            
            for i, (bar, frac) in enumerate(zip(bars, fractions)):
                if frac > 0.5:
                    bar.set_color('red')
                elif frac > 0.2:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(gates)
            ax6.set_xlabel('Fraction')
            ax6.set_title('Gate Fractions (Sample-level)')
            ax6.set_xlim([0, 1])
        
        plt.suptitle(f'Sample {sample_id} - Morpho-Phenotypic Analysis', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / f'{sample_id}_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
        
        # Clean up
        plt.close(fig)
        
        return save_path
    
    def match_samples(self) -> Dict[str, Dict]:
        """Match each FCS to images by filename prefix (robust to '_images_...' dirs)."""
        matches = {}
        fcs_files = list(self.fcs_dir.rglob('*.fcs'))

        print(f"\n📁 Found {len(fcs_files)} FCS files")
        print(f"Searching images under: {self.image_dir}")

        for fcs_path in fcs_files:
            sample_id = fcs_path.stem  # <-- keep FULL stem
            image_paths: List[Path] = []

            # 1) Common case: a directory like '<stem>_images_<timestamp>/'
            for d in self.image_dir.rglob(f"{sample_id}_images_*"):
                if d.is_dir():
                    image_paths += list(d.rglob("*.tif"))
                    image_paths += list(d.rglob("*.tiff"))

            # 2) Fallback: files whose name starts with the stem anywhere under image_dir
            if not image_paths:
                image_paths += list(self.image_dir.rglob(f"{sample_id}_*.tif"))
                image_paths += list(self.image_dir.rglob(f"{sample_id}_*.tiff"))

            # 3) Extra fallback: case/spacing-insensitive match on filename stems
            if not image_paths:
                def norm(s: str) -> str:
                    return "".join(ch.lower() for ch in s if ch.isalnum())
                nstem = norm(sample_id)
                for tif in self.image_dir.rglob("*.tif"):
                    if norm(tif.stem).startswith(nstem):
                        image_paths.append(tif)
                for tif in self.image_dir.rglob("*.tiff"):
                    if norm(tif.stem).startswith(nstem):
                        image_paths.append(tif)

            if image_paths:
                # Decide type from folder the FCS lives in
                image_paths = sorted(set(image_paths))
                
                pstr = str(fcs_path)
                if "/AML/" in pstr:
                    sample_type = "AML"
                elif "/Healthy BM/" in pstr:
                    sample_type = "Healthy"
                else:
                    # leave as Healthy vs AML by heuristic or Unknown
                    sample_type = (
                    "Healthy" if "BM" in pstr else "AML" if "AML" in pstr else "Unknown"
                )

                matches[sample_id] = {
                    "fcs_path": fcs_path,
                    "image_paths": image_paths,
                    "sample_type": sample_type,
                }
                print(f"  ✓ {sample_id}: {len(image_paths)} images, {sample_type}")
            else:
                print(f"  ✗ {sample_id}: no images found")

        return matches

    
    
    def run_analysis(self, max_samples: int = 5):
        """Run complete Phase 2 analysis"""
        
        print("\n" + "="*60)
        print("🚀 PHASE 2: Morpho-Phenotypic Analysis")
        print("    Building on Phase 1's 79% accurate model")
        print("="*60)
        
        matched_samples = self.match_samples()
        
        if not matched_samples:
            print("❌ No matched samples found!")
            return
        
        all_results = []
        
        for i, (sample_id, sample_data) in enumerate(matched_samples.items()):
            if i >= max_samples:
                break
            
            try:
                results = self.process_sample(sample_id, sample_data)
                all_results.append(results)
                self.visualize_results(results)
                
            except Exception as e:
                print(f"❌ Error processing {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # FIX 1: Print summary with safe column access
        csv_path = self.output_dir / "sample_summaries.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            cols = ['sample_id', 'sample_type', 'n_images_used']
            if 'gate_blast' in df.columns:
                cols.append('gate_blast')
            print("\n📊 Summary of processed samples:")
            print(df[cols].to_string(index=False))
        
        print("\n✅ Phase 2 analysis complete!")
        return all_results


# Main execution
if __name__ == "__main__":
    # Configuration with all knobs exposed
    config = {
        'arcsinh_cofactor': 150,
        'umap_neighbors': 30,
        'umap_min_dist': 0.3,
        'max_cells_per_sample': 1000,
        'max_tsne_samples': 20000,
        'dbscan_eps': 1.5,
    }
    
    pipeline = Phase2Pipeline(
        image_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        fcs_dir=Path('/scratch/project_2010376/BDS8/BDS8_data'),
        model_path='models/last.ckpt',  # Your 79% accurate Phase 1 model
        output_dir=Path('outputs/phase2'),
        config=config
    )
    
    results = pipeline.run_analysis(max_samples=10)
    
    print("\n✅ Phase 2 complete! Check outputs/phase2/ for:")
    print("  - sample_summaries.csv (aggregate metrics)")
    print("  - *_summary.json/yaml (per-sample metadata)")
    print("  - *_umap.npy, *_tsne.npy (embeddings)")
    print("  - *_analysis.png (visualizations)")
    print("  - logs/ (marker mappings and thresholds)")
    print("  - feat_cache/ (cached CNN features)")

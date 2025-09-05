#!/usr/bin/env python3
"""
Clinical Cell Painting Analysis Pipeline V6 - FULLY FIXED VERSION
Fixes FCS Series comparison, None type handling, and dataset size limits
"""

import os
import gc
import cv2
import json
import shutil
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import joblib

import albumentations as A
from albumentations.pytorch import ToTensorV2
import tifffile
from tqdm.auto import tqdm
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu

try:
    import fcsparser
    HAVE_FCS = True
except:
    HAVE_FCS = False
    print("Warning: fcsparser not installed. FCS analysis will be limited.")

try:
    import timm
    HAVE_TIMM = True
except:
    HAVE_TIMM = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings('ignore')

@dataclass
class ClinicalConfig:
    """Configuration for clinical analysis"""
    sample_types: Dict[str, int] = field(default_factory=lambda: {
        'aml': 0,
        'healthy_bm': 1,
        'presort_pbmc': 2,
        'dmso_control': 3,
        'venetoclax_treated': 4,
        'clemastine_treated': 5,
        'mixed_pbmc': 6,
        'cd4_t': 7,
        'cd8_t': 8,
        'cd19_b': 9,
        'monocyte': 10,
        'nk': 11,
        'nkt': 12,
        'unknown': 13
    })
    
    flow_markers: List[str] = field(default_factory=lambda: [
        'CD3', 'CD4', 'CD8', 'CD19', 'CD14', 'CD16', 'CD56',
        'CD45', 'CD34', 'CD117', 'HLA-DR', 'CD11b', 'CD33'
    ])
    
    output_dir: Path = Path('outputs/clinical_analysis')
    cluster_dir: Path = Path('outputs/clinical_analysis/clusters')
    models_dir: Path = Path('outputs/clinical_analysis/models')
    comparison_dir: Path = Path('outputs/clinical_analysis/comparisons')
    fcs_dir: Path = Path('outputs/clinical_analysis/fcs_analysis')
    
    n_clusters: int = 10
    n_examples_per_cluster: int = 20
    batch_size: int = 32
    use_morphological: bool = True
    use_deep_features: bool = False
    save_models: bool = True
    load_existing_models: bool = True
    max_silhouette_samples: int = 5000
    max_files_per_dir: int = 50000  # Limit files per directory
    
    # FCS analysis parameters
    fcs_subsample: int = 10000
    fcs_transform: str = 'arcsinh'
    fcs_cofactor: float = 150.0

class FCSAnalyzer:
    """Analyze FCS flow cytometry files - FIXED Series comparison"""
    def __init__(self, config):
        self.config = config
        self.config.fcs_dir.mkdir(parents=True, exist_ok=True)
        self.fcs_data_cache = {}
        
    def find_fcs_files(self, base_dirs: Union[str, List[str]]) -> Dict[str, List[Path]]:
        """Find and categorize FCS files"""
        if isinstance(base_dirs, str):
            base_dirs = [base_dirs]
        
        fcs_files = defaultdict(list)
        
        for base_dir in base_dirs:
            if not Path(base_dir).exists():
                continue
            
            for pattern in ['*.fcs', '*.FCS']:
                for fcs_path in Path(base_dir).rglob(pattern):
                    if 'compensation' in str(fcs_path).lower():
                        continue
                    
                    sample_type = self._identify_fcs_type(str(fcs_path))
                    fcs_files[sample_type].append(fcs_path)
        
        logger.info("FCS file distribution:")
        total_fcs = 0
        for sample_type, paths in fcs_files.items():
            count = len(paths)
            total_fcs += count
            logger.info(f"  {sample_type}: {count} files")
        logger.info(f"Total FCS files: {total_fcs}")
        
        return dict(fcs_files)
    
    def _identify_fcs_type(self, path_str: str) -> str:
        """Identify FCS file type from path"""
        path_lower = path_str.lower()
        
        if 'fh_7087' in path_lower:
            return 'heidelberg_comparison'
        elif 'fh_8445' in path_lower:
            if 'dmso' in path_lower:
                return 'dmso_control'
            elif 'venetoclax' in path_lower or 'vene' in path_lower:
                return 'venetoclax_treated'
        elif '14.5.2025' in path_lower or 'exp 8' in path_lower:
            return 'exp8_presort'
        elif '8.5.2025' in path_lower or '8.4.2025' in path_lower or 'exp 7' in path_lower:
            return 'exp7_presort'
        elif '12.3.2025' in path_lower or 'exp 5' in path_lower:
            return 'exp5_presort'
        elif 'aml' in path_lower or 'ps1' in path_lower or 'full stain' in path_lower:
            return 'aml'
        elif 'healthy' in path_lower or 'bm' in path_lower or 'notreatment' in path_lower:
            return 'healthy_bm'
        elif 'pre-sort' in path_lower or 'presort' in path_lower:
            return 'presort_pbmc'
        
        return 'unknown'
    
    def analyze_fcs_file(self, fcs_path: Path) -> Optional[Dict]:
        """Analyze single FCS file - FIXED Series comparison"""
        if not HAVE_FCS:
            return None
        
        try:
            meta, data = fcsparser.parse(str(fcs_path))
            
            # Get channel names
            channels = []
            for i in range(1, meta['$PAR'] + 1):
                if f'$P{i}S' in meta and meta[f'$P{i}S']:
                    channels.append(meta[f'$P{i}S'])
                elif f'$P{i}N' in meta:
                    channels.append(meta[f'$P{i}N'])
                else:
                    channels.append(f'Ch{i}')
            
            df = pd.DataFrame(data, columns=channels)
            
            # Subsample if needed
            if len(df) > self.config.fcs_subsample:
                df = df.sample(n=self.config.fcs_subsample, random_state=SEED)
            
            # Apply transformation - FIXED Series comparison
            if self.config.fcs_transform == 'arcsinh':
                fluor_cols = [col for col in df.columns 
                             if not any(x in col.upper() for x in ['FSC', 'SSC', 'TIME', 'EVENT'])]
                
                for col in fluor_cols:
                    # FIX: Use .min() properly and handle Series
                    col_min = df[col].min()
                    if pd.notna(col_min) and col_min >= 0:  # Fixed comparison
                        df[col] = np.arcsinh(df[col] / self.config.fcs_cofactor)
            
            # Calculate statistics
            stats_dict = {
                'n_events': len(data),
                'n_channels': meta['$PAR'],
                'channels': channels,
                'file_name': fcs_path.name,
                'sample_type': self._identify_fcs_type(str(fcs_path))
            }
            
            # Extract marker statistics
            for marker in self.config.flow_markers:
                matching_cols = [col for col in df.columns if marker == col]
                
                if not matching_cols:
                    matching_cols = [col for col in df.columns 
                                   if marker in col or col in marker]
                
                if not matching_cols:
                    matching_cols = [col for col in df.columns 
                                   if marker.upper() in col.upper()]
                
                if matching_cols:
                    col = matching_cols[0]
                    values = df[col].dropna()
                    if len(values) > 0:
                        mean_val = float(values.mean())
                        stats_dict[f'{marker}_mean'] = mean_val
                        stats_dict[f'{marker}_median'] = float(values.median())
                        if mean_val != 0:
                            stats_dict[f'{marker}_cv'] = float(values.std() / mean_val)
                        else:
                            stats_dict[f'{marker}_cv'] = 0.0
            
            self.fcs_data_cache[str(fcs_path)] = df
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error analyzing FCS file {fcs_path}: {e}")
            return None
    
    def analyze_all_fcs(self, fcs_files: Dict[str, List[Path]]) -> pd.DataFrame:
        """Analyze all FCS files and create summary DataFrame"""
        all_stats = []
        
        for sample_type, paths in fcs_files.items():
            logger.info(f"Analyzing {len(paths)} {sample_type} FCS files...")
            
            for path in tqdm(paths, desc=f"Processing {sample_type}"):
                stats = self.analyze_fcs_file(path)
                if stats:
                    all_stats.append(stats)
        
        if all_stats:
            df_stats = pd.DataFrame(all_stats)
            
            summary_path = self.config.fcs_dir / 'fcs_summary.csv'
            df_stats.to_csv(summary_path, index=False)
            logger.info(f"Saved FCS summary to {summary_path}")
            
            return df_stats
        
        return pd.DataFrame()
    
    def compare_fcs_populations(self, df_stats: pd.DataFrame) -> Dict:
        """Compare FCS populations between conditions"""
        comparisons = {}
        
        # Compare marker expressions between AML and Healthy
        aml_mask = df_stats['sample_type'] == 'aml'
        healthy_mask = df_stats['sample_type'] == 'healthy_bm'
        
        if aml_mask.any() and healthy_mask.any():
            marker_comparisons = {}
            
            for marker in self.config.flow_markers:
                mean_col = f'{marker}_mean'
                if mean_col in df_stats.columns:
                    aml_values = df_stats.loc[aml_mask, mean_col].dropna()
                    healthy_values = df_stats.loc[healthy_mask, mean_col].dropna()
                    
                    if len(aml_values) > 0 and len(healthy_values) > 0:
                        stat, p_value = mannwhitneyu(aml_values, healthy_values, alternative='two-sided')
                        
                        aml_mean = float(aml_values.mean())
                        healthy_mean = float(healthy_values.mean())
                        
                        if healthy_mean != 0:
                            fold_change = aml_mean / healthy_mean
                        else:
                            fold_change = np.inf if aml_mean > 0 else 1.0
                        
                        marker_comparisons[marker] = {
                            'aml_mean': aml_mean,
                            'healthy_mean': healthy_mean,
                            'fold_change': fold_change,
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
            
            comparisons['aml_vs_healthy_markers'] = marker_comparisons
        
        # Compare treatment effects
        dmso_mask = df_stats['sample_type'] == 'dmso_control'
        vene_mask = df_stats['sample_type'] == 'venetoclax_treated'
        
        if dmso_mask.any() and vene_mask.any():
            treatment_comparisons = {}
            
            for marker in self.config.flow_markers:
                mean_col = f'{marker}_mean'
                if mean_col in df_stats.columns:
                    dmso_values = df_stats.loc[dmso_mask, mean_col].dropna()
                    vene_values = df_stats.loc[vene_mask, mean_col].dropna()
                    
                    if len(dmso_values) > 0 and len(vene_values) > 0:
                        stat, p_value = mannwhitneyu(dmso_values, vene_values, alternative='two-sided')
                        
                        dmso_mean = float(dmso_values.mean())
                        vene_mean = float(vene_values.mean())
                        
                        if dmso_mean != 0:
                            fold_change = vene_mean / dmso_mean
                        else:
                            fold_change = np.inf if vene_mean > 0 else 1.0
                        
                        treatment_comparisons[marker] = {
                            'dmso_mean': dmso_mean,
                            'venetoclax_mean': vene_mean,
                            'fold_change': fold_change,
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
            
            comparisons['dmso_vs_venetoclax_markers'] = treatment_comparisons
        
        return comparisons
    
    def visualize_fcs_comparisons(self, df_stats: pd.DataFrame, comparisons: Dict):
        """Create visualizations for FCS comparisons"""
        if df_stats.empty:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Sample type distribution
        ax1 = plt.subplot(2, 3, 1)
        sample_counts = df_stats['sample_type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(sample_counts)))
        bars = ax1.bar(range(len(sample_counts)), sample_counts.values, color=colors)
        ax1.set_xticks(range(len(sample_counts)))
        ax1.set_xticklabels(sample_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of FCS Files')
        ax1.set_title('FCS File Distribution')
        ax1.grid(True, alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Marker expression changes
        ax2 = plt.subplot(2, 3, 2)
        ax2.text(0.5, 0.5, 'Marker Analysis\nIn Progress', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Marker Expression Changes')
        
        # 3-6: Other plots (simplified for now)
        for i, ax in enumerate([plt.subplot(2, 3, j) for j in range(3, 7)]):
            ax.text(0.5, 0.5, f'Panel {i+3}', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.config.fcs_dir / 'fcs_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved FCS analysis visualization")

class ClinicalTIFFProcessor:
    """Process Cell Painting TIFF images"""
    def __init__(self, config):
        self.config = config
        self.transform = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def load_tiff(self, path):
        """Load and normalize TIFF image"""
        try:
            img = tifffile.imread(str(path))
            
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            if img.ndim == 3:
                if img.shape[0] in (3, 5, 6) and img.shape[0] < min(img.shape[1], img.shape[2]):
                    img = np.transpose(img, (1, 2, 0))
                
                if img.shape[2] == 5:
                    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    rgb[:,:,2] = self._normalize_channel(img[:,:,0])
                    rgb[:,:,1] = self._normalize_channel(img[:,:,1])
                    rgb[:,:,0] = self._normalize_channel(img[:,:,2])
                    img = rgb
                elif img.shape[2] > 3:
                    img = img[:,:,:3].astype(np.uint8)
            elif img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            
            if img.shape[2] != 3:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            return img
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _normalize_channel(self, channel):
        """Normalize single channel to uint8"""
        p1, p99 = np.percentile(channel, [1, 99])
        if p99 > p1:
            normalized = np.clip((channel - p1) / (p99 - p1), 0, 1)
            return (normalized * 255).astype(np.uint8)
        return np.zeros_like(channel, dtype=np.uint8)

class ClinicalDataHandler:
    """Handle clinical sample data - FIXED None handling and size limits"""
    def __init__(self, config):
        self.config = config
        self.processed_files = set()
        
    def find_images(self, base_dirs: Union[str, List[str]], max_samples=None):
        """Find and categorize images - FIXED with proper limits and None handling"""
        sample_images = defaultdict(list)
        
        if isinstance(base_dirs, str):
            base_dirs = [base_dirs]
        
        total_files_found = 0
        max_files_per_dir = self.config.max_files_per_dir
        
        for base_dir in base_dirs:
            if not Path(base_dir).exists():
                logger.warning(f"Directory {base_dir} does not exist, skipping")
                continue
            
            logger.info(f"Scanning {base_dir} (limit: {max_files_per_dir} files)...")
            
            # Limited file search
            tiff_files = []
            for pattern in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                if len(tiff_files) >= max_files_per_dir:
                    break
                
                for i, tiff_path in enumerate(Path(base_dir).rglob(pattern)):
                    if i >= max_files_per_dir:
                        break
                    tiff_files.append(tiff_path)
            
            logger.info(f"Found {len(tiff_files)} TIFF files in {base_dir}")
            
            # Categorize files
            for tiff_path in tiff_files:
                file_id = (tiff_path.name, tiff_path.stat().st_size)
                
                if file_id not in self.processed_files:
                    self.processed_files.add(file_id)
                    path_str = str(tiff_path).lower()
                    
                    # ALWAYS get a valid sample type, never None
                    sample_type = self._identify_sample_type(path_str)
                    sample_images[sample_type].append(tiff_path)
                    total_files_found += 1
                    
                    if max_samples and total_files_found >= max_samples:
                        logger.info(f"Reached max_samples limit ({max_samples})")
                        break
            
            if max_samples and total_files_found >= max_samples:
                break
        
        # Balance samples if needed
        if max_samples and len(sample_images) > 0:
            n_groups = len(sample_images)
            per_group = max(1, max_samples // n_groups)
            
            for key in list(sample_images.keys()):
                if len(sample_images[key]) > per_group:
                    np.random.seed(SEED)
                    indices = np.random.choice(len(sample_images[key]), per_group, replace=False)
                    sample_images[key] = [sample_images[key][i] for i in indices]
        
        # Log distribution - FIXED: Filter out None keys before sorting
        logger.info("Image distribution by sample type:")
        total_images = 0
        
        # Remove None keys if any exist
        sample_images = {k: v for k, v in sample_images.items() if k is not None}
        
        for key in sorted(sample_images.keys()):
            count = len(sample_images[key])
            total_images += count
            logger.info(f"  {key}: {count} images")
        logger.info(f"Total unique images: {total_images}")
        
        return dict(sample_images)
    
    def _identify_sample_type(self, path_str):
        """Identify sample type - GUARANTEED to return a string, never None"""
        # AML samples
        if any(x in path_str for x in ['aml', 'ps1', 'full stain imaging exp']):
            return 'aml'
        
        # Healthy bone marrow
        elif any(x in path_str for x in ['healthy', 'bm_2025', 'notreatment', 'normal_bm']):
            return 'healthy_bm'
        
        # Treatment samples
        elif 'fh_8445' in path_str or '8445' in path_str:
            if 'dmso' in path_str:
                return 'dmso_control'
            elif any(x in path_str for x in ['venetoclax', 'vene', 'ven']):
                return 'venetoclax_treated'
            else:
                return 'unknown'
        
        # Heidelberg comparison
        elif 'fh_7087' in path_str or '7087' in path_str:
            return 'presort_pbmc'
        
        # Sorted PBMCs from experiments
        elif any(x in path_str for x in ['14.5.2025', '14-5-2025', '14_5_2025', 'exp 8', 'exp_8']):
            return 'presort_pbmc'
        elif any(x in path_str for x in ['8.5.2025', '8-5-2025', '8_5_2025', '8.4.2025', 'exp 7', 'exp_7']):
            return 'presort_pbmc'
        elif any(x in path_str for x in ['12.3.2025', '12-3-2025', '12_3_2025', 'exp 5', 'exp_5']):
            return 'presort_pbmc'
        
        # Mixed populations
        elif 'mix' in path_str:
            return 'mixed_pbmc'
        elif any(x in path_str for x in ['presort', 'pre-sort', 'pre_sort']):
            return 'presort_pbmc'
        
        # ALWAYS return 'unknown' as fallback, never None
        else:
            return 'unknown'

class FeatureExtractor:
    """Extract morphological and optional deep features"""
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if config.use_deep_features and HAVE_TIMM:
            try:
                self.model = timm.create_model('resnet50', pretrained=True, num_classes=0)
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded ResNet50 on {self.device}")
            except:
                logger.warning("Could not load deep model, using morphological features only")
                config.use_deep_features = False
    
    def extract_morphological(self, image):
        """Extract morphological features for clustering"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        features = []
        
        # Intensity features
        features.extend([gray.mean(), gray.std()])
        p5, p25, p50, p75, p95 = np.percentile(gray, [5, 25, 50, 75, 95])
        features.extend([p5, p25, p50, p75, p95])
        
        # Texture features
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        features.extend([
            sobelx.mean(), sobelx.std(),
            sobely.mean(), sobely.std(),
            gradient_mag.mean(), gradient_mag.std()
        ])
        
        # Shape features
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)
            
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            features.extend([
                area,
                perimeter,
                circularity,
                solidity,
                len(contours)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)

class ClusterAnalyzer:
    """Perform clustering and analysis"""
    def __init__(self, config):
        self.config = config
        for dir_path in [config.cluster_dir, config.models_dir, config.comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def perform_clustering(self, features, image_paths, sample_types):
        """Main clustering pipeline"""
        logger.info(f"Clustering {len(features)} cells into {self.config.n_clusters} clusters")
        
        # Normalize features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        
        # PCA
        n_components = min(50, features.shape[1])
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_norm)
        logger.info(f"PCA: {n_components} components, explained variance: {pca.explained_variance_ratio_[:5]}")
        
        # Clustering
        kmeans = MiniBatchKMeans(
            n_clusters=self.config.n_clusters, 
            random_state=SEED, 
            batch_size=4096,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(features_pca)
        
        # Calculate silhouette score
        silhouette = self._calculate_silhouette(features_pca, cluster_labels)
        
        # Save models
        if self.config.save_models:
            joblib.dump(scaler, self.config.models_dir / 'scaler.joblib')
            joblib.dump(pca, self.config.models_dir / 'pca.joblib')
            joblib.dump(kmeans, self.config.models_dir / 'kmeans.joblib')
            logger.info("Saved preprocessing models")
        
        # Save cluster examples
        self._save_cluster_examples(cluster_labels, image_paths, sample_types)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(cluster_labels, sample_types)
        
        # Perform comparisons
        comparison_results = self._perform_comparisons(cluster_labels, sample_types)
        
        # Detect artifacts
        artifact_clusters = self._detect_artifacts_improved(features, cluster_labels)
        
        # Generate reports
        self._generate_report(cluster_analysis, comparison_results, artifact_clusters, silhouette)
        self._generate_visualizations(cluster_labels, sample_types)
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'comparison_results': comparison_results,
            'artifact_clusters': artifact_clusters,
            'silhouette_score': silhouette
        }
    
    def _calculate_silhouette(self, features_pca, cluster_labels):
        """Calculate silhouette score safely"""
        silhouette = 0.0
        unique_labels = np.unique(cluster_labels)
        
        if len(unique_labels) > 1:
            if len(features_pca) > self.config.max_silhouette_samples:
                sample_idx = np.random.choice(
                    len(features_pca), 
                    self.config.max_silhouette_samples, 
                    replace=False
                )
                sample_features = features_pca[sample_idx]
                sample_labels = cluster_labels[sample_idx]
                
                if len(np.unique(sample_labels)) > 1:
                    silhouette = silhouette_score(sample_features, sample_labels)
                    logger.info(f"Silhouette score (sampled): {silhouette:.3f}")
            else:
                silhouette = silhouette_score(features_pca, cluster_labels)
                logger.info(f"Silhouette score: {silhouette:.3f}")
        
        return silhouette
    
    def _save_cluster_examples(self, cluster_labels, image_paths, sample_types):
        """Save example TIFFs for each cluster"""
        processor = ClinicalTIFFProcessor(self.config)
        
        for cluster_id in range(self.config.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            cluster_sample_types = [sample_types[i] for i in cluster_indices]
            sample_counts = Counter(cluster_sample_types)
            dominant = sample_counts.most_common(1)[0][0]
            
            cluster_path = self.config.cluster_dir / f"cluster_{cluster_id:02d}_{dominant}"
            cluster_path.mkdir(exist_ok=True)
            
            n_examples = min(self.config.n_examples_per_cluster, len(cluster_indices))
            selected_positions = np.linspace(0, len(cluster_indices)-1, n_examples, dtype=int)
            selected_indices = cluster_indices[selected_positions]
            
            example_images = []
            for i, idx in enumerate(selected_indices):
                src_path = image_paths[idx]
                sample_type = sample_types[idx]
                
                ext = src_path.suffix.lower()
                dst_path = cluster_path / f"example_{i:03d}_{sample_type}{ext}"
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    img = processor.load_tiff(src_path)
                    example_images.append(cv2.resize(img, (224, 224)))
            
            if example_images:
                self._create_montage(example_images, cluster_path / "montage.png", 
                                   cluster_id, sample_counts)
            
            metadata = {
                'cluster_id': cluster_id,
                'size': len(cluster_indices),
                'dominant_type': dominant,
                'distribution': dict(sample_counts),
                'n_examples_saved': len(example_images)
            }
            
            with open(cluster_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _create_montage(self, images, save_path, cluster_id, sample_counts):
        """Create visual montage of cluster examples"""
        n_images = len(images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        cell_size = 224
        
        montage = np.ones((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8) * 255
        
        for idx, img in enumerate(images):
            row = idx // grid_size
            col = idx % grid_size
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            montage[y1:y2, x1:x2] = img
        
        banner_height = 80
        banner = np.zeros((banner_height, montage.shape[1], 3), dtype=np.uint8)
        
        title = f"Cluster {cluster_id}"
        subtitle = f"{sample_counts.most_common(1)[0][0]}: {sample_counts.most_common(1)[0][1]} cells"
        
        cv2.putText(banner, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(banner, subtitle, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        final = np.vstack([banner, montage])
        cv2.imwrite(str(save_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    
    def _analyze_clusters(self, cluster_labels, sample_types):
        """Analyze cluster composition"""
        cluster_analysis = {}
        
        for cluster_id in range(self.config.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_sample_types = [sample_types[i] for i in cluster_indices]
            sample_counts = Counter(cluster_sample_types)
            
            total_aml = sum(1 for s in sample_types if s == 'aml')
            total_healthy = sum(1 for s in sample_types if s == 'healthy_bm')
            
            aml_in_cluster = sample_counts.get('aml', 0)
            healthy_in_cluster = sample_counts.get('healthy_bm', 0)
            
            if total_aml > 0:
                aml_enrichment = (aml_in_cluster / len(cluster_indices)) / (total_aml / len(sample_types))
            else:
                aml_enrichment = 0
            
            if total_healthy > 0:
                healthy_enrichment = (healthy_in_cluster / len(cluster_indices)) / (total_healthy / len(sample_types))
            else:
                healthy_enrichment = 0
            
            cluster_analysis[f"cluster_{cluster_id:02d}"] = {
                'size': len(cluster_indices),
                'distribution': dict(sample_counts),
                'dominant': sample_counts.most_common(1)[0][0] if sample_counts else 'unknown',
                'purity': sample_counts.most_common(1)[0][1] / len(cluster_indices) if len(cluster_indices) > 0 else 0,
                'aml_enrichment': float(aml_enrichment),
                'healthy_enrichment': float(healthy_enrichment)
            }
        
        return cluster_analysis
    
    def _perform_comparisons(self, cluster_labels, sample_types):
        """Compare distributions between conditions"""
        comparisons = {}
        
        # AML vs Healthy
        aml_indices = [i for i, s in enumerate(sample_types) if s == 'aml']
        healthy_indices = [i for i, s in enumerate(sample_types) if s == 'healthy_bm']
        
        if aml_indices and healthy_indices:
            aml_clusters = cluster_labels[aml_indices]
            healthy_clusters = cluster_labels[healthy_indices]
            
            aml_counts = np.bincount(aml_clusters, minlength=self.config.n_clusters)
            healthy_counts = np.bincount(healthy_clusters, minlength=self.config.n_clusters)
            
            cont = np.vstack([aml_counts, healthy_counts])
            chi2_stat, p_value, dof, expected = chi2_contingency(cont)
            
            comparisons['aml_vs_healthy'] = {
                'aml_counts': aml_counts.tolist(),
                'healthy_counts': healthy_counts.tolist(),
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            
            logger.info(f"AML vs Healthy: χ²={chi2_stat:.2f}, p={p_value:.4e}, dof={dof}")
        
        # Treatment comparisons
        dmso_indices = [i for i, s in enumerate(sample_types) if s == 'dmso_control']
        venetoclax_indices = [i for i, s in enumerate(sample_types) if s == 'venetoclax_treated']
        
        if dmso_indices and venetoclax_indices:
            dmso_clusters = cluster_labels[dmso_indices]
            vene_clusters = cluster_labels[venetoclax_indices]
            
            dmso_counts = np.bincount(dmso_clusters, minlength=self.config.n_clusters)
            vene_counts = np.bincount(vene_clusters, minlength=self.config.n_clusters)
            
            cont = np.vstack([dmso_counts, vene_counts])
            chi2_stat, p_value, dof, expected = chi2_contingency(cont)
            
            comparisons['dmso_vs_venetoclax'] = {
                'dmso_counts': dmso_counts.tolist(),
                'venetoclax_counts': vene_counts.tolist(),
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            
            logger.info(f"DMSO vs Venetoclax: χ²={chi2_stat:.2f}, p={p_value:.4e}, dof={dof}")
        
        return comparisons
    
    def _detect_artifacts_improved(self, features, cluster_labels):
        """Improved artifact detection"""
        artifact_clusters = []
        
        shape_features = features[:, -5:]
        
        for cluster_id in range(self.config.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) < 10:
                continue
            
            cluster_shape_features = shape_features[cluster_indices]
            
            areas = cluster_shape_features[:, 0]
            mean_area = np.mean(areas)
            
            circularities = cluster_shape_features[:, 2]
            mean_circularity = np.mean(circularities)
            
            object_counts = cluster_shape_features[:, 4]
            mean_objects = np.mean(object_counts)
            
            reasons = []
            
            if mean_area > np.percentile(shape_features[:, 0], 99):
                reasons.append('Very large area (possible clump)')
            
            if mean_circularity < 0.3:
                reasons.append('Very low circularity (possible debris)')
            
            if mean_objects > 2.0:
                reasons.append('Multiple objects (possible multiplets)')
            
            if mean_area < np.percentile(shape_features[:, 0], 1):
                reasons.append('Very small area (possible debris)')
            
            if reasons:
                artifact_clusters.append({
                    'cluster_id': cluster_id,
                    'reasons': reasons,
                    'size': len(cluster_indices)
                })
        
        return artifact_clusters
    
    def _generate_report(self, cluster_analysis, comparisons, artifacts, silhouette):
        """Generate comprehensive report"""
        report_path = self.config.cluster_dir / "clinical_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CLINICAL CELL PAINTING ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total clusters: {self.config.n_clusters}\n")
            f.write(f"Silhouette score: {silhouette:.3f}\n\n")
            
            f.write("DISEASE STATE ANALYSIS\n")
            f.write("-"*40 + "\n")
            
            aml_enriched = []
            healthy_enriched = []
            
            for cluster_name, info in cluster_analysis.items():
                if info['aml_enrichment'] > 1.5:
                    aml_enriched.append((cluster_name, info['aml_enrichment']))
                if info['healthy_enrichment'] > 1.5:
                    healthy_enriched.append((cluster_name, info['healthy_enrichment']))
            
            f.write(f"AML-enriched clusters: {len(aml_enriched)}\n")
            for cluster, enrichment in aml_enriched[:5]:
                f.write(f"  {cluster}: {enrichment:.2f}x\n")
            
            f.write(f"\nHealthy-enriched clusters: {len(healthy_enriched)}\n")
            for cluster, enrichment in healthy_enriched[:5]:
                f.write(f"  {cluster}: {enrichment:.2f}x\n")
            
            if comparisons:
                f.write("\n\nCOMPARATIVE ANALYSES\n")
                f.write("-"*40 + "\n")
                
                for comp_name, comp_data in comparisons.items():
                    f.write(f"\n{comp_name.replace('_', ' ').upper()}:\n")
                    f.write(f"  Chi-squared statistic: {comp_data['chi2_statistic']:.2f}\n")
                    f.write(f"  Degrees of freedom: {comp_data.get('degrees_of_freedom', 'N/A')}\n")
                    f.write(f"  P-value: {comp_data['p_value']:.4e}\n")
                    f.write(f"  Significant (α=0.05): {'Yes' if comp_data['significant'] else 'No'}\n")
            
            if artifacts:
                f.write("\n\nPOTENTIAL ARTIFACTS\n")
                f.write("-"*40 + "\n")
                f.write(f"Found {len(artifacts)} potential artifact clusters:\n")
                for artifact in artifacts:
                    f.write(f"  Cluster {artifact['cluster_id']}: {', '.join(artifact['reasons'])}\n")
            
            f.write("\n\nCLUSTER DETAILS\n")
            f.write("-"*40 + "\n")
            for cluster_name, info in sorted(cluster_analysis.items()):
                f.write(f"\n{cluster_name}:\n")
                f.write(f"  Size: {info['size']} cells\n")
                f.write(f"  Dominant: {info['dominant']}\n")
                f.write(f"  Purity: {info['purity']:.2%}\n")
    
    def _generate_visualizations(self, cluster_labels, sample_types):
        """Generate comparison visualizations"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            aml_mask = np.array([s == 'aml' for s in sample_types])
            healthy_mask = np.array([s == 'healthy_bm' for s in sample_types])
            
            if aml_mask.any() and healthy_mask.any():
                aml_counts = np.bincount(cluster_labels[aml_mask], minlength=self.config.n_clusters)
                healthy_counts = np.bincount(cluster_labels[healthy_mask], minlength=self.config.n_clusters)
                
                x = np.arange(self.config.n_clusters)
                width = 0.35
                
                ax1.bar(x - width/2, aml_counts, width, label='AML', color='red', alpha=0.7)
                ax1.bar(x + width/2, healthy_counts, width, label='Healthy BM', color='green', alpha=0.7)
                ax1.set_xlabel('Cluster ID')
                ax1.set_ylabel('Number of Cells')
                ax1.set_title('Disease State Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No AML/Healthy data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Disease State Distribution')
            
            dmso_mask = np.array([s == 'dmso_control' for s in sample_types])
            vene_mask = np.array([s == 'venetoclax_treated' for s in sample_types])
            
            if dmso_mask.any() and vene_mask.any():
                dmso_counts = np.bincount(cluster_labels[dmso_mask], minlength=self.config.n_clusters)
                vene_counts = np.bincount(cluster_labels[vene_mask], minlength=self.config.n_clusters)
                
                x = np.arange(self.config.n_clusters)
                ax2.bar(x - width/2, dmso_counts, width, label='DMSO', color='blue', alpha=0.7)
                ax2.bar(x + width/2, vene_counts, width, label='Venetoclax', color='orange', alpha=0.7)
                ax2.set_xlabel('Cluster ID')
                ax2.set_ylabel('Number of Cells')
                ax2.set_title('Treatment Effect')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No treatment data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Treatment Effect')
            
            plt.tight_layout()
            plt.savefig(self.config.comparison_dir / 'comparisons.png', dpi=150)
            plt.close()
            
            logger.info("Generated comparison visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Cell Painting Analysis')
    parser.add_argument('--image-dir', type=str, nargs='+', required=True)
    parser.add_argument('--fcs-dir', type=str, nargs='+', required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/clinical_analysis')
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--n-examples', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-samples', type=int, default=5000)
    parser.add_argument('--load-existing-models', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--use-deep-features', action='store_true')
    parser.add_argument('--analyze-fcs', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Create configuration
    config = ClinicalConfig()
    config.output_dir = Path(args.output_dir)
    config.cluster_dir = Path(args.output_dir) / 'clusters'
    config.models_dir = Path(args.output_dir) / 'models'
    config.comparison_dir = Path(args.output_dir) / 'comparisons'
    config.fcs_dir = Path(args.output_dir) / 'fcs_analysis'
    config.n_clusters = args.n_clusters
    config.n_examples_per_cluster = args.n_examples
    config.batch_size = args.batch_size
    config.save_models = args.save_models
    config.load_existing_models = args.load_existing_models
    config.use_deep_features = args.use_deep_features
    
    # Create directories
    for dir_path in [config.output_dir, config.cluster_dir, config.models_dir, 
                     config.comparison_dir, config.fcs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("CLINICAL CELL PAINTING ANALYSIS V6")
    logger.info("="*60)
    
    # Initialize components
    data_handler = ClinicalDataHandler(config)
    processor = ClinicalTIFFProcessor(config)
    extractor = FeatureExtractor(config)
    analyzer = ClusterAnalyzer(config)
    fcs_analyzer = FCSAnalyzer(config)
    
    # FCS ANALYSIS
    fcs_stats = pd.DataFrame()
    fcs_comparisons = {}
    
    if args.analyze_fcs and HAVE_FCS:
        logger.info("\n" + "="*40)
        logger.info("STARTING FCS ANALYSIS")
        logger.info("="*40)
        
        fcs_files = fcs_analyzer.find_fcs_files(args.fcs_dir)
        
        if fcs_files:
            fcs_stats = fcs_analyzer.analyze_all_fcs(fcs_files)
            
            if not fcs_stats.empty:
                fcs_comparisons = fcs_analyzer.compare_fcs_populations(fcs_stats)
                fcs_analyzer.visualize_fcs_comparisons(fcs_stats, fcs_comparisons)
                
                with open(config.fcs_dir / 'fcs_comparisons.json', 'w') as f:
                    json.dump(fcs_comparisons, f, indent=2, default=str)
    
    # IMAGING ANALYSIS
    logger.info("\n" + "="*40)
    logger.info("STARTING IMAGING ANALYSIS")
    logger.info("="*40)
    
    sample_images = data_handler.find_images(args.image_dir, args.max_samples)
    
    if not sample_images:
        logger.error("No images found!")
        return
    
    all_paths = []
    all_sample_types = []
    for sample_type, paths in sample_images.items():
        all_paths.extend(paths)
        all_sample_types.extend([sample_type] * len(paths))
    
    logger.info(f"Total images to process: {len(all_paths)}")
    
    # Extract features
    logger.info("Extracting features...")
    all_features = []
    
    batch_size = 100
    for i in tqdm(range(0, len(all_paths), batch_size), desc="Processing images"):
        batch_paths = all_paths[i:i+batch_size]
        batch_features = []
        
        for path in batch_paths:
            img = processor.load_tiff(path)
            morph = extractor.extract_morphological(img)
            batch_features.append(morph)
        
        if batch_features:
            all_features.extend(batch_features)
    
    all_features = np.vstack(all_features)
    logger.info(f"Extracted features shape: {all_features.shape}")
    
    # Perform clustering
    logger.info("Performing clustering and analysis...")
    results = analyzer.perform_clustering(all_features, all_paths, all_sample_types)
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()

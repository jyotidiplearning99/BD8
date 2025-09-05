#!/usr/bin/env python3
"""
Clinical Cell Painting Analysis Pipeline V4 - COMPLETE FIXED VERSION
With improved FCS analysis, artifact detection, and visualizations
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
from sklearn.manifold import TSNE
import joblib

import albumentations as A
from albumentations.pytorch import ToTensorV2
import tifffile
from tqdm.auto import tqdm
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal

try:
    import fcsparser
    import flowkit as fk
    HAVE_FCS = True
except:
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
    
    # Flow cytometry markers - expanded list
    flow_markers: List[str] = field(default_factory=lambda: [
        'CD3', 'CD4', 'CD8', 'CD19', 'CD14', 'CD16', 'CD56',
        'CD45', 'CD34', 'CD117', 'HLA-DR', 'CD11b', 'CD33',
        'CD36', 'CD38', 'BCL2', 'MPO', 'CTSG', 'HLA-ABC'
    ])
    
    output_dir: Path = Path('outputs/clinical_analysis')
    cluster_dir: Path = Path('outputs/clinical_analysis/clusters')
    models_dir: Path = Path('outputs/clinical_analysis/models')
    comparison_dir: Path = Path('outputs/clinical_analysis/comparisons')
    fcs_dir: Path = Path('outputs/clinical_analysis/fcs_analysis')
    
    # REDUCED CLUSTERS FOR BETTER SEPARATION
    n_clusters: int = 10  # Reduced from 15
    n_examples_per_cluster: int = 30
    batch_size: int = 32
    use_morphological: bool = True
    use_deep_features: bool = False
    save_models: bool = True
    load_existing_models: bool = True
    max_silhouette_samples: int = 5000
    
    # FCS analysis parameters
    fcs_subsample: int = 10000
    fcs_transform: str = 'arcsinh'
    fcs_cofactor: float = 150.0

class FCSAnalyzer:
    """Analyze FCS flow cytometry files with improved marker detection"""
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
            
            # Find all FCS files
            for pattern in ['*.fcs', '*.FCS']:
                for fcs_path in Path(base_dir).rglob(pattern):
                    # Skip compensation files
                    if 'compensation' in str(fcs_path).lower():
                        continue
                    
                    sample_type = self._identify_fcs_type(str(fcs_path))
                    fcs_files[sample_type].append(fcs_path)
        
        # Log distribution
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
        
        # Specific experiments
        if 'fh_7087' in path_lower:
            return 'heidelberg_comparison'
        elif 'fh_8445' in path_lower:
            if 'dmso' in path_lower:
                return 'dmso_control'
            elif 'venetoclax' in path_lower or 'vene' in path_lower:
                return 'venetoclax_treated'
        
        # Pre-sort experiments by date
        elif '14.5.2025' in path_lower or 'exp 8' in path_lower:
            return 'exp8_presort'
        elif '8.5.2025' in path_lower or '8.4.2025' in path_lower or 'exp 7' in path_lower:
            return 'exp7_presort'
        elif '12.3.2025' in path_lower or 'exp 5' in path_lower:
            return 'exp5_presort'
        
        # General categories
        elif 'aml' in path_lower or 'ps1' in path_lower or 'full stain' in path_lower:
            return 'aml'
        elif 'healthy' in path_lower or 'bm' in path_lower or 'notreatment' in path_lower:
            return 'healthy_bm'
        elif 'pre-sort' in path_lower or 'presort' in path_lower:
            return 'presort_pbmc'
        
        return 'unknown'
    
    def analyze_fcs_file(self, fcs_path: Path) -> Optional[Dict]:
        """Analyze single FCS file with improved marker detection"""
        if not HAVE_FCS:
            return None
        
        try:
            # Try fcsparser first
            meta, data = fcsparser.parse(str(fcs_path))
            
            # Get channel names more robustly
            channels = []
            for i in range(1, meta['$PAR'] + 1):
                # Try stain name first, then parameter name
                if f'$P{i}S' in meta and meta[f'$P{i}S']:
                    channels.append(meta[f'$P{i}S'])
                elif f'$P{i}N' in meta:
                    channels.append(meta[f'$P{i}N'])
                else:
                    channels.append(f'Ch{i}')
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=channels)
            
            # Debug: Log first few channel names
            if logger.level <= logging.DEBUG:
                logger.debug(f"Channels in {fcs_path.name}: {channels[:10]}")
            
            # Subsample if needed
            if len(df) > self.config.fcs_subsample:
                df = df.sample(n=self.config.fcs_subsample, random_state=SEED)
            
            # Apply transformation
            if self.config.fcs_transform == 'arcsinh':
                # Find fluorescence channels (exclude scatter and time)
                fluor_cols = [col for col in df.columns 
                             if not any(x in col.upper() for x in ['FSC', 'SSC', 'TIME', 'EVENT'])]
                
                for col in fluor_cols:
                    # Only transform positive values
                    if df[col].min() >= 0:
                        df[col] = np.arcsinh(df[col] / self.config.fcs_cofactor)
            
            # Calculate statistics
            stats_dict = {
                'n_events': len(data),
                'n_channels': meta['$PAR'],
                'channels': channels,
                'file_name': fcs_path.name,
                'sample_type': self._identify_fcs_type(str(fcs_path))
            }
            
            # Extract marker statistics - improved matching
            for marker in self.config.flow_markers:
                # Try exact match first
                matching_cols = [col for col in df.columns if marker == col]
                
                # Then try substring match
                if not matching_cols:
                    matching_cols = [col for col in df.columns 
                                   if marker in col or col in marker]
                
                # Then try case-insensitive match
                if not matching_cols:
                    matching_cols = [col for col in df.columns 
                                   if marker.upper() in col.upper()]
                
                # Use the first match
                if matching_cols:
                    col = matching_cols[0]
                    values = df[col].dropna()
                    if len(values) > 0:
                        stats_dict[f'{marker}_mean'] = float(values.mean())
                        stats_dict[f'{marker}_median'] = float(values.median())
                        stats_dict[f'{marker}_cv'] = float(values.std() / values.mean()) if values.mean() != 0 else 0
            
            # Store processed data for later use
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
            
            # Save summary
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
                        # Mann-Whitney U test
                        stat, p_value = mannwhitneyu(aml_values, healthy_values, alternative='two-sided')
                        
                        marker_comparisons[marker] = {
                            'aml_mean': float(aml_values.mean()),
                            'healthy_mean': float(healthy_values.mean()),
                            'fold_change': float(aml_values.mean() / healthy_values.mean()) if healthy_values.mean() != 0 else np.inf,
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
                        
                        treatment_comparisons[marker] = {
                            'dmso_mean': float(dmso_values.mean()),
                            'venetoclax_mean': float(vene_values.mean()),
                            'fold_change': float(vene_values.mean() / dmso_values.mean()) if dmso_values.mean() != 0 else np.inf,
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
            
            comparisons['dmso_vs_venetoclax_markers'] = treatment_comparisons
        
        return comparisons
    
    def visualize_fcs_comparisons(self, df_stats: pd.DataFrame, comparisons: Dict):
        """Create IMPROVED visualizations for FCS comparisons"""
        if df_stats.empty:
            return
        
        # Create figure with multiple subplots
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
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Marker expression heatmap for AML vs Healthy - FIXED
        ax2 = plt.subplot(2, 3, 2)
        
        if 'aml_vs_healthy_markers' in comparisons and comparisons['aml_vs_healthy_markers']:
            markers = []
            fold_changes = []
            p_values = []
            
            for marker, stats in comparisons['aml_vs_healthy_markers'].items():
                if not np.isinf(stats['fold_change']) and stats['fold_change'] > 0:
                    markers.append(marker)
                    fold_changes.append(np.log2(stats['fold_change']))
                    p_values.append(-np.log10(stats['p_value']))
            
            if markers:
                y_pos = np.arange(len(markers))
                colors = ['red' if fc > 0 else 'blue' for fc in fold_changes]
                bars = ax2.barh(y_pos, fold_changes, color=colors, alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(markers)
                ax2.set_xlabel('Log2 Fold Change (AML/Healthy)')
                ax2.set_title('Marker Expression Changes')
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax2.grid(True, alpha=0.3)
                
                # Add significance stars
                for i, (fc, pval) in enumerate(zip(fold_changes, p_values)):
                    if pval > -np.log10(0.05):  # p < 0.05
                        ax2.text(fc, i, '*', ha='left' if fc > 0 else 'right', va='center')
            else:
                ax2.text(0.5, 0.5, 'No significant\nmarker changes', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Marker Expression Changes')
        else:
            ax2.text(0.5, 0.5, 'No AML vs Healthy\ncomparison available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Marker Expression Changes')
        
        # 3. Treatment effect visualization - FIXED
        ax3 = plt.subplot(2, 3, 3)
        
        if 'dmso_vs_venetoclax_markers' in comparisons and comparisons['dmso_vs_venetoclax_markers']:
            markers = []
            fold_changes = []
            significant = []
            
            for marker, stats in comparisons['dmso_vs_venetoclax_markers'].items():
                if not np.isinf(stats['fold_change']) and stats['fold_change'] > 0:
                    markers.append(marker)
                    fold_changes.append(np.log2(stats['fold_change']))
                    significant.append(stats['significant'])
            
            if markers:
                # Only show significant markers
                sig_markers = [m for m, s in zip(markers, significant) if s]
                sig_fold_changes = [fc for fc, s in zip(fold_changes, significant) if s]
                
                if sig_markers:
                    y_pos = np.arange(len(sig_markers))
                    colors = ['orange' if fc > 0 else 'purple' for fc in sig_fold_changes]
                    ax3.barh(y_pos, sig_fold_changes, color=colors, alpha=0.7)
                    ax3.set_yticks(y_pos)
                    ax3.set_yticklabels(sig_markers)
                    ax3.set_xlabel('Log2 Fold Change (Venetoclax/DMSO)')
                    ax3.set_title('Treatment Effects (p < 0.05)')
                    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'No significant\ntreatment effects', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Treatment Effects')
            else:
                ax3.text(0.5, 0.5, 'No markers\nanalyzed', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Treatment Effects')
        else:
            ax3.text(0.5, 0.5, 'No treatment\ncomparison available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Treatment Effects')
        
        # 4. Event count distribution
        ax4 = plt.subplot(2, 3, 4)
        if 'n_events' in df_stats.columns:
            ax4.hist(df_stats['n_events'], bins=30, alpha=0.7, color='green', edgecolor='black')
            ax4.set_xlabel('Number of Events')
            ax4.set_ylabel('Frequency')
            ax4.set_title('FCS Event Count Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics
            mean_events = df_stats['n_events'].mean()
            median_events = df_stats['n_events'].median()
            ax4.axvline(mean_events, color='red', linestyle='--', label=f'Mean: {mean_events:.0f}')
            ax4.axvline(median_events, color='blue', linestyle='--', label=f'Median: {median_events:.0f}')
            ax4.legend()
        
        # 5. Experiment timeline
        ax5 = plt.subplot(2, 3, 5)
        exp_types = ['exp5_presort', 'exp7_presort', 'exp8_presort']
        exp_counts = [len(df_stats[df_stats['sample_type'] == exp]) for exp in exp_types]
        exp_labels = ['Exp 5\n(12.3.2025)', 'Exp 7\n(8.5.2025)', 'Exp 8\n(14.5.2025)']
        
        if any(exp_counts):
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            bars = ax5.bar(range(len(exp_labels)), exp_counts, color=colors, alpha=0.7)
            ax5.set_xticks(range(len(exp_labels)))
            ax5.set_xticklabels(exp_labels)
            ax5.set_ylabel('Number of FCS Files')
            ax5.set_title('Pre-sort Experiments Timeline')
            ax5.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, exp_counts):
                if count > 0:
                    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{count}', ha='center', va='bottom')
        else:
            ax5.text(0.5, 0.5, 'No experiment-specific\nFCS files found', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Pre-sort Experiments Timeline')
        
        # 6. Marker correlation matrix
        ax6 = plt.subplot(2, 3, 6)
        
        # Find marker columns that exist
        marker_cols = []
        for marker in self.config.flow_markers[:8]:  # Limit to first 8 markers
            mean_col = f'{marker}_mean'
            if mean_col in df_stats.columns:
                marker_cols.append(mean_col)
        
        if len(marker_cols) > 1:
            # Calculate correlation matrix
            corr_data = df_stats[marker_cols].dropna()
            if len(corr_data) > 1:
                corr_matrix = corr_data.corr()
                
                # Plot heatmap
                im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                ax6.set_xticks(range(len(marker_cols)))
                ax6.set_yticks(range(len(marker_cols)))
                ax6.set_xticklabels([col.replace('_mean', '') for col in marker_cols], rotation=45, ha='right')
                ax6.set_yticklabels([col.replace('_mean', '') for col in marker_cols])
                ax6.set_title('Marker Correlation Matrix')
                
                # Add correlation values
                for i in range(len(marker_cols)):
                    for j in range(len(marker_cols)):
                        text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                                       fontsize=8)
                
                plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
            else:
                ax6.text(0.5, 0.5, 'Insufficient data\nfor correlation', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Marker Correlation Matrix')
        else:
            ax6.text(0.5, 0.5, 'Insufficient markers\nfor correlation', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Marker Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.config.fcs_dir / 'fcs_analysis_improved.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved improved FCS analysis visualization to {self.config.fcs_dir / 'fcs_analysis_improved.png'}")

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
            
            # Handle different dtypes
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Handle channel dimensions
            if img.ndim == 3:
                if img.shape[0] in (3, 5, 6) and img.shape[0] < min(img.shape[1], img.shape[2]):
                    img = np.transpose(img, (1, 2, 0))
                
                if img.shape[2] == 5:
                    # 5-channel Cell Painting
                    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    rgb[:,:,2] = self._normalize_channel(img[:,:,0])  # Nucleus
                    rgb[:,:,1] = self._normalize_channel(img[:,:,1])  # ER
                    rgb[:,:,0] = self._normalize_channel(img[:,:,2])  # AGP
                    img = rgb
                elif img.shape[2] > 3:
                    img = img[:,:,:3].astype(np.uint8)
            elif img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            
            # Ensure correct shape
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
    """Handle clinical sample data with deduplication"""
    def __init__(self, config):
        self.config = config
        self.processed_files = set()
        
    def find_images(self, base_dirs: Union[str, List[str]], max_samples=None):
        """Find and categorize images from multiple directories without duplication"""
        sample_images = defaultdict(list)
        
        if isinstance(base_dirs, str):
            base_dirs = [base_dirs]
        
        # Process each directory
        for base_dir in base_dirs:
            if not Path(base_dir).exists():
                logger.warning(f"Directory {base_dir} does not exist, skipping")
                continue
            
            # Find all TIFF files
            tiff_files = []
            for pattern in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                tiff_files.extend(Path(base_dir).rglob(pattern))
            
            logger.info(f"Found {len(tiff_files)} TIFF files in {base_dir}")
            
            # Categorize by sample type (avoiding duplicates)
            for tiff_path in tiff_files:
                # Create unique identifier
                file_id = (tiff_path.name, tiff_path.stat().st_size)
                
                if file_id not in self.processed_files:
                    self.processed_files.add(file_id)
                    path_str = str(tiff_path).lower()
                    sample_type = self._identify_sample_type(path_str)
                    sample_images[sample_type].append(tiff_path)
        
        # Balance samples if needed
        if max_samples and len(sample_images) > 0:
            n_groups = len(sample_images)
            per_group = max(1, max_samples // n_groups)
            
            for key in list(sample_images.keys()):
                if len(sample_images[key]) > per_group:
                    np.random.seed(SEED)
                    indices = np.random.choice(len(sample_images[key]), per_group, replace=False)
                    sample_images[key] = [sample_images[key][i] for i in indices]
        
        # Log distribution
        logger.info("Image distribution by sample type (deduplicated):")
        total_images = 0
        for key in sorted(sample_images.keys()):
            count = len(sample_images[key])
            total_images += count
            logger.info(f"  {key}: {count} images")
        logger.info(f"Total unique images: {total_images}")
        
        return dict(sample_images)
    
    def _identify_sample_type(self, path_str):
        """Identify sample type from path"""
        # AML samples
        if any(x in path_str for x in ['aml', 'ps1', 'full stain imaging exp']):
            return 'aml'
        
        # Healthy bone marrow
        elif any(x in path_str for x in ['healthy', 'bm_2025', 'notreatment', 'normal_bm']):
            return 'healthy_bm'
        
        # Treatment samples from FH_8445
        elif 'fh_8445' in path_str or '8445' in path_str:
            if 'dmso' in path_str:
                return 'dmso_control'
            elif any(x in path_str for x in ['venetoclax', 'vene', 'ven']):
                return 'venetoclax_treated'
        
        # Heidelberg comparison (FH_7087)
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
    """Perform clustering and analysis with IMPROVED artifact detection"""
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
        
        # Clustering with reduced clusters
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
        
        # Detect artifacts with IMPROVED thresholds
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
        """Save example TIFFs for each cluster with preserved extensions"""
        processor = ClinicalTIFFProcessor(self.config)
        
        for cluster_id in range(self.config.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Get sample type distribution
            cluster_sample_types = [sample_types[i] for i in cluster_indices]
            sample_counts = Counter(cluster_sample_types)
            dominant = sample_counts.most_common(1)[0][0]
            
            # Create cluster directory
            cluster_path = self.config.cluster_dir / f"cluster_{cluster_id:02d}_{dominant}"
            cluster_path.mkdir(exist_ok=True)
            
            # Save examples
            n_examples = min(self.config.n_examples_per_cluster, len(cluster_indices))
            selected_positions = np.linspace(0, len(cluster_indices)-1, n_examples, dtype=int)
            selected_indices = cluster_indices[selected_positions]
            
            example_images = []
            for i, idx in enumerate(selected_indices):
                src_path = image_paths[idx]
                sample_type = sample_types[idx]
                
                # Preserve original file extension
                ext = src_path.suffix.lower()
                dst_path = cluster_path / f"example_{i:03d}_{sample_type}{ext}"
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    img = processor.load_tiff(src_path)
                    example_images.append(cv2.resize(img, (224, 224)))
            
            # Create montage
            if example_images:
                self._create_montage(example_images, cluster_path / "montage.png", 
                                   cluster_id, sample_counts)
            
            # Save metadata
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
        
        # Add text overlay
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
            
            # Calculate enrichment
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
        """Compare distributions between conditions using proper chi-squared test"""
        comparisons = {}
        
        # AML vs Healthy
        aml_indices = [i for i, s in enumerate(sample_types) if s == 'aml']
        healthy_indices = [i for i, s in enumerate(sample_types) if s == 'healthy_bm']
        
        if aml_indices and healthy_indices:
            aml_clusters = cluster_labels[aml_indices]
            healthy_clusters = cluster_labels[healthy_indices]
            
            # Use chi2_contingency for proper 2×K contingency table
            aml_counts = np.bincount(aml_clusters, minlength=self.config.n_clusters)
            healthy_counts = np.bincount(healthy_clusters, minlength=self.config.n_clusters)
            
            # Create contingency table
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
            
            # Use chi2_contingency for treatment comparison
            dmso_counts = np.bincount(dmso_clusters, minlength=self.config.n_clusters)
            vene_counts = np.bincount(vene_clusters, minlength=self.config.n_clusters)
            
            # Create contingency table
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
        else:
            logger.warning(f"No treatment samples found - DMSO: {len(dmso_indices)}, Venetoclax: {len(venetoclax_indices)}")
        
        return comparisons
    
    def _detect_artifacts_improved(self, features, cluster_labels):
        """IMPROVED artifact detection with better thresholds"""
        artifact_clusters = []
        
        # Assuming last 5 features are shape-based
        shape_features = features[:, -5:]
        
        for cluster_id in range(self.config.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) < 10:
                continue
            
            cluster_shape_features = shape_features[cluster_indices]
            
            # Check area (feature -5)
            areas = cluster_shape_features[:, 0]
            mean_area = np.mean(areas)
            
            # Check circularity (feature -3)
            circularities = cluster_shape_features[:, 2]
            mean_circularity = np.mean(circularities)
            
            # Check object count (feature -1)
            object_counts = cluster_shape_features[:, 4]
            mean_objects = np.mean(object_counts)
            
            # IMPROVED artifact criteria with more reasonable thresholds
            reasons = []
            
            # Use 99th percentile instead of 95th for area
            if mean_area > np.percentile(shape_features[:, 0], 99):
                reasons.append('Very large area (possible clump)')
            
            # Lower circularity threshold from 0.4 to 0.3
            if mean_circularity < 0.3:
                reasons.append('Very low circularity (possible debris)')
            
            # Only flag if average object count is > 2
            if mean_objects > 2.0:
                reasons.append('Multiple objects (possible multiplets)')
            
            # Additional check: extremely small area
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
            
            # Disease analysis
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
            
            # Comparisons
            if comparisons:
                f.write("\n\nCOMPARATIVE ANALYSES\n")
                f.write("-"*40 + "\n")
                
                for comp_name, comp_data in comparisons.items():
                    f.write(f"\n{comp_name.replace('_', ' ').upper()}:\n")
                    f.write(f"  Chi-squared statistic: {comp_data['chi2_statistic']:.2f}\n")
                    f.write(f"  Degrees of freedom: {comp_data.get('degrees_of_freedom', 'N/A')}\n")
                    f.write(f"  P-value: {comp_data['p_value']:.4e}\n")
                    f.write(f"  Significant (α=0.05): {'Yes' if comp_data['significant'] else 'No'}\n")
            
            # Artifacts
            if artifacts:
                f.write("\n\nPOTENTIAL ARTIFACTS (with improved detection)\n")
                f.write("-"*40 + "\n")
                f.write(f"Found {len(artifacts)} potential artifact clusters:\n")
                for artifact in artifacts:
                    f.write(f"  Cluster {artifact['cluster_id']}: {', '.join(artifact['reasons'])}\n")
            else:
                f.write("\n\nARTIFACT DETECTION\n")
                f.write("-"*40 + "\n")
                f.write("No clear artifact clusters detected with improved thresholds.\n")
            
            # Cluster details
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
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Count distributions
            aml_mask = np.array([s == 'aml' for s in sample_types])
            healthy_mask = np.array([s == 'healthy_bm' for s in sample_types])
            
            # Plot Disease comparison if data exists
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
            
            # Treatment comparison
            dmso_mask = np.array([s == 'dmso_control' for s in sample_types])
            vene_mask = np.array([s == 'venetoclax_treated' for s in sample_types])
            
            # Plot Treatment comparison if data exists
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

def integrate_fcs_with_imaging(fcs_stats: pd.DataFrame, cluster_analysis: Dict, config: ClinicalConfig):
    """Integrate FCS flow cytometry data with imaging cluster results"""
    integration_report = config.output_dir / 'integrated_analysis_report.txt'
    
    with open(integration_report, 'w') as f:
        f.write("INTEGRATED FCS AND IMAGING ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        # Summary statistics
        f.write("DATA SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total FCS files analyzed: {len(fcs_stats) if not fcs_stats.empty else 0}\n")
        f.write(f"Total imaging clusters: {len(cluster_analysis)}\n\n")
        
        # Cross-modal comparisons
        if not fcs_stats.empty:
            f.write("FCS SAMPLE TYPE DISTRIBUTION\n")
            f.write("-"*40 + "\n")
            for sample_type, count in fcs_stats['sample_type'].value_counts().items():
                f.write(f"  {sample_type}: {count} files\n")
            f.write("\n")
            
            # Marker expression summary
            f.write("KEY MARKER EXPRESSION (Mean ± SD)\n")
            f.write("-"*40 + "\n")
            
            # Find available markers
            marker_found = False
            for marker in ['CD3', 'CD4', 'CD8', 'CD19', 'CD14', 'CD45', 'CD34']:
                mean_col = f'{marker}_mean'
                if mean_col in fcs_stats.columns:
                    values = fcs_stats[mean_col].dropna()
                    if len(values) > 0:
                        f.write(f"  {marker}: {values.mean():.2f} ± {values.std():.2f}\n")
                        marker_found = True
            
            if not marker_found:
                f.write("  No marker data available\n")
        
        # Imaging cluster summary
        f.write("\nIMAGING CLUSTER SUMMARY\n")
        f.write("-"*40 + "\n")
        
        total_cells = sum(info['size'] for info in cluster_analysis.values())
        f.write(f"Total cells clustered: {total_cells}\n")
        
        # Find disease-specific clusters
        aml_specific = []
        healthy_specific = []
        
        for cluster_name, info in cluster_analysis.items():
            if info['aml_enrichment'] > 2.0:
                aml_specific.append(cluster_name)
            if info['healthy_enrichment'] > 2.0:
                healthy_specific.append(cluster_name)
        
        f.write(f"AML-specific clusters (>2x enrichment): {', '.join(aml_specific) if aml_specific else 'None'}\n")
        f.write(f"Healthy-specific clusters (>2x enrichment): {', '.join(healthy_specific) if healthy_specific else 'None'}\n")
    
    logger.info(f"Saved integrated analysis report to {integration_report}")

def main():
    """Main execution function with FCS integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Cell Painting Analysis with FCS Integration')
    parser.add_argument('--image-dir', type=str, nargs='+', required=True,
                        help='One or more directories containing images')
    parser.add_argument('--fcs-dir', type=str, nargs='+', required=True,
                        help='One or more directories containing FCS files')
    parser.add_argument('--output-dir', type=str, default='outputs/clinical_analysis')
    parser.add_argument('--n-clusters', type=int, default=10)  # Reduced default
    parser.add_argument('--n-examples', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--load-existing-models', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--use-deep-features', action='store_true')
    parser.add_argument('--analyze-fcs', action='store_true', default=True,
                        help='Perform FCS analysis')
    
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
    logger.info("CLINICAL CELL PAINTING ANALYSIS WITH FCS INTEGRATION")
    logger.info("="*60)
    
    # Initialize components
    data_handler = ClinicalDataHandler(config)
    processor = ClinicalTIFFProcessor(config)
    extractor = FeatureExtractor(config)
    analyzer = ClusterAnalyzer(config)
    fcs_analyzer = FCSAnalyzer(config)
    
    # ========== FCS ANALYSIS ==========
    fcs_stats = pd.DataFrame()
    fcs_comparisons = {}
    
    if args.analyze_fcs and HAVE_FCS:
        logger.info("\n" + "="*40)
        logger.info("STARTING FCS ANALYSIS")
        logger.info("="*40)
        
        # Find FCS files
        fcs_files = fcs_analyzer.find_fcs_files(args.fcs_dir)
        
        if fcs_files:
            # Analyze all FCS files
            fcs_stats = fcs_analyzer.analyze_all_fcs(fcs_files)
            
            if not fcs_stats.empty:
                # Compare populations
                fcs_comparisons = fcs_analyzer.compare_fcs_populations(fcs_stats)
                
                # Generate FCS visualizations
                fcs_analyzer.visualize_fcs_comparisons(fcs_stats, fcs_comparisons)
                
                # Save FCS comparison results
                with open(config.fcs_dir / 'fcs_comparisons.json', 'w') as f:
                    json.dump(fcs_comparisons, f, indent=2, default=str)
    
    # ========== IMAGING ANALYSIS ==========
    logger.info("\n" + "="*40)
    logger.info("STARTING IMAGING ANALYSIS")
    logger.info("="*40)
    
    # Find images from multiple directories
    sample_images = data_handler.find_images(args.image_dir, args.max_samples)
    
    if not sample_images:
        logger.error("No images found!")
        return
    
    # Flatten data
    all_paths = []
    all_sample_types = []
    for sample_type, paths in sample_images.items():
        all_paths.extend(paths)
        all_sample_types.extend([sample_type] * len(paths))
    
    logger.info(f"Total unique images to process: {len(all_paths)}")
    
    # Extract features
    logger.info("Extracting features...")
    all_features = []
    
    batch_size = 100
    for i in tqdm(range(0, len(all_paths), batch_size), desc="Processing images"):
        batch_paths = all_paths[i:i+batch_size]
        batch_images = []
        batch_features = []
        
        # Load images
        for path in batch_paths:
            img = processor.load_tiff(path)
            batch_images.append(img)
            
            # Extract morphological features
            morph = extractor.extract_morphological(img)
            batch_features.append(morph)
        
        if batch_features:
            all_features.extend(batch_features)
    
    all_features = np.vstack(all_features)
    logger.info(f"Extracted features shape: {all_features.shape}")
    
    # Perform clustering and analysis
    logger.info("Performing clustering and analysis...")
    results = analyzer.perform_clustering(all_features, all_paths, all_sample_types)
    
    # ========== INTEGRATION ==========
    if not fcs_stats.empty:
        logger.info("\n" + "="*40)
        logger.info("INTEGRATING FCS AND IMAGING DATA")
        logger.info("="*40)
        
        integrate_fcs_with_imaging(fcs_stats, results['cluster_analysis'], config)
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("Key outputs:")
    logger.info(f"  1. Cluster examples: {config.cluster_dir}")
    logger.info(f"  2. Imaging report: {config.cluster_dir}/clinical_analysis_report.txt")
    logger.info(f"  3. FCS analysis: {config.fcs_dir}")
    logger.info(f"  4. Visualizations: {config.comparison_dir}")
    logger.info(f"  5. Integrated report: {config.output_dir}/integrated_analysis_report.txt")
    logger.info("="*60)

if __name__ == "__main__":
    main()

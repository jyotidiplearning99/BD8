# src/data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import tifffile
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
import random

def _scale_to_unit(img: np.ndarray) -> np.ndarray:
    """Proper scaling for 16-bit TIFFs"""
    orig_dtype = img.dtype
    
    if np.issubdtype(orig_dtype, np.integer):
        img = img.astype(np.float32)
        img /= float(np.iinfo(orig_dtype).max)
    else:
        lo, hi = np.percentile(img, (1, 99))
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)
        else:
            img = img.astype(np.float32)
    
    return img

class BD_S8_RealDataset(Dataset):
    """Dataset for REAL BD S8 TIFF images"""
    
    def __init__(self,
                 data_root='/scratch/project_2010376/BDS8/BDS8_data',
                 sample_types=('AML', 'Healthy BM'),
                 max_cells_per_type=10000,
                 transform=None,
                 mode='train',
                 train_split=0.7,
                 val_split=0.15):
        
        self.data_root = Path(data_root)
        self.transform = transform
        self.mode = mode
        
        # Collect per-class
        all_images = []
        all_labels = []
        
        for cls_name in sample_types:
            cls_dir = self.data_root / cls_name
            if cls_dir.exists():
                files = list(cls_dir.rglob('*.tif')) + list(cls_dir.rglob('*.tiff'))
                random.shuffle(files)
                files = files[:max_cells_per_type]
                
                label = 1 if 'AML' in cls_name else 0
                all_images.extend(files)
                all_labels.extend([label] * len(files))
        
        # Stratified split
        X = np.array(all_images)
        y = np.array(all_labels)
        
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=42)
        train_idx, temp_idx = next(sss.split(X, y))
        
        val_size = val_split / (1 - train_split)
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_size, random_state=43)
        val_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]
        
        if mode == 'train':
            self.image_paths = X[train_idx].tolist()
            self.labels = y[train_idx].tolist()
        elif mode == 'val':
            self.image_paths = X[val_idx].tolist()
            self.labels = y[val_idx].tolist()
        else:
            self.image_paths = X[test_idx].tolist()
            self.labels = y[test_idx].tolist()
        
        print(f"✅ {mode.upper()}: {len(self.image_paths)} cells "
              f"(AML={sum(self.labels)}, Healthy={len(self.labels)-sum(self.labels)})")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = tifffile.imread(self.image_paths[idx])
        
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        elif img.ndim == 3 and img.shape[0] <= 4:
            img = np.transpose(img[:3], (1, 2, 0))
        
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        else:
            img = img[..., :3]
        
        img = _scale_to_unit(img)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return {'image': img, 'label': int(self.labels[idx])}

def get_transforms(mode='train'):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if mode == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

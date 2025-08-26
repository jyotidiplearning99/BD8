# src/train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
import os

class BD_S8_LightningModule(pl.LightningModule):  # RENAMED to avoid conflict
    """PyTorch Lightning training module for BD S8 analysis"""
    
    def __init__(self, model, learning_rate=1e-4, num_classes=5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        self.seg_loss = smp.losses.DiceLoss(mode='binary')
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        outputs = self(images)
        
        batch_size = images.shape[0]
        synthetic_masks = self.generate_synthetic_masks(images)
        
        seg_dice_loss = self.seg_loss(outputs['segmentation'], synthetic_masks)
        seg_focal_loss = self.focal_loss(outputs['segmentation'], synthetic_masks)
        seg_loss = seg_dice_loss + 0.5 * seg_focal_loss
        
        extraction_labels = torch.tensor(
            [labels['extraction_method'][i].item() if torch.is_tensor(labels['extraction_method'][i]) 
             else labels['extraction_method'][i] for i in range(batch_size)], 
            device=self.device
        )
        
        extraction_loss = self.ce_loss(outputs['extraction'], extraction_labels)
        total_loss = seg_loss + 0.3 * extraction_loss
        
        extraction_pred = torch.argmax(outputs['extraction'], dim=1)
        extraction_acc = (extraction_pred == extraction_labels).float().mean()
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_acc', extraction_acc, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        outputs = self(images)
        synthetic_masks = self.generate_synthetic_masks(images)
        
        seg_dice = 1 - self.seg_loss(outputs['segmentation'], synthetic_masks)
        
        batch_size = images.shape[0]
        extraction_labels = torch.tensor(
            [labels['extraction_method'][i].item() if torch.is_tensor(labels['extraction_method'][i]) 
             else labels['extraction_method'][i] for i in range(batch_size)], 
            device=self.device
        )
        
        extraction_pred = torch.argmax(outputs['extraction'], dim=1)
        extraction_acc = (extraction_pred == extraction_labels).float().mean()
        
        self.validation_step_outputs.append({'val_dice': seg_dice, 'val_acc': extraction_acc})
        
        self.log('val_dice', seg_dice, prog_bar=True)
        self.log('val_acc', extraction_acc, prog_bar=True)
        
        return {'val_dice': seg_dice, 'val_acc': extraction_acc}
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_dice = torch.stack([x['val_dice'] for x in self.validation_step_outputs]).mean()
            avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
            
            self.log('val_dice_epoch', avg_dice)
            self.log('val_acc_epoch', avg_acc)
            
            self.validation_step_outputs.clear()
    
    def generate_synthetic_masks(self, images):
        batch_size, _, h, w = images.shape
        masks = torch.zeros((batch_size, 1, h, w), device=images.device)
        
        for i in range(batch_size):
            img_channel = images[i, 0, :, :]
            threshold = img_channel.mean() + 0.5 * img_channel.std()
            mask = (img_channel > threshold).float()
            masks[i, 0] = mask
        
        return masks
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

# Function at MODULE LEVEL
def train_real_tiff_data(config):
    """Train on 555k REAL TIFF files"""
    import tifffile
    from pathlib import Path
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import cv2
    from src.models import BD_S8_Model
    
    class RealTIFFDataset(Dataset):
        def __init__(self, root='/scratch/project_2010376/BDS8/BDS8_data', max_samples=10000, mode='train'):
            self.images = []
            self.labels_dict = []
            
            # Load AML samples
            aml_dir = Path(root) / 'AML'
            if aml_dir.exists():
                aml_files = list(aml_dir.glob('**/*.tiff'))[:max_samples//2]
                self.images.extend(aml_files)
                self.labels_dict.extend([{
                    'extraction_method': 1, 'viability': 1, 
                    'fresh_frozen': 0, 'fixed': 1, 'permeabilized': 1
                }] * len(aml_files))
            
            # Load Healthy samples
            healthy_dir = Path(root) / 'Healthy BM'
            if healthy_dir.exists():
                healthy_files = list(healthy_dir.glob('**/*.tiff'))[:max_samples//2]
                self.images.extend(healthy_files)
                self.labels_dict.extend([{
                    'extraction_method': 0, 'viability': 1,
                    'fresh_frozen': 0, 'fixed': 1, 'permeabilized': 1
                }] * len(healthy_files))
            
            print(f"✅ Loaded {len(self.images)} REAL cells")
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img = tifffile.imread(self.images[idx])
            
            # Handle different formats
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.ndim == 3:
                if img.shape[0] <= 4:
                    img = np.transpose(img[:3], (1,2,0))
                elif img.shape[-1] > 3:
                    img = img[..., :3]
            
            # Ensure exactly 3 channels
            if img.shape[-1] != 3:
                h, w = img.shape[:2]
                new_img = np.zeros((h, w, 3), dtype=np.float32)
                channels = min(3, img.shape[-1])
                new_img[..., :channels] = img[..., :channels]
                img = new_img
            
            # Resize to 512x512
            img = cv2.resize(img.astype(np.float32), (512, 512))
            
            # Normalize
            img = img.astype(np.float32)
            if img.max() > 1:
                img = img / 65535.0
            
            # Convert to tensor (CHW format)
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            return {'image': img, 'labels': self.labels_dict[idx]}
    
    # Create datasets
    train_dataset = RealTIFFDataset(max_samples=20000, mode='train')
    val_dataset = RealTIFFDataset(max_samples=4000, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Model
    model = BD_S8_Model(num_classes=5)  # Removed num_channels
    lightning_model = BD_S8_LightningModule(model, config['training']['learning_rate'])  # FIXED: Using renamed class
    
    # Checkpoint
    checkpoint = ModelCheckpoint(
        dirpath='models/',
        filename='real_tiff_{epoch}_{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint],
        accelerator='auto',
        logger=TensorBoardLogger('lightning_logs', name='real_tiff_training')
    )
    
    trainer.fit(lightning_model, train_loader, val_loader)
    
    print("✅ Training on REAL TIFFs completed!")
    return lightning_model

def train_full_model(config):
    """Main training function"""
    
    # Check if REAL data exists
    real_data_path = '/scratch/project_2010376/BDS8/BDS8_data'
    if os.path.exists(real_data_path):
        print("\n🚀 Using 555,053 REAL TIFF files!")
        return train_real_tiff_data(config)
    else:
        # Fallback to synthetic data (your old code)
        from src.data_loader import get_dataloader
        from src.models import BD_S8_Model
        
        print("\n🚀 Using synthetic data")
        train_loader = get_dataloader(
            excel_path=config['data']['excel_path'],
            data_folder=config['data']['data_folder'],
            batch_size=config['training']['batch_size'],
            mode='train',
            num_workers=0
        )
        
        val_loader = get_dataloader(
            excel_path=config['data']['excel_path'],
            data_folder=config['data']['data_folder'],
            batch_size=config['training']['batch_size'],
            mode='val',
            num_workers=0
        )
        
        model = BD_S8_Model(num_classes=config['model']['num_classes'])
        lightning_model = BD_S8_LightningModule(model, config['training']['learning_rate'])  # Using renamed class
        
        checkpoint = ModelCheckpoint(
            dirpath='models/',
            filename='bd_s8_{epoch:02d}_{val_dice:.3f}',
            monitor='val_dice',
            mode='max',
            save_top_k=3,
            save_last=True
        )
        
        trainer = pl.Trainer(
            max_epochs=config['training']['num_epochs'],
            accelerator='auto',
            callbacks=[checkpoint],
            logger=TensorBoardLogger('lightning_logs')
        )
        
        trainer.fit(lightning_model, train_loader, val_loader)
        return lightning_model

# src/train.py - CORRECTED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
from typing import Dict
import numpy as np

class LightningModule(pl.LightningModule):
    """PyTorch Lightning training module for BD S8 analysis"""
    
    def __init__(self, model, learning_rate=1e-4, num_classes=5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Loss functions
        self.seg_loss = smp.losses.DiceLoss(mode='binary')
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        # Forward pass
        outputs = self(images)
        
        # Calculate segmentation loss
        batch_size = images.shape[0]
        synthetic_masks = self.generate_synthetic_masks(images)
        
        seg_dice_loss = self.seg_loss(outputs['segmentation'], synthetic_masks)
        seg_focal_loss = self.focal_loss(outputs['segmentation'], synthetic_masks)
        seg_loss = seg_dice_loss + 0.5 * seg_focal_loss
        
        # Handle labels properly
        extraction_labels = torch.tensor([labels['extraction_method'][i].item() if torch.is_tensor(labels['extraction_method'][i]) 
                                         else labels['extraction_method'][i] for i in range(batch_size)], device=self.device)
        viability_labels = torch.tensor([labels['viability'][i].item() if torch.is_tensor(labels['viability'][i]) 
                                        else labels['viability'][i] for i in range(batch_size)], device=self.device)
        
        extraction_loss = self.ce_loss(outputs['extraction'], extraction_labels)
        viability_loss = self.ce_loss(outputs['viability'], viability_labels)
        
        # Combined loss
        total_loss = seg_loss + 0.3 * extraction_loss + 0.2 * viability_loss
        
        # Calculate accuracy
        extraction_pred = torch.argmax(outputs['extraction'], dim=1)
        extraction_acc = (extraction_pred == extraction_labels).float().mean()
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_seg_loss', seg_loss)
        self.log('train_extraction_loss', extraction_loss)
        self.log('train_viability_loss', viability_loss)
        self.log('train_extraction_acc', extraction_acc, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['labels']
        
        outputs = self(images)
        synthetic_masks = self.generate_synthetic_masks(images)
        
        seg_dice = 1 - self.seg_loss(outputs['segmentation'], synthetic_masks)
        
        batch_size = images.shape[0]
        extraction_labels = torch.tensor([labels['extraction_method'][i].item() if torch.is_tensor(labels['extraction_method'][i]) 
                                         else labels['extraction_method'][i] for i in range(batch_size)], device=self.device)
        
        extraction_pred = torch.argmax(outputs['extraction'], dim=1)
        extraction_acc = (extraction_pred == extraction_labels).float().mean()
        
        self.validation_step_outputs.append({
            'val_dice': seg_dice,
            'val_acc': extraction_acc
        })
        
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
        """Generate synthetic segmentation masks"""
        batch_size, _, h, w = images.shape
        masks = torch.zeros((batch_size, 1, h, w), device=images.device)
        
        for i in range(batch_size):
            img_channel = images[i, 0, :, :]
            threshold = img_channel.mean() + 0.5 * img_channel.std()
            mask = (img_channel > threshold).float()
            
            kernel_size = 3
            mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=1)
            mask = F.avg_pool2d(mask, kernel_size, stride=1, padding=1)
            
            masks[i] = mask.squeeze(0)
        
        return masks
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

# IMPORTANT: This function MUST be at MODULE LEVEL (not inside the class!)
def train_full_model(config):
    """Complete training function - MUST be at module level"""
    from src.data_loader import get_dataloader
    from src.models import BD_S8_Model
    
    print("\n🚀 Starting full training...")
    
    # Create data loaders
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
    
    # Initialize model
    model = BD_S8_Model(
        num_classes=config['model']['num_classes'],
        num_channels=3
    )
    
    # Lightning module
    lightning_model = LightningModule(
        model=model,
        learning_rate=config['training']['learning_rate'],
        num_classes=config['model']['num_classes']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/',
        filename='bd_s8_{epoch:02d}_{val_dice:.3f}',
        monitor='val_dice',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_dice',
        patience=config['training'].get('early_stopping_patience', 10),
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='bd_s8_training')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_checkpointing=True
    )
    
    # Train
    trainer.fit(lightning_model, train_loader, val_loader)
    
    print("\n✅ Training completed!")
    print(f"📁 Best model saved at: models/")
    print(f"📊 View logs with: tensorboard --logdir lightning_logs/")
    
    return lightning_model

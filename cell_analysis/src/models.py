# src/models.py - FIXED VERSION
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm

class BD_S8_Model(nn.Module):
    """Multi-task model for BD S8 analysis"""
    
    def __init__(self, num_classes=5, encoder_name="efficientnet-b4"):
        super().__init__()
        
        # Fixed: num_channels is always 3 for your data
        num_channels = 3
        
        # Segmentation branch
        self.segmentation = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,  # No internet, no pretrained
            in_channels=num_channels,
            classes=1,
            activation=None
        )
        
        # Classification backbone
        self.encoder = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            in_chans=num_channels,
            num_classes=0
        )
        encoder_dim = self.encoder.num_features
        
        # Multi-task heads
        self.extraction_head = nn.Sequential(
            nn.Linear(encoder_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )
        self.viability_head = nn.Sequential(
            nn.Linear(encoder_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )
        self.cell_type_head = nn.Sequential(
            nn.Linear(encoder_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        seg_mask = self.segmentation(x)
        features = self.encoder(x)
        return {
            "segmentation": torch.sigmoid(seg_mask),
            "extraction": self.extraction_head(features),
            "viability": self.viability_head(features),
            "cell_type": self.cell_type_head(features)
        }

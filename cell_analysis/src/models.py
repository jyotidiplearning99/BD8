import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
from typing import Dict

class BD_S8_Model(nn.Module):
    """Multi-task model for BD S8 analysis"""

    def __init__(
        self,
        num_classes: int = 5,
        num_channels: int = 3,
        encoder_name: str = "efficientnet-b4",
    ):
        super().__init__()

        # Segmentation (no pretrained weights -> no internet)
        self.segmentation = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=num_channels,
            classes=1,
            activation=None,
        )

        # Classification backbone (no pretrained -> no internet)
        self.encoder = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            in_chans=num_channels,
            num_classes=0,  # feature extractor
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_mask = self.segmentation(x)
        features = self.encoder(x)
        return {
            "segmentation": torch.sigmoid(seg_mask),
            "extraction": self.extraction_head(features),
            "viability": self.viability_head(features),
            "cell_type": self.cell_type_head(features),
        }

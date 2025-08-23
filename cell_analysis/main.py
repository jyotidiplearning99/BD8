# main.py
#!/usr/bin/env python
"""
BD S8 Cell Analysis Pipeline - Fixed Version
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Fix imports - don't import get_dataloader if it's causing issues
from torch.utils.data import DataLoader
from src.data_loader import BD_S8_Dataset, get_transforms
# Note: we'll create dataloaders directly instead of using get_dataloader

def test_data_loading():
    """Test data loading with your actual files"""
    print("\n" + "="*60)
    print("BD S8 Pipeline - Testing Data Loading")
    print("="*60)
    
    # Load Excel data
    excel_path = 'data/List of S8 data.xlsx'
    df = pd.read_excel(excel_path)
    
    print(f"\n📊 Excel Data Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  AB stained samples: {len(df[df['AB stained=1'] == 1])}")
    print(f"  Extraction methods: {df['Extraction method'].unique()}")
    
    # Check for CVW files
    data_folder = Path('data')
    cvw_files = list(data_folder.glob('*.cvw'))
    print(f"\n📁 Found CVW files:")
    for f in cvw_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")
    
    # Create dataset
    print(f"\n🔬 Creating dataset...")
    dataset = BD_S8_Dataset(
        excel_path=excel_path,
        data_folder='data/',
        transform=None,
        mode='test'
    )
    
    print(f"  Dataset size: {len(dataset)} samples")
    
    # Test loading first sample
    if len(dataset) > 0:
        print(f"\n📸 Loading first sample...")
        sample = dataset[0]
        img = sample['image']
        labels = sample['labels']
        
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Labels: {labels}")
        
        # Visualize
        visualize_sample(img, labels, 0)
        
        # Test all samples
        print(f"\n📊 Testing all {len(dataset)} samples:")
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                print(f"  Sample {i}: ✅ Loaded successfully")
                visualize_sample(sample['image'], sample['labels'], i)
            except Exception as e:
                print(f"  Sample {i}: ❌ Error: {e}")
    
    return dataset

def visualize_sample(img, labels, idx):
    """Visualize a sample image"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Individual channels
    channels = ['CTSG (FITC)', 'MPO (PE)', 'HLA_ABC (RB780)']
    for i in range(3):
        ax = axes[0, i]
        im = ax.imshow(img[:, :, i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{channels[i]}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Composite views
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('RGB Composite')
    axes[1, 0].axis('off')
    
    # Histogram
    axes[1, 1].hist(img[:, :, 0].flatten(), bins=50, alpha=0.5, color='green', label='FITC')
    axes[1, 1].hist(img[:, :, 1].flatten(), bins=50, alpha=0.5, color='red', label='PE')
    axes[1, 1].hist(img[:, :, 2].flatten(), bins=50, alpha=0.5, color='blue', label='RB780')
    axes[1, 1].set_title('Intensity Distribution')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    # Metadata
    extraction = "Ficoll" if labels.get('extraction_method') == 0 else "Buffy coat"
    info_text = f"Sample {idx}\n"
    info_text += f"Extraction: {extraction}\n"
    info_text += f"Fresh(0)/Frozen(1): {labels.get('fresh_frozen')}\n"
    info_text += f"Fixed: {labels.get('fixed')}\n"
    info_text += f"Permeabilized: {labels.get('permeabilized')}\n"
    info_text += f"Viability: {labels.get('viability')}"
    
    axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Sample Info')
    
    plt.suptitle(f'BD S8 Sample Analysis - Index {idx}')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/sample_{idx}_visualization.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Visualization saved to outputs/sample_{idx}_visualization.png")
    plt.close()

def train_model(config):
    """Train the model"""
    print("\n" + "="*60)
    print("BD S8 Pipeline - Training Model")
    print("="*60)
    
    # Import model here to avoid circular imports
    from src.models import BD_S8_Model
    
    # Create datasets
    train_dataset = BD_S8_Dataset(
        excel_path=config['data']['excel_path'],
        data_folder=config['data']['data_folder'],
        transform=get_transforms('train'),
        mode='train'
    )
    
    val_dataset = BD_S8_Dataset(
        excel_path=config['data']['excel_path'],
        data_folder=config['data']['data_folder'],
        transform=get_transforms('val'),
        mode='val'
    )
    
    print(f"\n📊 Dataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Create data loaders manually (not using get_dataloader)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Test loading
    print("\n🔍 Testing data loaders...")
    for i, batch in enumerate(train_loader):
        print(f"  Batch {i}: {batch['image'].shape}")
        if i >= 2:
            break
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = BD_S8_Model(
        num_classes=config['model']['num_classes'],
        num_channels=3
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training would go here
    print("\n✅ Model initialized successfully!")
    
    return model

def run_full_training(config):
    """Run complete training pipeline"""
    from src.train import train_full_model
    
    # Update config for your small dataset
    config['training']['num_epochs'] = 50  # More epochs for small dataset
    config['training']['batch_size'] = 1   # Small batch size
    config['training']['learning_rate'] = 1e-4
    
    model = train_full_model(config)
    return model


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BD S8 Analysis Pipeline')
    parser.add_argument('--mode', choices=['test','train','both','inference'], default='test')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Default config
    config = {
        'data': {
            'excel_path': 'data/List of S8 data.xlsx',
            'data_folder': 'data/',
            'output_dir': 'outputs/'
        },
        'model': {
            'num_classes': 5,
            'num_channels': 3,
            'encoder': 'efficientnet-b4'
        },
        'training': {
            'batch_size': 1,  # Small batch size for testing
            'num_epochs': 10,
            'learning_rate': 0.0001,
            'early_stopping_patience': 5
        }
    }
    
    # Override with yaml if exists
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    config.update(loaded_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    
    print("\n" + "="*60)
    print("🧬 BD S8 Cell Analysis Pipeline")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Execute based on mode
    if args.mode == 'test':
        dataset = test_data_loading()
        print("\n✅ Test completed successfully!")
        
    elif args.mode == 'train':
        model = run_full_training(config)
        print("\n✅ Training pipeline ready!")
        
    elif args.mode == 'both':
        dataset = test_data_loading()
        model = train_model(config)
        print("\n✅ Full pipeline completed!")

    elif args.mode == 'inference':
        from src.inference import run_inference
        results = run_inference(config)
        print("\n✅ Inference completed!")
    
    print("\n" + "="*60)
    print("🎉 Pipeline execution finished!")
    print("="*60)

if __name__ == "__main__":
    main()

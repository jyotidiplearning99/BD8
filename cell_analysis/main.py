# main.py - Updated for REAL TIFF data
#!/usr/bin/env python
"""BD S8 Cell Analysis Pipeline - REAL Data Version"""

import os
import yaml
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='BD S8 Analysis Pipeline')
    parser.add_argument('--mode', choices=['train', 'inference'], default='train')
    parser.add_argument('--use-real-data', action='store_true', default=True,
                       help='Use 555k REAL TIFF data instead of synthetic')
    parser.add_argument('--config', default='config/config.yaml')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Default values
    config.setdefault('data', {}).setdefault('use_real_data', True)
    config.setdefault('training', {}).setdefault('batch_size', 32)
    
    print("\n" + "="*60)
    print("🧬 BD S8 Cell Analysis Pipeline")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Using: {'555k REAL TIFF data' if config['data']['use_real_data'] else 'Synthetic data'}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if args.mode == 'train':
        if config['data']['use_real_data']:
            from src.train import train_real_tiff_data
            model = train_real_tiff_data(config)
        else:
            from src.train import train_full_model
            model = train_full_model(config)
        print("✅ Training completed!")
        
    elif args.mode == 'inference':
        from src.inference import run_inference
        results = run_inference(config)
        print("✅ Inference completed!")

if __name__ == "__main__":
    main()

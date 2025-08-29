# main.py - PRODUCTION VERSION
#!/usr/bin/env python
"""BD S8 Cell Analysis Pipeline - Production Version"""

import os
import argparse
import yaml
import torch

def main():
    parser = argparse.ArgumentParser(description='BD S8 Analysis Pipeline')
    parser.add_argument('--mode', choices=['test', 'train', 'both', 'inference'], 
                       default='train', help='Execution mode')
    parser.add_argument('--config', default='config/config.yaml')
    
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
            'batch_size': 8,
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'early_stopping_patience': 10
        }
    }
    
    # Load config if exists
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    config.update(loaded_config)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    print("\n" + "="*60)
    print("🧬 BD S8 Cell Analysis Pipeline")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if args.mode == 'test':
        # Your existing test code
        from main import test_data_loading
        dataset = test_data_loading()
        print("✅ Test completed!")
        
    elif args.mode == 'train':
        from src.train import train_full_model
        model = train_full_model(config)
        print("✅ Training completed!")
        
    elif args.mode == 'both':
        # Both test and train
        from main import test_data_loading
        from src.train import train_full_model
        dataset = test_data_loading()
        model = train_full_model(config)
        print("✅ Full pipeline completed!")
        
    elif args.mode == 'inference':
        from src.inference import run_inference
        results = run_inference(config)
        print("✅ Inference completed!")

if __name__ == "__main__":
    main()

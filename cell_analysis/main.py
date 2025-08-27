# main.py
import argparse
import yaml
import torch
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['test','train','both','inference'], default='train')
    parser.add_argument('--config', default='config/config.yaml')
    
    args = parser.parse_args()
    
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
            'batch_size': 32,  # Larger batch for real data
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
    print("🧬 BD S8 AML Classification - REAL DATA")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if args.mode == 'train':
        from src.train import train_full_model
        model = train_full_model(config)
        print("✅ Training completed!")
        
    elif args.mode == 'inference':
        from src.inference import run_inference
        results = run_inference(config)
        print("✅ Inference completed!")

if __name__ == "__main__":
    main()

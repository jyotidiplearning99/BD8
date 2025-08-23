# src/inference.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import os

class InferencePipeline:
    """Inference pipeline for BD S8 models"""
    
    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.model_path = model_path
        
        # Import here to avoid circular imports
        from src.models import BD_S8_Model
        from src.train import LightningModule
        
        # Initialize model architecture
        base_model = BD_S8_Model(
            num_classes=config["model"]["num_classes"],
            num_channels=config["model"]["num_channels"],
            encoder_name=config["model"]["encoder"]
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model weights from checkpoint
        if 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    # Remove 'model.' prefix
                    new_key = key[6:]
                    state_dict[new_key] = value
            base_model.load_state_dict(state_dict)
        else:
            # Direct state dict
            base_model.load_state_dict(checkpoint)
        
        self.model = base_model
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
    
    def process_all(self):
        """Process all samples in the dataset"""
        from src.data_loader import BD_S8_Dataset, get_transforms
        
        # Create dataset
        ds = BD_S8_Dataset(
            excel_path=self.config["data"]["excel_path"],
            data_folder=self.config["data"]["data_folder"],
            transform=get_transforms("val"),
            mode="test"
        )
        
        # Create dataloader
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        
        results = []
        with torch.no_grad():
            for i, sample in enumerate(dl):
                img = sample["image"]
                labels = sample["labels"]
                
                # Run model
                out = self.model(img)
                
                # Process predictions
                extraction_pred = torch.argmax(out["extraction"], dim=1).item()
                viability_pred = torch.argmax(out["viability"], dim=1).item()
                cell_type_pred = torch.argmax(out["cell_type"], dim=1).item()
                seg_mean = out["segmentation"].mean().item()
                
                # Get actual labels
                actual_extraction = labels["extraction_method"][0].item() if torch.is_tensor(labels["extraction_method"]) else labels["extraction_method"][0]
                actual_viability = labels["viability"][0].item() if torch.is_tensor(labels["viability"]) else labels["viability"][0]
                
                result = {
                    "idx": i,
                    "segmentation_mean": seg_mean,
                    "extraction_pred": "Ficoll" if extraction_pred == 0 else "Buffy coat",
                    "extraction_actual": "Ficoll" if actual_extraction == 0 else "Buffy coat",
                    "extraction_correct": extraction_pred == actual_extraction,
                    "viability_pred": "Viable" if viability_pred == 1 else "Non-viable",
                    "viability_actual": "Viable" if actual_viability == 1 else "Non-viable",
                    "viability_correct": viability_pred == actual_viability,
                    "cell_type_pred": cell_type_pred,
                }
                
                results.append(result)
                
                print(f"Sample {i}: Extraction={result['extraction_pred']} (Actual: {result['extraction_actual']}), "
                      f"Viability={result['viability_pred']} (Actual: {result['viability_actual']})")
        
        return results

def run_inference(config):
    """Run inference with the trained model"""
    import os
    
    # Find the best checkpoint
    model_path = "models/last.ckpt"
    
    if not os.path.exists(model_path):
        # Try to find any checkpoint
        from pathlib import Path
        model_files = list(Path('models').glob('*.ckpt'))
        if model_files:
            model_path = str(model_files[0])
            print(f"📊 Using model: {model_path}")
        else:
            print("❌ No trained model found. Please train first.")
            return None
    else:
        print(f"📊 Loading model from: {model_path}")
    
    # Create inference pipeline
    pipeline = InferencePipeline(model_path, config)
    results = pipeline.process_all()
    
    # Calculate accuracy
    if results:
        extraction_correct = sum(r['extraction_correct'] for r in results)
        viability_correct = sum(r['viability_correct'] for r in results)
        total = len(results)
        
        print(f"\n📊 Inference Results Summary:")
        print(f"  Total samples: {total}")
        print(f"  Extraction accuracy: {extraction_correct}/{total} ({100*extraction_correct/total:.1f}%)")
        print(f"  Viability accuracy: {viability_correct}/{total} ({100*viability_correct/total:.1f}%)")
        
        # Save results to CSV
        df = pd.DataFrame(results)
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/inference_results.csv", index=False)
        print(f"\n📁 Results saved to outputs/inference_results.csv")
    
    return results

# src/inference.py
import torch
import numpy as np
from pathlib import Path

def run_inference(config):
    from src.data_loader import BD_S8_RealDataset, get_transforms
    from src.models import BD_S8_Model
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path('models/last.ckpt')
    if not model_path.exists():
        model_files = list(Path('models').glob('*.ckpt'))
        model_path = model_files[0] if model_files else None
    
    if not model_path:
        print("❌ No model found")
        return None
    
    print(f"📊 Loading: {model_path}")
    
    model = BD_S8_Model(num_classes=2)
    ckpt = torch.load(model_path, map_location=device)
    
    if "state_dict" in ckpt:
        state_dict = {k.replace("model.", ""): v 
                      for k,v in ckpt["state_dict"].items() 
                      if k.startswith("model.")}
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    test_dataset = BD_S8_RealDataset(
        data_root=config.get('data_root', '/scratch/project_2010376/BDS8/BDS8_data'),
        sample_types=["AML", "Healthy BM"],
        max_cells_per_type=1000,
        transform=get_transforms("val"),
        mode="test"
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"]
            
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Test Accuracy: {accuracy:.3f}")
    
    return all_preds, all_labels

# test_simple.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing BD S8 Pipeline...")

# Test 1: Basic imports
try:
    from src.data_loader import BD_S8_Dataset, get_transforms, get_dataloader
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Load dataset
try:
    dataset = BD_S8_Dataset(
        excel_path='data/List of S8 data.xlsx',
        data_folder='data/',
        transform=None,
        mode='test'
    )
    print(f"✅ Dataset created: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Dataset creation failed: {e}")
    exit(1)

# Test 3: Load a sample
try:
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"✅ Sample loaded: Image shape {sample['image'].shape}")
        print(f"   Labels: {list(sample['labels'].keys())}")
except Exception as e:
    print(f"❌ Sample loading failed: {e}")
    exit(1)

# Test 4: Create dataloader
try:
    dataloader = get_dataloader(
        excel_path='data/List of S8 data.xlsx',
        data_folder='data/',
        batch_size=2,
        mode='test',
        num_workers=0
    )
    print(f"✅ Dataloader created")
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        print(f"✅ Batch {i}: {batch['image'].shape}")
        if i >= 1:  # Test 2 batches
            break
except Exception as e:
    print(f"❌ Dataloader failed: {e}")
    exit(1)

print("\n🎉 All tests passed!")

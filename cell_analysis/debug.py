# debug.py
import sys
import os

print("=" * 50)
print("BD S8 Pipeline Debug Script")
print("=" * 50)

# 1. Check Python version
print(f"\n1. Python version: {sys.version}")

# 2. Check current directory
print(f"\n2. Current directory: {os.getcwd()}")

# 3. Check if src directory exists
print(f"\n3. Checking src directory:")
if os.path.exists('src'):
    print("   ✅ src/ directory exists")
    src_files = os.listdir('src')
    print(f"   Files in src/: {src_files}")
else:
    print("   ❌ src/ directory NOT found")

# 4. Check if data directory exists
print(f"\n4. Checking data directory:")
if os.path.exists('data'):
    print("   ✅ data/ directory exists")
    data_files = os.listdir('data')
    print(f"   Files in data/: {data_files}")
else:
    print("   ❌ data/ directory NOT found")

# 5. Try importing modules
print(f"\n5. Testing imports:")
try:
    import pandas as pd
    print("   ✅ pandas imported")
except ImportError as e:
    print(f"   ❌ pandas import failed: {e}")

try:
    import torch
    print(f"   ✅ torch imported (version {torch.__version__})")
except ImportError as e:
    print(f"   ❌ torch import failed: {e}")

try:
    import cv2
    print("   ✅ cv2 imported")
except ImportError as e:
    print(f"   ❌ cv2 import failed: {e}")

try:
    import albumentations
    print("   ✅ albumentations imported")
except ImportError as e:
    print(f"   ❌ albumentations import failed: {e}")

# 6. Try importing our module
print(f"\n6. Testing BD_S8_Dataset import:")
try:
    from src.data_loader import BD_S8_Dataset
    print("   ✅ BD_S8_Dataset imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("   Trying to diagnose...")
    
    # Try direct import
    try:
        sys.path.insert(0, 'src')
        import data_loader
        print("   ✅ data_loader module found")
        print(f"   Available functions: {dir(data_loader)}")
    except Exception as e2:
        print(f"   ❌ Direct import also failed: {e2}")

# 7. Test Excel file
print(f"\n7. Testing Excel file:")
try:
    import pandas as pd
    df = pd.read_excel('data/List of S8 data.xlsx')
    print(f"   ✅ Excel loaded: {len(df)} rows")
    print(f"   Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
except Exception as e:
    print(f"   ❌ Excel loading failed: {e}")

print("\n" + "=" * 50)
print("Debug complete!")

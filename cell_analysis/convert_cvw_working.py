# save as convert_cvw_working.py
import numpy as np
import tifffile
from pathlib import Path

def convert_cvw_to_tiff(cvw_path, output_path):
    """Convert CVW to TIFF if it can be read"""
    
    # If you found that CVW files can be read somehow
    # Perhaps they're actually TIFFs with wrong extension?
    try:
        # Method 1: Direct read with tifffile
        print(f"Attempting to read {cvw_path}...")
        image = tifffile.imread(cvw_path)
        print(f"✅ Successfully read! Shape: {image.shape}, dtype: {image.dtype}")
        
        # Save as proper TIFF
        tifffile.imwrite(output_path, image)
        print(f"✅ Saved to {output_path}")
        return True
        
    except Exception as e1:
        print(f"tifffile failed: {e1}")
        
        # Method 2: Read as binary and interpret
        try:
            with open(cvw_path, 'rb') as f:
                # Skip any header bytes if needed
                # f.seek(1024)  # Uncomment if there's a header
                
                # Read the rest as image data
                data = np.fromfile(f, dtype=np.uint16)
                
                # Reshape to expected dimensions
                # Adjust these based on your microscopy setup
                possible_shapes = [
                    (2048, 2048),
                    (4096, 4096),
                    (1024, 1024),
                ]
                
                for shape in possible_shapes:
                    if data.size == np.prod(shape):
                        image = data.reshape(shape)
                        print(f"✅ Reshaped to {shape}")
                        tifffile.imwrite(output_path, image)
                        return True
                        
        except Exception as e2:
            print(f"Binary read failed: {e2}")
    
    return False

# Convert your files
data_dir = Path("data")

# Convert PS1.cvw
if convert_cvw_to_tiff(data_dir / "PS1.cvw", data_dir / "PS1.tiff"):
    print("PS1.cvw converted successfully!")

# Convert Full stain imaging
if convert_cvw_to_tiff(
    data_dir / "Full stain imaging EXP_41.cvw", 
    data_dir / "Full_stain_EXP_41.tiff"
):
    print("Full stain imaging converted successfully!")

# utils/convert_cvw.py
"""
Convert CVW files to TIFF without Java dependencies
"""

import subprocess
import os
from pathlib import Path
import numpy as np
import tifffile

def convert_cvw_with_fiji(cvw_path: str, output_dir: str):
    """
    Convert CVW using Fiji (if installed)
    """
    output_path = os.path.join(output_dir, Path(cvw_path).stem + '.tif')
    
    # Fiji macro script
    macro = f"""
    run("Bio-Formats Importer", "open=[{cvw_path}] autoscale color_mode=Composite rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
    saveAs("Tiff", "{output_path}");
    close();
    """
    
    # Save macro
    macro_path = "convert_macro.ijm"
    with open(macro_path, 'w') as f:
        f.write(macro)
    
    # Run Fiji
    try:
        subprocess.run([
            "fiji", "--headless", "--run", macro_path
        ], check=True)
        print(f"✅ Converted {cvw_path} to {output_path}")
    except:
        print(f"❌ Fiji not found. Please install Fiji or use alternative method")
    
    return output_path

def convert_cvw_with_bftools(cvw_path: str, output_dir: str):
    """
    Convert using bftools command line (no Python Java needed)
    """
    output_path = os.path.join(output_dir, Path(cvw_path).stem + '.tif')
    
    try:
        # Use bfconvert tool
        subprocess.run([
            "bfconvert",
            "-overwrite",
            cvw_path,
            output_path
        ], check=True)
        print(f"✅ Converted using bftools")
    except:
        print(f"❌ bftools not found. Download from: https://www.openmicroscopy.org/bio-formats/downloads/")
    
    return output_path

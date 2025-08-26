#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

# CVW = BD Chorwave, header starts "CHOR"
CVW_MAGIC = b"CHOR"

def is_cvw(path: Path) -> bool:
    if path.suffix.lower() == ".cvw":
        return True
    try:
        with path.open("rb") as f:
            return f.read(4) == CVW_MAGIC
    except Exception:
        return False

def run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def bfconvert_available() -> bool:
    return shutil.which("bfconvert") is not None

def convert_with_bfconvert(src: Path, dst: Path, bigtiff=True, compression="LZW") -> bool:
    cmd = ["bfconvert", "-nogroup"]
    if bigtiff: cmd += ["-bigtiff"]
    if compression: cmd += ["-compression", compression]
    cmd += [str(src), str(dst)]
    code, out = run(cmd)
    if code != 0:
        print(f"[bfconvert FAIL] {src}\n{out.strip()}\n")
        return False
    print(f"[OK] {src.name} -> {dst}")
    return True

def scan_inputs(in_dir: Path) -> List[Path]:
    exts = {".cvw", ".czi", ".nd2", ".lif", ".oif", ".oir", ".tif", ".tiff"}
    return sorted(p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)

def write_manifest(cvw_files: List[Path], out_dir: Path, manifest_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_cvw", "size_gb", "suggested_output_tif"])
        for p in cvw_files:
            size_gb = p.stat().st_size / (1024**3)
            suggested = out_dir / (p.stem + ".tif")
            w.writerow([str(p), f"{size_gb:.2f}", str(suggested)])
    print(f"[INFO] Wrote manifest: {manifest_path}")

def pack_cvw(cvw_files: List[Path], tar_path: Path):
    import tarfile
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in cvw_files:
            tar.add(f, arcname=f.name)
    print(f"[INFO] Packed {len(cvw_files)} CVW file(s) into {tar_path}")

def ingest_tiffs(from_dir: Path, to_dir: Path):
    to_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in from_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            dest = to_dir / p.name
            if dest.exists():
                print(f"[SKIP] exists: {dest}")
                continue
            shutil.copy2(p, dest)
            n += 1
    print(f"[INGEST] Copied {n} TIFF(s) into {to_dir}")

def main():
    # Defaults assume this file is run from src/ and data lives one level up
    ap = argparse.ArgumentParser(description="Prepare/convert BD S8 files; CVW -> manifest/pack; ingest extracted TIFFs.")
    ap.add_argument("-i", "--input",  default="../data", help="Input dir (default: ../data)")
    ap.add_argument("-o", "--output", default="../data/converted", help="Output dir for TIFFs (default: ../data/converted)")
    ap.add_argument("--pack",   default=None, help="Make tar.gz of CVWs for Windows extraction (e.g., ../cvw_to_extract.tar.gz)")
    ap.add_argument("--ingest", default=None, help="Copy extracted TIFFs from this dir into --output")
    args = ap.parse_args()

    in_dir  = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ingest mode
    if args.ingest:
        ingest_tiffs(Path(args.ingest).resolve(), out_dir)
        return

    files = scan_inputs(in_dir)
    if not files:
        print(f"[INFO] No candidate files under {in_dir}")
        return

    have_bf = bfconvert_available()
    cvw_list, other_list = [], []
    for f in files:
        (cvw_list if is_cvw(f) else other_list).append(f)

    # Convert non-CVW (or just copy existing TIFFs)
    if other_list:
        if not have_bf:
            print("[WARN] bfconvert not found; will only copy existing TIFFs.")
        for src in other_list:
            if src.suffix.lower() in {".tif", ".tiff"}:
                dest = out_dir / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)
                    print(f"[COPY] {src} -> {dest}")
                else:
                    print(f"[SKIP] exists: {dest}")
            elif have_bf:
                dest = out_dir / (src.stem + ".tif")
                if dest.exists():
                    print(f"[SKIP] exists: {dest}")
                else:
                    convert_with_bfconvert(src, dest)

    # CVW: write manifest (+ optional pack)
    if cvw_list:
        manifest = in_dir / "CVW_EXTRACT_MANIFEST.csv"
        write_manifest(cvw_list, out_dir, manifest)
        print("\n[NEED WINDOWS EXTRACTION]")
        print("  • These are BD CVW (Chorwave) and can’t be converted here.")
        print("  • Use BD CellView™ Image Extractor on Windows to export TIFFs.")
        print(f"  • See manifest: {manifest}\n")
        if args.pack:
            pack_cvw(cvw_list, Path(args.pack).resolve())
            print("Next steps:")
            print(f"  1) Copy the tar.gz to Windows and extract it.")
            print("  2) Run BD CellView Image Extractor on the .cvw to a folder (e.g., extracted_tiffs/).")
            print(f"  3) Back here: python {Path(__file__).name} --ingest /path/to/extracted_tiffs -o {out_dir}")

if __name__ == "__main__":
    main()

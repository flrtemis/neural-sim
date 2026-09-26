"""
prepare_model.py — Decimate an OBJ mesh and prepare it for browser loading.

Reads a high-poly OBJ, keeps every Nth face to reach target face count,
writes a decimated OBJ + corrected MTL with texture reference.
No external dependencies beyond Python stdlib.
"""

import os
import sys
import shutil
from pathlib import Path


def decimate_obj(src_obj: str, dst_obj: str, target_faces: int = 100_000):
    """Read OBJ, decimate by uniform face subsampling, write output."""

    print(f"  Reading {src_obj} ...")
    
    # First pass: count faces
    total_faces = 0
    with open(src_obj, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("f "):
                total_faces += 1

    if total_faces == 0:
        print("  ERROR: No faces found in OBJ")
        return

    # Calculate skip ratio
    keep_ratio = min(1.0, target_faces / total_faces)
    skip = max(1, int(1.0 / keep_ratio))
    actual_target = total_faces // skip

    print(f"  Original: {total_faces:,} faces")
    print(f"  Target: ~{actual_target:,} faces (keeping every {skip}th face)")

    # Second pass: collect all vertices, normals, UVs (we need all of them
    # since faces reference them by absolute index)
    # Then selectively write faces

    # Collect header lines (comments, mtllib, etc.)
    # and geometry data, then selectively output faces
    
    face_count = 0
    kept_faces = 0
    
    with open(src_obj, "r", encoding="utf-8", errors="replace") as fin, \
         open(dst_obj, "w", encoding="utf-8") as fout:
        
        for line in fin:
            stripped = line.strip()
            
            if stripped.startswith("f "):
                face_count += 1
                if face_count % skip == 0:
                    fout.write(line)
                    kept_faces += 1
            else:
                # Keep all non-face lines (vertices, normals, UVs, groups, etc.)
                fout.write(line)

    print(f"  Written: {kept_faces:,} faces to {dst_obj}")
    return kept_faces


def prepare_mtl(dst_dir: str, texture_filename: str):
    """Write a corrected MTL file that includes the texture reference."""
    mtl_path = os.path.join(dst_dir, "model.mtl")
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(f"""# Corrected MTL for Sage model
newmtl material_1
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.8
Ks 0.1 0.1 0.1
Ns 100
d 1.0
map_Kd {texture_filename}
""")
    print(f"  Written MTL: {mtl_path}")


def main():
    src_dir = r"c:\Users\l3ung\Pictures\Saved Pictures\3d_Ripper_Pro_v102\Downloads\studioscans\01- Girl.Body.Scan.Studio.13"
    dst_dir = r"c:\Users\l3ung\Desktop\ReverseEngineering\toolbox\neural-sim\static\model"

    src_obj = os.path.join(src_dir, "50968891d86149a48ec4781a6074f7d2.obj")
    src_texture = os.path.join(src_dir, "23e3114f322a437f80d76b053e95a7ab_RGB_Untitled.png")
    
    dst_obj = os.path.join(dst_dir, "sage.obj")
    dst_texture = os.path.join(dst_dir, "texture.png")

    # Create output directory
    os.makedirs(dst_dir, exist_ok=True)

    # Decimate
    print("Step 1: Decimating mesh...")
    decimate_obj(src_obj, dst_obj, target_faces=100_000)

    # Prepare MTL
    print("\nStep 2: Writing MTL...")
    prepare_mtl(dst_dir, "texture.png")

    # Copy texture
    print("\nStep 3: Copying texture...")
    if os.path.exists(src_texture):
        shutil.copy2(src_texture, dst_texture)
        size_mb = os.path.getsize(dst_texture) / 1024 / 1024
        print(f"  Texture copied: {size_mb:.1f} MB")
    else:
        print(f"  WARNING: Texture not found at {src_texture}")

    # Fix the mtllib reference in the OBJ (it says "model.mtl")
    # The OBJ already references "model.mtl" which matches what we wrote
    
    # Summary
    print("\n" + "=" * 50)
    print("Model prepared for browser loading:")
    for f in os.listdir(dst_dir):
        fpath = os.path.join(dst_dir, f)
        size = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()

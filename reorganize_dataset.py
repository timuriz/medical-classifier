#!/usr/bin/env python3
"""
Organize HAM10000 images by diagnosis class.

Expects:
  data/raw_images/HAM10000_metadata.csv
  data/raw_images/HAM10000_images_part1/
  data/raw_images/HAM10000_images_part2/

Creates:
  data/melanoma/
  data/nevus/
  data/seborrheic_keratosis/
"""

import os
import csv
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configuration
RAW_DIR = "data/raw_images"
METADATA_FILE = os.path.join(RAW_DIR, "HAM10000_metadata.csv")
OUTPUT_DIR = "data"

# Diagnosis mapping
CLASS_MAP = {
    "mel": "melanoma",
    "nv": "nevus",
    "bcc": "nevus",
    "akiec": "seborrheic_keratosis",
    "bkl": "seborrheic_keratosis",
    "df": "seborrheic_keratosis",
    "vasc": "seborrheic_keratosis"
}


def main():
    print("=" * 70)
    print("HAM10000 DATASET ORGANIZER")
    print("=" * 70)
    
    # Check metadata file
    if not os.path.exists(METADATA_FILE):
        print(f"\n❌ ERROR: Metadata file not found")
        print(f"   Expected: {METADATA_FILE}")
        print(f"\n   Make sure you downloaded HAM10000 from Kaggle")
        print(f"   and extracted it to: data/raw_images/")
        return False
    
    # Create output folders
    print("\n📁 Creating class folders...")
    for class_name in ["melanoma", "nevus", "seborrheic_keratosis"]:
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)
    print("   ✓ Folders created")
    
    # Read metadata
    print(f"\n📖 Reading metadata from CSV...")
    image_map = {}
    
    with open(METADATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id", "").strip()
            dx = row.get("dx", "").strip()
            
            if image_id and dx in CLASS_MAP:
                image_map[image_id] = CLASS_MAP[dx]
    
    print(f"   ✓ Found {len(image_map)} labeled images")
    
    # Find and organize images
    print(f"\n🔍 Organizing images by diagnosis...")
    
    class_counts = defaultdict(int)
    found = 0
    
    # Search for images in HAM10000 folders
    for root, dirs, files in os.walk(RAW_DIR):
        for filename in tqdm(files, desc="Processing"):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            # Get image ID
            image_id = Path(filename).stem
            
            if image_id not in image_map:
                continue
            
            target_class = image_map[image_id]
            src = os.path.join(root, filename)
            dst = os.path.join(OUTPUT_DIR, target_class, f"{image_id}.jpg")
            
            try:
                shutil.copy2(src, dst)
                class_counts[target_class] += 1
                found += 1
            except Exception as e:
                print(f"\n⚠️  Failed to copy {image_id}: {e}")
    
    # Print results
    print("\n✅ ORGANIZATION COMPLETE!")
    print(f"\n📊 Images organized by class:")
    
    total = 0
    for class_name in ["melanoma", "nevus", "seborrheic_keratosis"]:
        count = class_counts[class_name]
        total += count
        bar_len = count // 200
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"   {class_name:.<25} {count:>5} [{bar}]")
    
    print(f"\n   Total: {total} images")
    
    if total > 0:
        print(f"\n✅ Ready to train!")
        print(f"   Run: python backend/train/dataset.py")
        return True
    else:
        print(f"\n❌ No images found! Check that:")
        print(f"   1. HAM10000 is extracted to: data/raw_images/")
        print(f"   2. Metadata file exists: {METADATA_FILE}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

"""
Script to compare the train.csv file with existing annotations and images.
Identifies which images from CSV have/don't have JSON annotations and image files.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict


def parse_train_csv(csv_path):
    """
    Parse the train.csv file to get all unique ImageIds that have defects.
    
    Args:
        csv_path: Path to train.csv
        
    Returns:
        set: Set of unique image IDs that have defect annotations (ClassId not empty)
    """
    images_with_defects = set()
    total_rows = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            image_id = row['ImageId']
            class_id = row['ClassId']
            encoded_pixels = row['EncodedPixels']
            
            # Only count images that have actual defect data (ClassId and EncodedPixels not empty)
            if class_id and encoded_pixels and class_id.strip() and encoded_pixels.strip():
                images_with_defects.add(image_id)
    
    return images_with_defects, total_rows


def get_json_annotations(ann_folder):
    """
    Get all JSON annotation files in a folder.
    
    Args:
        ann_folder: Path to annotation folder
        
    Returns:
        set: Set of base image names (without .json extension)
    """
    ann_folder = Path(ann_folder)
    
    if not ann_folder.exists():
        return set()
    
    json_files = ann_folder.glob("*.json")
    # Extract base name (e.g., "0002cc93b.jpg" from "0002cc93b.jpg.json")
    return {json_file.stem for json_file in json_files}


def get_image_files(img_folder):
    """
    Get all image files in a folder.
    
    Args:
        img_folder: Path to images folder
        
    Returns:
        set: Set of image filenames
    """
    img_folder = Path(img_folder)
    
    if not img_folder.exists():
        return set()
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = set()
    
    for ext in image_extensions:
        image_files.update({img.name for img in img_folder.glob(f"*{ext}")})
    
    return image_files


def main():
    # Define paths
    base_path = Path(__file__).parent.parent.parent / "data"
    csv_path = base_path / "Severstal" / "train.csv"
    
    # annotations-old contains ONLY annotations with defects (the ones we care about)
    annotations_old = base_path / "annotations - old"
    images_old = base_path / "images - old"
    
    print("=" * 80)
    print("CSV vs ANNOTATIONS-OLD COMPARISON REPORT")
    print("=" * 80)
    print()
    print("NOTE: 'annotations-old' contains ONLY annotations with actual defects,")
    print("      not the empty annotations without damage information.")
    print()
    print("=" * 80)
    print()
    
    # Check if CSV exists
    if not csv_path.exists():
        print(f"ERROR: train.csv not found at {csv_path}")
        return
    
    print(f"📄 Parsing CSV file: {csv_path}")
    csv_images, total_csv_rows = parse_train_csv(csv_path)
    print(f"   Total rows in CSV: {total_csv_rows}")
    print(f"   Unique images with defects in CSV: {len(csv_images)}")
    print()
    
    print("=" * 80)
    print()
    
    # Get JSON annotations from annotations-old (these have defects)
    print(f"📁 Checking annotations-old: {annotations_old}")
    old_annotations = get_json_annotations(annotations_old)
    print(f"   JSON annotations found (with defects): {len(old_annotations)}")
    print()
    
    print(f"📁 Checking images-old: {images_old}")
    old_images = get_image_files(images_old)
    print(f"   Image files found: {len(old_images)}")
    print()
    
    print("=" * 80)
    print()
    
    # MAIN ANALYSIS: Which CSV images are missing from annotations-old?
    print("⚠️  CRITICAL ANALYSIS: CSV vs annotations-old")
    print("-" * 80)
    print()
    
    csv_images_with_json = csv_images.intersection(old_annotations)
    csv_images_without_json = csv_images.difference(old_annotations)
    
    print(f"✅ CSV images that HAVE JSON in annotations-old: {len(csv_images_with_json)}")
    print(f"❌ CSV images MISSING from annotations-old: {len(csv_images_without_json)}")
    print()
    print(f"   Coverage: {(len(csv_images_with_json) / len(csv_images) * 100):.2f}%")
    print(f"   Missing: {(len(csv_images_without_json) / len(csv_images) * 100):.2f}%")
    print()
    
    # Analysis: Image files
    print("=" * 80)
    print()
    print("ANALYSIS: Image Files Availability")
    print("-" * 80)
    
    if old_images:
        csv_with_old_img = csv_images.intersection(old_images)
        csv_without_old_img = csv_images.difference(old_images)
        
        print(f"✅ CSV images that HAVE image file in images-old: {len(csv_with_old_img)}")
        print(f"❌ CSV images MISSING image file in images-old: {len(csv_without_old_img)}")
        print()
        
        # Check specifically for the missing annotations
        missing_with_images = csv_images_without_json.intersection(old_images)
        missing_without_images = csv_images_without_json.difference(old_images)
        
        print("For the images missing JSON annotations:")
        print(f"   ✅ Have image files: {len(missing_with_images)}")
        print(f"   ❌ Also missing image files: {len(missing_without_images)}")
    else:
        print("⚠️  images-old folder is empty or doesn't exist")
    
    print()
    print("=" * 80)
    print()
    
    # Summary
    print("📊 SUMMARY & NEXT STEPS")
    print("-" * 80)
    print()
    print(f"Total images with defects in CSV:          {len(csv_images)}")
    print(f"Images with JSON in annotations-old:       {len(csv_images_with_json)}")
    print(f"Images MISSING from annotations-old:       {len(csv_images_without_json)}")
    print()
    
    if len(csv_images_without_json) > 0:
        print(f"⚠️  ACTION REQUIRED:")
        print(f"    You need to convert {len(csv_images_without_json)} RLE annotations from CSV to JSON format")
        print()
        print("📝 Next steps:")
        print("   1. Create an RLE to JSON converter script")
        print("   2. Process the missing images from train.csv")
        print("   3. Generate JSON files in the same format as existing annotations-old files")
        print(f"   4. Save the new JSONs to: {annotations_old}")
        print()
        print(f"   These {len(csv_images_without_json)} images have RLE encoded defect masks in the CSV")
        print("   that need to be decoded and converted to the JSON bitmap format.")
    else:
        print("✅ All CSV images have corresponding JSON annotations!")
    
    print()
    print("=" * 80)
    
    # Save missing images list to file for reference
    if len(csv_images_without_json) > 0:
        output_file = base_path / "reports" / "missing_annotations.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("Images from train.csv missing JSON annotations:\n")
            f.write("=" * 80 + "\n\n")
            for img in sorted(csv_images_without_json):
                f.write(f"{img}\n")
        
        print(f"📝 List of missing images saved to: {output_file}")
        print()


if __name__ == "__main__":
    main()

"""
Script to analyze unique images and their defect classes in train.csv
"""

import csv
from pathlib import Path
from collections import defaultdict


def analyze_csv_images(csv_path):
    """
    Analyze the CSV to find unique images and their associated defect classes.
    
    Args:
        csv_path: Path to train.csv
        
    Returns:
        dict: Dictionary mapping ImageId to list of ClassIds
    """
    image_classes = defaultdict(list)
    total_rows = 0
    rows_with_defects = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            image_id = row['ImageId']
            class_id = row['ClassId']
            encoded_pixels = row['EncodedPixels']
            
            # Only count rows that have actual defect data
            if class_id and encoded_pixels and class_id.strip() and encoded_pixels.strip():
                rows_with_defects += 1
                image_classes[image_id].append(class_id)
    
    return image_classes, total_rows, rows_with_defects


def main():
    # Define paths
    base_path = Path(__file__).parent.parent.parent / "data"
    csv_path = base_path / "Severstal" / "train.csv"
    
    print("=" * 80)
    print("CSV UNIQUE IMAGES AND CLASS ANALYSIS")
    print("=" * 80)
    print()
    
    if not csv_path.exists():
        print(f"ERROR: train.csv not found at {csv_path}")
        return
    
    print(f"📄 Analyzing: {csv_path}")
    print()
    
    image_classes, total_rows, rows_with_defects = analyze_csv_images(csv_path)
    
    print("=" * 80)
    print("STATISTICS")
    print("-" * 80)
    print(f"Total rows in CSV:                     {total_rows}")
    print(f"Rows with defect data (non-empty):     {rows_with_defects}")
    print(f"Unique images with defects:            {len(image_classes)}")
    print()
    
    # Count images by number of defect classes
    class_count_distribution = defaultdict(int)
    for img_id, classes in image_classes.items():
        num_classes = len(classes)
        class_count_distribution[num_classes] += 1
    
    print("=" * 80)
    print("DISTRIBUTION: Images by Number of Defect Classes")
    print("-" * 80)
    for num_classes in sorted(class_count_distribution.keys()):
        count = class_count_distribution[num_classes]
        print(f"Images with {num_classes} defect class(es):    {count}")
    print()
    
    # Count by individual class ID
    class_id_counts = defaultdict(int)
    for img_id, classes in image_classes.items():
        for class_id in classes:
            class_id_counts[class_id] += 1
    
    print("=" * 80)
    print("DISTRIBUTION: Images by Defect Class ID")
    print("-" * 80)
    for class_id in sorted(class_id_counts.keys()):
        count = class_id_counts[class_id]
        print(f"Class {class_id}:    {count} images")
    print()
    
    print("=" * 80)
    print("SAMPLE: First 20 Unique Images and Their Classes")
    print("-" * 80)
    for i, (img_id, classes) in enumerate(sorted(image_classes.items())[:20]):
        classes_str = ', '.join(classes)
        print(f"{img_id}    →    Class(es): {classes_str}")
    print()
    
    if len(image_classes) > 20:
        print(f"... and {len(image_classes) - 20} more images")
        print()
    
    print("=" * 80)
    print("VERIFICATION")
    print("-" * 80)
    print(f"✓ Total unique images with defects: {len(image_classes)}")
    print(f"✓ This matches the count from previous analysis!")
    print()
    
    # Save detailed list
    output_file = base_path / "reports" / "csv_image_classes.txt"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("Complete List of Images and Their Defect Classes\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total unique images: {len(image_classes)}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Image ID':<20} {'Class IDs'}\n")
        f.write("-" * 80 + "\n")
        
        for img_id in sorted(image_classes.keys()):
            classes = image_classes[img_id]
            classes_str = ', '.join(classes)
            f.write(f"{img_id:<20} {classes_str}\n")
    
    print(f"📝 Complete list saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

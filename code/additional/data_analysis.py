"""
Data Analysis Script for Severstal Steel Defect Dataset
Analyzes data integrity, class distribution, and annotation-image matching.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import pandas as pd


class SeverstalDataAnalyzer:
    """Analyzer for Severstal dataset integrity and statistics."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.annotations_dir = self.data_root / "annotations" / "ann_dmg"
        self.images_dir = self.data_root / "images" / "img_dmg"
        
        # Results storage
        self.annotation_files: Set[str] = set()
        self.image_files: Set[str] = set()
        self.class_distribution: Counter = Counter()
        self.defect_types: Set[str] = set()
        self.annotation_stats: Dict = {}
        self.integrity_report: Dict = {}
        
    def scan_files(self) -> None:
        """Scan annotation and image directories."""
        print("🔍 Scanning files...")
        
        # Scan annotations
        if self.annotations_dir.exists():
            for file_path in self.annotations_dir.glob("*.json"):
                # Extract base name (remove .jpg.json)
                base_name = file_path.name.replace(".jpg.json", "")
                self.annotation_files.add(base_name)
        
        # Scan images  
        if self.images_dir.exists():
            for file_path in self.images_dir.glob("*.jpg"):
                # Extract base name (remove .jpg)
                base_name = file_path.stem
                self.image_files.add(base_name)
                
        print(f"   Found {len(self.annotation_files)} annotation files")
        print(f"   Found {len(self.image_files)} image files")
    
    def check_file_matching(self) -> Dict[str, Set[str]]:
        """Check if all images have corresponding annotations and vice versa."""
        print("\n📋 Checking file matching...")
        
        # Find mismatches
        missing_annotations = self.image_files - self.annotation_files
        missing_images = self.annotation_files - self.image_files
        matched_files = self.annotation_files & self.image_files
        
        results = {
            'matched': matched_files,
            'missing_annotations': missing_annotations,
            'missing_images': missing_images
        }
        
        print(f"   ✅ Matched pairs: {len(matched_files)}")
        print(f"   ❌ Images missing annotations: {len(missing_annotations)}")
        print(f"   ❌ Annotations missing images: {len(missing_images)}")
        
        if missing_annotations:
            print(f"      Examples of images without annotations: {list(missing_annotations)[:5]}")
        if missing_images:
            print(f"      Examples of annotations without images: {list(missing_images)[:5]}")
            
        return results
    
    def analyze_annotations(self) -> Dict:
        """Analyze annotation content and class distribution."""
        print("\n🏷️  Analyzing annotations...")
        
        defect_classes = set()
        files_with_defects = 0
        files_without_defects = 0
        multi_class_files = 0
        class_combinations = Counter()
        
        total_files = len(self.annotation_files)
        processed = 0
        
        for base_name in self.annotation_files:
            annotation_path = self.annotations_dir / f"{base_name}.jpg.json"
            
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract defect classes from this file
                file_classes = set()
                for obj in data.get('objects', []):
                    class_title = obj.get('classTitle', '')
                    if class_title:
                        defect_classes.add(class_title)
                        file_classes.add(class_title)
                        self.class_distribution[class_title] += 1
                
                # Track file-level statistics
                if file_classes:
                    files_with_defects += 1
                    if len(file_classes) > 1:
                        multi_class_files += 1
                    # Track class combinations
                    class_combo = "+".join(sorted(file_classes))
                    class_combinations[class_combo] += 1
                else:
                    files_without_defects += 1
                    class_combinations['no_defect'] += 1
                
                processed += 1
                if processed % 1000 == 0:
                    print(f"   Progress: {processed}/{total_files} files processed")
                    
            except Exception as e:
                print(f"   ⚠️  Error processing {annotation_path}: {e}")
        
        self.defect_types = defect_classes
        
        results = {
            'total_annotations': total_files,
            'files_with_defects': files_with_defects,
            'files_without_defects': files_without_defects,
            'multi_class_files': multi_class_files,
            'unique_defect_classes': list(defect_classes),
            'class_distribution': dict(self.class_distribution),
            'class_combinations': dict(class_combinations.most_common(20))
        }
        
        print(f"   📊 Annotation Statistics:")
        print(f"      Total files: {total_files}")
        print(f"      Files with defects: {files_with_defects}")
        print(f"      Files without defects: {files_without_defects}")
        print(f"      Multi-class files: {multi_class_files}")
        print(f"      Unique defect classes: {len(defect_classes)}")
        print(f"      Defect classes found: {sorted(defect_classes)}")
        
        return results
    
    def analyze_class_distribution(self) -> None:
        """Analyze and display class distribution details."""
        print("\n📈 Class Distribution Analysis:")
        
        total_defect_instances = sum(self.class_distribution.values())
        
        for class_name, count in self.class_distribution.most_common():
            percentage = (count / total_defect_instances) * 100
            print(f"   {class_name}: {count} instances ({percentage:.2f}%)")
    
    def check_annotation_completeness(self) -> Dict:
        """Check if all annotations have required defect class information."""
        print("\n🔍 Checking annotation completeness...")
        
        valid_annotations = 0
        invalid_annotations = []
        empty_annotations = []
        
        for base_name in list(self.annotation_files)[:100]:  # Sample check first 100
            annotation_path = self.annotations_dir / f"{base_name}.jpg.json"
            
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                objects = data.get('objects', [])
                
                if not objects:
                    empty_annotations.append(base_name)
                    continue
                
                has_valid_class = False
                for obj in objects:
                    class_title = obj.get('classTitle', '')
                    if class_title and class_title.startswith('defect_'):
                        has_valid_class = True
                        break
                
                if has_valid_class:
                    valid_annotations += 1
                else:
                    invalid_annotations.append(base_name)
                    
            except Exception as e:
                print(f"   ⚠️  Error reading {annotation_path}: {e}")
                invalid_annotations.append(base_name)
        
        print(f"   ✅ Valid annotations (sample): {valid_annotations}/100")
        print(f"   ❌ Invalid annotations (sample): {len(invalid_annotations)}")
        print(f"   📝 Empty annotations (sample): {len(empty_annotations)}")
        
        if invalid_annotations:
            print(f"      Examples of invalid: {invalid_annotations[:5]}")
        if empty_annotations:
            print(f"      Examples of empty: {empty_annotations[:5]}")
        
        return {
            'valid_sample': valid_annotations,
            'invalid_sample': len(invalid_annotations),
            'empty_sample': len(empty_annotations),
            'invalid_examples': invalid_annotations[:10],
            'empty_examples': empty_annotations[:10]
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive data analysis report."""
        print("\n" + "="*60)
        print("🔬 SEVERSTAL DATASET ANALYSIS REPORT")
        print("="*60)
        
        # Run all analyses
        self.scan_files()
        file_matching = self.check_file_matching()
        annotation_analysis = self.analyze_annotations()
        self.analyze_class_distribution()
        completeness_check = self.check_annotation_completeness()
        
        # Create comprehensive report
        report = {
            'dataset_path': str(self.data_root),
            'timestamp': pd.Timestamp.now().isoformat(),
            'file_matching': {
                'total_matched': len(file_matching['matched']),
                'missing_annotations': len(file_matching['missing_annotations']),
                'missing_images': len(file_matching['missing_images']),
                'match_rate': len(file_matching['matched']) / max(len(self.image_files), len(self.annotation_files)) * 100
            },
            'annotation_analysis': annotation_analysis,
            'completeness_check': completeness_check,
            'recommendations': self.generate_recommendations(file_matching, annotation_analysis, completeness_check)
        }
        
        # Print summary
        print(f"\n📋 SUMMARY:")
        print(f"   Dataset Quality: {'✅ GOOD' if report['file_matching']['match_rate'] > 95 else '⚠️  NEEDS ATTENTION'}")
        print(f"   File Match Rate: {report['file_matching']['match_rate']:.2f}%")
        print(f"   Defect Classes: {len(annotation_analysis['unique_defect_classes'])}")
        print(f"   Multi-label Files: {annotation_analysis['multi_class_files']}")
        
        return report
    
    def generate_recommendations(self, file_matching: Dict, annotation_analysis: Dict, completeness_check: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # File matching recommendations
        if len(file_matching['missing_annotations']) > 0:
            recommendations.append(f"⚠️  Remove {len(file_matching['missing_annotations'])} images without annotations")
        
        if len(file_matching['missing_images']) > 0:
            recommendations.append(f"⚠️  Handle {len(file_matching['missing_images'])} annotations without images")
        
        # Class distribution recommendations
        if annotation_analysis['files_without_defects'] > 0:
            recommendations.append(f"ℹ️  Consider {annotation_analysis['files_without_defects']} files without defects as negative samples")
        
        if annotation_analysis['multi_class_files'] > 0:
            recommendations.append(f"✅ Good: {annotation_analysis['multi_class_files']} multi-label samples for training")
        
        # Ensure we have expected defect classes
        expected_classes = {'defect_1', 'defect_2', 'defect_3', 'defect_4'}
        found_classes = set(annotation_analysis['unique_defect_classes'])
        missing_classes = expected_classes - found_classes
        
        if missing_classes:
            recommendations.append(f"⚠️  Missing expected defect classes: {missing_classes}")
        
        if len(found_classes) == 4 and all(cls.startswith('defect_') for cls in found_classes):
            recommendations.append("✅ All 4 defect classes found - ready for multi-label classification")
        
        return recommendations
    
    def export_report(self, output_path: str = None) -> str:
        """Export analysis report to JSON file."""
        if output_path is None:
            output_path = self.data_root / "data_analysis_report.json"
        
        report = self.generate_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Report exported to: {output_path}")
        return str(output_path)


def main():
    """Main function to run data analysis."""
    # Path to data directory
    data_root = r"C:\Users\User\PycharmProjects\DL & SE\data"
    
    # Initialize analyzer
    analyzer = SeverstalDataAnalyzer(data_root)
    
    # Generate and export report
    report_path = analyzer.export_report()
    
    print(f"\n🎉 Analysis complete! Check the report at: {report_path}")


if __name__ == "__main__":
    main()
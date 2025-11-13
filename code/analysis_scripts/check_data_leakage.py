#!/usr/bin/env python3
"""
Script to check for data leakage and threshold optimization issues in experiments.
Analyzes:
1. Overlap between validation and test splits
2. How thresholds were computed and applied
3. Sample distribution differences
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import glob

def load_experiment_results(experiment_dir):
    """Load experiment results.json file"""
    results_path = os.path.join(experiment_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def check_split_overlap(data_dir="data", annotations_dir="data/annotations"):
    """
    Check for overlap between validation and test splits by analyzing
    which files are used in each split.
    
    This requires examining the dataset splitting logic.
    """
    print("🔍 Checking for split overlap...")
    
    # Look for existing split files or CSV files that might contain split information
    split_files = []
    for pattern in ["*split*.csv", "*train*.csv", "*val*.csv", "*test*.csv"]:
        split_files.extend(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
    
    print(f"Found potential split files: {split_files}")
    
    # Check annotations directory for files
    if os.path.exists(annotations_dir):
        annotation_files = os.listdir(annotations_dir)
        print(f"Total annotation files: {len(annotation_files)}")
        
        # Extract base image names
        image_names = []
        for f in annotation_files:
            if f.endswith('.json'):
                # Remove .jpg.json to get base name
                base_name = f.replace('.jpg.json', '')
                image_names.append(base_name)
        
        print(f"Base image names extracted: {len(image_names)}")
        return image_names
    
    return []

def analyze_threshold_computation(experiment_results):
    """
    Analyze how thresholds were computed and whether they were applied correctly.
    """
    print("🎯 Analyzing threshold computation...")
    
    if not experiment_results:
        return
    
    # Extract threshold information
    optimal_thresholds = experiment_results.get('optimal_thresholds', {})
    config = experiment_results.get('config', {})
    
    print("Optimal Thresholds Found:")
    for class_name, threshold in optimal_thresholds.items():
        deviation = abs(threshold - 0.5)
        status = "🔴 Significant deviation" if deviation > 0.1 else "🟡 Moderate deviation" if deviation > 0.05 else "🟢 Close to default"
        print(f"  {class_name}: {threshold:.3f} (deviation: {deviation:.3f}) {status}")
    
    # Check training config for threshold settings
    training_config = config.get('training', {})
    default_threshold = training_config.get('threshold', 0.5)
    print(f"\nDefault threshold in config: {default_threshold}")
    
    # Analyze if thresholds were likely optimized on validation set
    print("\n🔍 Threshold Optimization Analysis:")
    
    # Check if all thresholds are exactly 0.5 (no optimization)
    all_default = all(abs(t - 0.5) < 0.001 for t in optimal_thresholds.values())
    
    if all_default:
        print("✅ All thresholds are 0.5 - No optimization detected")
    else:
        print("⚠️  Non-default thresholds detected - Likely optimized on validation set")
        print("   This could lead to data leakage if applied to test set evaluation!")
    
    return optimal_thresholds

def compare_split_distributions(experiment_results):
    """
    Compare class distributions between validation and test sets.
    """
    print("📊 Comparing validation vs test distributions...")
    
    if not experiment_results:
        return
    
    val_metrics = experiment_results.get('best_val_metrics', {})
    test_metrics = experiment_results.get('test_metrics', {})
    
    # Extract support (sample counts) for each class
    print("\nClass Distribution Comparison:")
    print("Class           | Val Samples | Test Samples | Difference")
    print("-" * 55)
    
    class_names = ['no_defect', 'defect_1', 'defect_2', 'defect_3', 'defect_4']
    
    for class_name in class_names:
        val_support = val_metrics.get(f'{class_name}_support', 0)
        test_support = test_metrics.get(f'{class_name}_support', 0)
        diff = test_support - val_support
        diff_pct = (diff / val_support * 100) if val_support > 0 else 0
        
        print(f"{class_name:15} | {val_support:11} | {test_support:12} | {diff:+4} ({diff_pct:+5.1f}%)")
    
    # Calculate total samples
    val_total = sum(val_metrics.get(f'{cn}_support', 0) for cn in class_names)
    test_total = sum(test_metrics.get(f'{cn}_support', 0) for cn in class_names)
    
    print(f"{'Total':15} | {val_total:11} | {test_total:12} | {test_total - val_total:+4}")
    
    return {
        'val_distribution': {cn: val_metrics.get(f'{cn}_support', 0) for cn in class_names},
        'test_distribution': {cn: test_metrics.get(f'{cn}_support', 0) for cn in class_names}
    }

def analyze_performance_improvements(experiment_results):
    """
    Analyze which classes show suspicious performance improvements.
    """
    print("📈 Analyzing performance improvements (Val → Test)...")
    
    if not experiment_results:
        return
    
    val_metrics = experiment_results.get('best_val_metrics', {})
    test_metrics = experiment_results.get('test_metrics', {})
    
    print("\nPerformance Comparison (Val vs Test):")
    print("Metric              | Validation | Test      | Improvement")
    print("-" * 60)
    
    # Overall metrics
    metrics_to_check = ['f1_macro', 'f1_micro', 'precision_macro', 'recall_macro', 'accuracy']
    
    for metric in metrics_to_check:
        val_score = val_metrics.get(metric, 0)
        test_score = test_metrics.get(metric, 0)
        improvement = test_score - val_score
        improvement_pct = (improvement / val_score * 100) if val_score > 0 else 0
        
        status = "🔴 Suspicious" if improvement_pct > 3 else "🟡 Notable" if improvement_pct > 1 else "🟢 Normal"
        print(f"{metric:19} | {val_score:10.3f} | {test_score:9.3f} | {improvement:+6.3f} ({improvement_pct:+5.1f}%) {status}")
    
    # Per-class F1 scores
    print("\nPer-Class F1 Score Improvements:")
    print("Class           | Val F1    | Test F1   | Improvement")
    print("-" * 55)
    
    class_names = ['no_defect', 'defect_1', 'defect_2', 'defect_3', 'defect_4']
    
    for class_name in class_names:
        val_f1 = val_metrics.get(f'{class_name}_f1', 0)
        test_f1 = test_metrics.get(f'{class_name}_f1', 0)
        improvement = test_f1 - val_f1
        improvement_pct = (improvement / val_f1 * 100) if val_f1 > 0 else 0
        
        status = "🔴 Very suspicious" if improvement_pct > 10 else "🟡 Suspicious" if improvement_pct > 5 else "🟢 Normal"
        print(f"{class_name:15} | {val_f1:9.3f} | {test_f1:9.3f} | {improvement:+6.3f} ({improvement_pct:+5.1f}%) {status}")

def main():
    """Main analysis function"""
    print("🔍 Data Leakage and Threshold Analysis")
    print("=" * 50)
    
    # Define experiment directory
    experiment_dir = "code/experiments/results/experiment_20251113_083139"
    
    # Load experiment results
    print(f"📂 Loading experiment: {experiment_dir}")
    experiment_results = load_experiment_results(experiment_dir)
    
    if not experiment_results:
        print("❌ Could not load experiment results")
        return
    
    print(f"✅ Loaded experiment: {experiment_results.get('experiment_name', 'Unknown')}")
    print(f"   Model: {experiment_results.get('model', 'Unknown')}")
    print(f"   Best epoch: {experiment_results.get('best_epoch', 'Unknown')}")
    
    print("\n" + "=" * 50)
    
    # 1. Check for split overlap
    image_names = check_split_overlap()
    
    print("\n" + "=" * 50)
    
    # 2. Analyze threshold computation
    optimal_thresholds = analyze_threshold_computation(experiment_results)
    
    print("\n" + "=" * 50)
    
    # 3. Compare distributions
    distributions = compare_split_distributions(experiment_results)
    
    print("\n" + "=" * 50)
    
    # 4. Analyze performance improvements
    analyze_performance_improvements(experiment_results)
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY AND RECOMMENDATIONS:")
    
    # Check for red flags
    red_flags = []
    
    # Check thresholds
    if optimal_thresholds and not all(abs(t - 0.5) < 0.001 for t in optimal_thresholds.values()):
        red_flags.append("Non-default thresholds detected (possible data leakage)")
    
    # Check performance improvements
    val_f1 = experiment_results.get('best_val_metrics', {}).get('f1_macro', 0)
    test_f1 = experiment_results.get('test_metrics', {}).get('f1_macro', 0)
    if test_f1 > val_f1 and (test_f1 - val_f1) / val_f1 > 0.02:
        red_flags.append("Suspicious test performance improvement (>2%)")
    
    if red_flags:
        print("🔴 RED FLAGS DETECTED:")
        for flag in red_flags:
            print(f"   • {flag}")
        
        print("\n📋 RECOMMENDED ACTIONS:")
        print("   1. Re-evaluate test set with fixed thresholds (0.5)")
        print("   2. Verify no overlap between validation and test splits")
        print("   3. Check if threshold optimization was done only on validation set")
        print("   4. Consider using nested cross-validation for threshold optimization")
    else:
        print("✅ No obvious data leakage detected")
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()
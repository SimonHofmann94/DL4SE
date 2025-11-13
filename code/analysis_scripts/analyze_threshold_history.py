"""
Analyze threshold evolution during training.

This script visualizes how optimal thresholds change over epochs.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(experiment_dir: str) -> dict:
    """Load results.json from experiment directory."""
    results_path = Path(experiment_dir) / "results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_threshold_history(results: dict, save_path: str = None):
    """
    Plot threshold evolution during training.
    
    Args:
        results: Loaded results dictionary
        save_path: Optional path to save the plot
    """
    if "threshold_history" not in results:
        print("❌ No threshold_history found in results.")
        print("   This experiment was trained with the old implementation.")
        return
    
    threshold_history = results["threshold_history"]
    
    if not threshold_history["epoch"]:
        print("❌ Threshold history is empty.")
        return
    
    epochs = threshold_history["epoch"]
    thresholds = threshold_history["thresholds"]
    
    # Get class names
    class_names = list(thresholds[0].keys())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for class_name in class_names:
        values = [t[class_name] for t in thresholds]
        ax.plot(epochs, values, marker='o', label=class_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Optimal Threshold', fontsize=12)
    ax.set_title('Threshold Evolution During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()


def print_threshold_statistics(results: dict):
    """Print statistics about threshold evolution."""
    if "threshold_history" not in results:
        return
    
    threshold_history = results["threshold_history"]
    thresholds = threshold_history["thresholds"]
    
    if not thresholds:
        return
    
    class_names = list(thresholds[0].keys())
    
    print("\n" + "="*70)
    print("THRESHOLD STATISTICS")
    print("="*70)
    print(f"{'Class':<15} {'Initial':>10} {'Final':>10} {'Mean':>10} {'Std':>10} {'Range':>10}")
    print("-"*70)
    
    for class_name in class_names:
        values = np.array([t[class_name] for t in thresholds])
        
        initial = values[0]
        final = values[-1]
        mean = values.mean()
        std = values.std()
        value_range = values.max() - values.min()
        
        print(f"{class_name:<15} {initial:>10.3f} {final:>10.3f} {mean:>10.3f} {std:>10.3f} {value_range:>10.3f}")
    
    print("="*70)


def compare_val_test_gap(results: dict):
    """Compare Val-Test gap with and without optimized thresholds."""
    print("\n" + "="*70)
    print("VAL-TEST PERFORMANCE GAP")
    print("="*70)
    
    # Get best val metrics
    if "best_val_metrics" not in results:
        print("❌ No validation metrics found.")
        return
    
    val_f1 = results["best_val_metrics"]["f1_macro"]
    
    # Get test metrics
    if "test_metrics_standard" in results:
        test_f1_standard = results["test_metrics_standard"]["f1_macro"]
        gap_standard = test_f1_standard - val_f1
        
        print(f"\n📊 With Standard Thresholds (0.5):")
        print(f"   Val F1:  {val_f1:.4f}")
        print(f"   Test F1: {test_f1_standard:.4f}")
        print(f"   Gap:     {gap_standard:+.4f} ({gap_standard/val_f1*100:+.2f}%)")
    
    if "test_metrics_optimized" in results:
        test_f1_optimized = results["test_metrics_optimized"]["f1_macro"]
        gap_optimized = test_f1_optimized - val_f1
        
        print(f"\n🎯 With Optimized Thresholds:")
        print(f"   Val F1:  {val_f1:.4f}")
        print(f"   Test F1: {test_f1_optimized:.4f}")
        print(f"   Gap:     {gap_optimized:+.4f} ({gap_optimized/val_f1*100:+.2f}%)")
    
    if "threshold_optimization_benefit" in results:
        benefit = results["threshold_optimization_benefit"]
        print(f"\n✨ Threshold Optimization Benefit:")
        print(f"   F1 Improvement: {benefit['f1_macro_improvement']:+.4f} ({benefit['f1_macro_improvement_percent']:+.2f}%)")
    
    print("="*70)
    
    # Interpretation
    print("\n💡 Interpretation:")
    if "test_metrics_optimized" in results:
        gap = gap_optimized if "test_metrics_optimized" in results else gap_standard
        if abs(gap) < 0.02:
            print("   ✅ Excellent! Val and Test are very similar → Good generalization")
        elif abs(gap) < 0.05:
            print("   ✅ Good! Val and Test are reasonably close → Normal performance")
        else:
            print("   ⚠️  Large gap between Val and Test → Check for issues")


def main():
    parser = argparse.ArgumentParser(description="Analyze threshold evolution during training")
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment directory name (e.g., experiment_20251112_171646_best)")
    parser.add_argument("--results-dir", type=str,
                       default="code/experiments/results",
                       help="Base directory for experiment results")
    parser.add_argument("--save-plot", action="store_true",
                       help="Save plot instead of displaying")
    
    args = parser.parse_args()
    
    # Build full path
    experiment_dir = Path(args.results_dir) / args.experiment
    
    print(f"📂 Loading results from: {experiment_dir}")
    
    try:
        results = load_results(experiment_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"✓ Loaded results for: {results['experiment_name']}")
    print(f"  Model: {results['model']}")
    print(f"  Best Epoch: {results['best_epoch']}")
    
    # Print statistics
    print_threshold_statistics(results)
    
    # Compare gaps
    compare_val_test_gap(results)
    
    # Plot threshold history
    save_path = None
    if args.save_plot:
        save_path = experiment_dir / "threshold_evolution.png"
    
    plot_threshold_history(results, save_path)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

# Training Improvements - Learnable Thresholds & Enhanced Metrics

## Changes Made

### 1. **Learnable Per-Class Thresholds** ✅

**Problem:** Fixed 0.5 threshold is suboptimal for imbalanced multi-label data
- High recall but low precision indicates threshold is too low
- Different classes need different thresholds

**Solution:** Learn optimal threshold per class on validation set
- Searches thresholds from 0.1 to 0.9 in steps of 0.05
- Maximizes F1 score independently for each class
- Applied automatically during test evaluation

**Code Changes:**
- `metrics.py`: Added `find_optimal_thresholds()` function
- `trainer.py`: Learn thresholds after training, before testing
- Thresholds saved in `results.json`

**Example output:**
```
Learning optimal per-class thresholds on validation set:
  defect_1: threshold=0.45, F1=0.8523
  defect_2: threshold=0.65, F1=0.2341  (rare class needs higher threshold)
  defect_3: threshold=0.40, F1=0.9512
  defect_4: threshold=0.50, F1=0.7834
```

---

### 2. **Per-Class Confusion Matrix** ✅

**Added metrics per class:**
- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)
- Support (number of true samples)

**Code Changes:**
- `metrics.py`: Added `multilabel_confusion_matrix` from sklearn
- Confusion matrix values saved per class in metrics dict

**Benefits:**
- See exactly where model fails
- Identify if model is too conservative (low FP, high FN) or too aggressive (high FP, low FN)

---

### 3. **Enhanced Logging** ✅

**New table format for validation/test:**
```
================================================================================
PER-CLASS METRICS
================================================================================
Class           Precision     Recall         F1    Support     TP     FP     TN     FN
--------------------------------------------------------------------------------
defect_1           0.8523     0.8912     0.8713        423    377     65   1234     46
defect_2           0.2341     0.6521     0.3445         23     15     49   1567      8
defect_3           0.9512     0.9234     0.9371       1156   1067     55    413     89
defect_4           0.7834     0.8123     0.7976        567    461     82    986    106
--------------------------------------------------------------------------------
MACRO AVG          0.7053     0.8198     0.7376
MICRO AVG          0.8312     0.8734     0.8518
================================================================================
```

**Code Changes:**
- `metrics.py`: Added `log_per_class_metrics()` function
- Pretty-printed table with all metrics
- Called automatically during validation and testing

---

## What You'll See Now

### During Training:
1. Normal training epochs with loss
2. Validation metrics with basic summary
3. **After last epoch:** Threshold learning phase
4. **Test evaluation:** Uses learned thresholds automatically

### In `results.json`:
```json
{
  "optimal_thresholds": {
    "defect_1": 0.45,
    "defect_2": 0.65,
    "defect_3": 0.40,
    "defect_4": 0.50
  },
  "test_metrics": {
    "defect_1_precision": 0.8523,
    "defect_1_recall": 0.8912,
    "defect_1_f1": 0.8713,
    "defect_1_tp": 377,
    "defect_1_fp": 65,
    "defect_1_fn": 46,
    ...
  }
}
```

---

## Why This Helps

### Problem: High Recall, Low Precision
- **Before:** Model predicts "defect" too often → many false positives
- **Cause:** Threshold 0.5 is too low for some classes
- **Fix:** Learned threshold (e.g., 0.65) reduces false positives

### Per-Class Insights
- **Confusion matrix** shows if model is:
  - Missing defects (high FN) → lower threshold
  - Over-predicting (high FP) → raise threshold
  - Balanced (optimal)

### Class Imbalance Handling
- **Rare classes** (like defect_2 at 1.6%) need higher thresholds
- **Common classes** (like defect_3 at 73%) can use lower thresholds
- Automatic optimization finds best balance

---

## Next Training Run

Just run normally - threshold learning happens automatically:

```bash
python code/train.py
```

Check the detailed per-class table in the logs to understand model behavior!

---

## Files Modified

1. `code/core/training/metrics.py` - Added threshold learning & confusion matrix
2. `code/core/training/trainer.py` - Integrated threshold learning into training loop
3. Both files updated to use per-class thresholds during evaluation

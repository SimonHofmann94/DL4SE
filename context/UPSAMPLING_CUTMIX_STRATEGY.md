# Upsampling vs. CutMix - Strategien für defect_2

## 🎯 Problem: Defect_2 wird nicht gelernt

**Aktueller Status:**
- defect_2: 37 val samples, ~172 train samples
- F1 Score: 0.0000 (Modell sagt niemals defect_2 voraus)
- TP=0, FP=0 → Modell ignoriert diese Klasse komplett

**Root Cause:** Zu wenig Daten + zu hohe Alpha-Penalty (0.99) = Modell spielt safe

---

## 🔧 Lösung 1: Upsampling (Mehr Samples)

### Strategie: Repliziere seltene Samples

**Wie funktioniert es?**
1. **Weighted Sampling:** Samples mit defect_2 werden häufiger gezogen
2. **Hard Upsampling:** Samples mit defect_2 werden N-mal dupliziert im Dataset

### Implementierung

#### Option A: Weighted Random Sampler (EMPFOHLEN)
```python
# In train.py / DataLoader creation
from core.data.upsampling import get_class_balanced_indices

# Get sampling weights
indices, weights = get_class_balanced_indices(
    labels=train_labels,
    strategy='weighted_sampling',
    mode='sqrt'  # Moderate balancing
)

# Create sampler
sampler = torch.utils.data.WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)

# Use in DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,  # Instead of shuffle=True
    num_workers=num_workers
)
```

**Config:**
```yaml
data:
  upsampling:
    enabled: true
    strategy: "weighted_sampling"
    mode: "sqrt"  # Options: 'sqrt', 'inv', 'effective_num'
```

**Pros:**
- ✅ Einfach zu implementieren
- ✅ Keine Datenduplikation (nur Sampling-Gewichte)
- ✅ Flexibel (mode='sqrt' für moderate, 'inv' für aggressive Balancing)

**Cons:**
- ❌ Könnte zu Overfitting auf defect_2 Samples führen
- ❌ Andere Klassen werden seltener gesehen

---

#### Option B: Hard Upsampling (Dataset Replication)
```python
# Replicate minority samples
indices, _ = get_class_balanced_indices(
    labels=train_labels,
    strategy='upsample',
    target_ratio=0.5  # Replicate to 50% of majority class
)

# Create subset with replicated indices
train_dataset_upsampled = Subset(train_dataset, indices)
```

**Config:**
```yaml
data:
  upsampling:
    enabled: true
    strategy: "upsample"
    target_ratio: 0.5  # Or 'match_max' to match majority class
    max_replications: 5  # Don't replicate more than 5x
```

**Pros:**
- ✅ Garantiert mehr defect_2 Samples pro Epoch
- ✅ Kombinierbar mit CutMix (siehe unten)

**Cons:**
- ❌ Vergrößert effektive Dataset-Größe (längere Epochs)
- ❌ Gefahr von Overfitting auf duplizierten Samples

---

### Erwarteter Impact

**Weighted Sampling (mode='sqrt'):**
- defect_2 wird **2-3x häufiger** gesehen
- **+3-5% F1** für defect_2
- Minimal impact auf andere Klassen

**Hard Upsampling (target_ratio=0.5):**
- defect_2 Samples werden **~3-4x repliziert** (von 172 → ~600)
- **+5-8% F1** für defect_2
- **+10-20% längere Epochs**

---

## 🎨 Lösung 2: CutMix (Bessere Generalisierung)

### Strategie: Mische Bilder, kombiniere Labels

**Wie funktioniert es?**
1. Nehme zwei Bilder (img1 mit defect_2, img2 mit anderen defects)
2. Schneide Region aus img2, paste in img1
3. **WICHTIG:** Stelle sicher, dass Defekte NICHT abgeschnitten werden
4. Kombiniere Labels: mixed_label = img1_label OR img2_label

### Beispiel
```
img1: [0, 0, 1, 0, 0]  (hat defect_2)
img2: [1, 0, 0, 1, 0]  (hat no_defect + defect_3)

CutMix → mixed_img mit Labels: [1, 0, 1, 1, 0]
         (alle Defekte sind im mixed image sichtbar)
```

### Implementierung

**In Dataset (defect-aware):**
```python
from core.augmentation import DefectAwareCutMix

cutmix = DefectAwareCutMix(
    prob=0.5,
    alpha=1.0,
    min_cut_ratio=0.1,
    max_cut_ratio=0.3  # Conservative for defects
)

# In __getitem__:
if self.training and random.random() < 0.5:
    # Pick random second sample
    idx2 = random.randint(0, len(self) - 1)
    img2, label2, bbox2 = self.get_raw_sample(idx2)
    
    # Apply CutMix
    img, label = cutmix(img, label, bbox, img2, label2, bbox2)
```

**Config:**
```yaml
augmentation:
  cutmix:
    enabled: true
    prob: 0.5  # 50% chance per sample
    alpha: 1.0
    min_cut_ratio: 0.1
    max_cut_ratio: 0.3  # Don't cut too large (might remove context)
```

### Erwarteter Impact

**CutMix allein:**
- **+1-3% F1 Macro** (bessere Generalisierung für ALLE Klassen)
- **+2-4% F1 defect_2** (durch Kombination mit anderen Samples)
- Modell lernt robustere Features

**Mit Upsampling kombiniert:**
- Upsampling erhöht defect_2 Häufigkeit
- CutMix erstellt NEUE defect_2 Kombinationen (nicht nur Duplikate)
- **Synergieeffekt: +5-10% F1 defect_2**

---

## 📊 Vergleich: Upsampling vs. CutMix vs. Kombiniert

| Strategie | defect_2 F1 | Training Zeit | Overfitting Risk | Implementation |
|-----------|-------------|---------------|------------------|----------------|
| **Baseline** | 0.00 | 1x | Low | - |
| **Weighted Sampling** | +3-5% | 1x | Medium | Easy ✅ |
| **Hard Upsampling** | +5-8% | 1.2x | High | Easy ✅ |
| **CutMix** | +2-4% | 1.05x | Low | Medium |
| **Upsampling + CutMix** | +5-10% | 1.2x | Medium | Medium |

---

## 🎯 Empfohlener Workflow

### Phase 1: Quick Fix (Alpha Tuning)
```yaml
# Aktuelles Training - SOFORT testen
loss:
  alpha: [0.375, 0.99, 0.70, 0.375, 0.99]  # defect_2: 0.99 → 0.70
```
**Erwartung:** +2-3% defect_2 F1  
**Zeit:** 0 min (nur Config-Änderung)

---

### Phase 2: Weighted Sampling (nächster Run)
```yaml
data:
  upsampling:
    enabled: true
    strategy: "weighted_sampling"
    mode: "sqrt"
```
**Erwartung:** +4-6% defect_2 F1  
**Zeit:** ~5 min Integration

---

### Phase 3: CutMix (experimentell)
```yaml
augmentation:
  cutmix:
    enabled: true
    prob: 0.5
    max_cut_ratio: 0.3
```
**Erwartung:** +3-5% Macro F1, +2-4% defect_2 F1  
**Zeit:** ~30 min Integration (muss in Dataset eingebaut werden)

---

### Phase 4: Kombination (Best Case)
```yaml
loss:
  alpha: [0.375, 0.99, 0.70, 0.375, 0.99]

data:
  upsampling:
    enabled: true
    strategy: "weighted_sampling"
    mode: "inv"  # Aggressive

augmentation:
  cutmix:
    enabled: true
    prob: 0.5
```
**Erwartung:** +8-12% defect_2 F1  
**Zeit:** ~45 min Integration

---

## 🔍 Monitoring & Debugging

### Was du tracken solltest:

1. **Per-Epoch Class Distribution im Training:**
   ```python
   # Wie oft wird jede Klasse gesehen?
   train_class_counts = []
   for batch in train_loader:
       train_class_counts.append(batch['label'].sum(dim=0))
   
   print(f"Avg class counts per epoch: {torch.stack(train_class_counts).mean(dim=0)}")
   ```

2. **Training vs. Validation Metrics für defect_2:**
   - Wenn Train F1 hoch, Val F1 niedrig → Overfitting
   - Wenn beide niedrig → Nicht genug Daten / zu schwierige Features

3. **Logits-Verteilung für defect_2:**
   ```python
   # Schaue dir die Raw Predictions an (vor Threshold)
   logits_defect2 = predictions[:, 2]  # defect_2 = Index 2
   print(f"Logits: min={logits_defect2.min()}, max={logits_defect2.max()}, mean={logits_defect2.mean()}")
   ```
   - Wenn mean < 0.0 → Modell ist zu pessimistisch
   - Wenn max < 0.5 → Threshold 0.5 ist zu hoch

---

## 💡 Final Recommendation

**START HERE (Heute):**
1. ✅ Alpha auf 0.70 für defect_2
2. ✅ Weighted Sampling mit mode='sqrt'

**Total Implementation Time:** ~5 Minuten  
**Expected Impact:** +5-8% defect_2 F1

**NEXT (Wenn nicht genug):**
3. Add CutMix mit prob=0.5, max_cut_ratio=0.3
4. Switch to mode='inv' für aggressiveres Balancing

---

## 📝 Code-Beispiele

Siehe:
- `code/core/data/upsampling.py` - Upsampling-Strategien
- `code/core/augmentation/cutmix.py` - Defect-Aware CutMix
- `config/UPSAMPLING_AND_CUTMIX_CONFIG.yaml` - Vollständige Config

Viel Erfolg! 🚀

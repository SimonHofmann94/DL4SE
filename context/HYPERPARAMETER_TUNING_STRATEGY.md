# Hyperparameter Tuning - Strategie und Überlegungen

## Deine Frage: Random Search mit nur 20 Epochen sinnvoll?

**Kurze Antwort: NEIN - nicht bei eurem Setup.**

## Warum Random Search nach 20 Epochen problematisch ist

### Beweis aus deinen Ergebnissen:

```
Beste Epoche: 83
F1 Macro (Epoch 20): 0.5228
F1 Macro (Epoch 83): 0.8370

Verbesserung von Epoch 20 → 83: +60%!
```

**Epoch-by-Epoch Analyse:**
- Epoch 1-10: F1 = 0.36-0.51 (Modell lernt Basics)
- Epoch 11-30: F1 = 0.47-0.62 (Langsamer Progress)
- Epoch 31-50: F1 = 0.57-0.71 (Beschleunigung)
- Epoch 51-83: F1 = 0.72-0.84 (Feintuning & Konvergenz)

### Das Problem:

Bei Epoch 20 hättest du die besten Hyperparameter FALSCH eingeschätzt:
- Ein Setup mit schnellem Early Progress (z.B. hohe LR, niedriges gamma) sieht gut aus
- Ein Setup mit langsamem Start aber besserem Endresultat (z.B. Class-Balanced Loss) sieht schlecht aus

**Beispiel:**
```
Setup A: lr=0.01, gamma=2.0  → Epoch 20: F1=0.60, Epoch 80: F1=0.75
Setup B: lr=0.0001, gamma=3.0 → Epoch 20: F1=0.52, Epoch 80: F1=0.85

Nach 20 Epochen würdest du Setup A wählen... falsch!
```

## Alternative Strategien

### ✅ Strategie 1: Sequential Grid Search (Empfohlen)

**Idee:** Tune Hyperparameter sequentiell, nicht alle gleichzeitig.

**Phase 1: Focal Loss Parameter (3-4 Runs)**
```yaml
Variante A: gamma=2.5, use_effective_num=false  (Baseline)
Variante B: gamma=3.0, use_effective_num=false
Variante C: gamma=3.0, use_effective_num=true, beta=0.999
Variante D: gamma=3.0, use_effective_num=true, beta=0.9999
```
- Vollständiges Training (100 Epochen)
- Best Result wird Baseline für Phase 2

**Phase 2: Batch Size Tuning (2-3 Runs)**
```yaml
Best from Phase 1 +
  - batch_size=16 (current baseline)
  - batch_size=8
  - batch_size=4
```
- **Warum kleinere Batch Sizes testen?**
  - Kleinere Batches = mehr Weight Updates pro Epoch = bessere Generalisierung
  - Noisier Gradients können aus Local Minima entkommen
  - Oft bessere Performance bei schwierigen Klassen (wie defect_2!)
  - **Trade-off:** Langsameres Training, weniger stabile Konvergenz

**Phase 3: Learning Rate & Optimizer (2-3 Runs)**
```yaml
Best from Phase 1 + Best Batch Size +
  - lr=0.001 (current)
  - lr=0.0005 (falls Batch Size kleiner → oft niedrigere LR besser)
  - lr=0.0001 (optional, falls Phase 2 zeigte dass kleinere BS gut ist)
```

**Phase 4: Augmentation & Blackout (2-3 Runs)**
```yaml
Best from Phase 1+2+3 +
  - Blackout disabled (current)
  - Blackout enabled, defect_2 only, prob=0.3
  - Blackout enabled, defect_2 only, prob=0.5
```

**Total: 10-13 vollständige Trainings** (machbar in 2-3 Wochen auf RunPod)

**Vorteil:**
- Jeder Run ist aussagekräftig (100 Epochen)
- Verstehst du den Einfluss jedes Parameters
- Keine "versteckten Interaktionen" übersehen

---

### ✅ Strategie 2: Warmstart + Validation-Based Early Selection

**Idee:** Starte viele Konfigurationen, kill schlechte frühzeitig.

**Setup:**
```
Start 12 Runs parallel (wenn GPU-Kapazität da ist)
  - Checkpoint nach Epoch 10, 20, 30, 40...
  - Vergleiche Validation F1 Macro
  
Nach Epoch 30:
  - Kill worst 50% (6 Runs)
  
Nach Epoch 50:
  - Kill worst 50% der verbleibenden (3 Runs)
  
Verbleibende 3 Runs: Bis Epoch 100 laufen lassen
```

**Vorteil:**
- Spart Rechenzeit
- Mehr Konfigurationen ausprobiert
- Bessere Coverage von Hyperparameter-Space

**Nachteil:**
- Komplexer zu implementieren
- Könnte "langsam startende aber gute" Setups fälschlich eliminieren

---

### ✅ Strategie 3: Multi-Fidelity mit Scaling

**Idee:** Train auf reduziertem Subset für Screening, dann volle Daten für Best Candidates.

**Phase 1: Quick Screening (30% der Daten, 40 Epochen)**
```
Test 20 Hyperparameter-Kombinationen
Verwende nur 30% der 12k Bilder (3.6k)
Train für 40 Epochen

Identifiziere Top 5 Setups
```

**Phase 2: Full Training (100% der Daten, 100 Epochen)**
```
Top 5 Setups von Phase 1
Vollständige 12k Bilder
Volle 100 Epochen
```

**Bewertung ob das funktioniert:**
- ✅ 30% der Daten: Wahrscheinlich repräsentativ wenn stratified sampling
- ⚠️ 40 Epochen: Immer noch kritisch... aber besser als 20
- ✅ Top 5: Guter Buffer gegen false negatives

---

## Was ich empfehle für EUCH

**Empfehlung: Strategie 1 (Sequential Grid Search)**

**Begründung:**
1. Ihr habt bereits 1 komplettes Training (Baseline)
2. Die nächsten 2-3 Runs sollten gezielt sein:
   - Gamma 3.0 + Class-Balanced Loss (JETZT)
   - Ggf. Learning Rate Anpassung
   - Ggf. Blackout für defect_2
3. Ihr könnt innerhalb 1 Woche 5-6 volle Trainings machen
4. Das reicht für solid Verbesserungen

**Konkrete Aktionen:**

### Run 1 (JETZT): Gamma 3.0 + Class-Balanced Loss
```bash
python code/train.py
# Nutzt bereits aktualisierte train_config.yaml
```

**Erwartung:**
- Defect 2 F1 sollte steigen (von 0.68 → 0.72+)
- Leichter Trade-off bei anderen Klassen möglich
- Overall F1 Macro: 0.84-0.86 (Target)

---

### Run 2 (falls Run 1 gut): Lower Learning Rate
```bash
python code/train.py optimizer.lr=0.0005
```

**Warum:**
- Gamma 3.0 = aggressiveres Training
- Könnte von kleinerer LR profitieren
- Feineres Tuning der schweren Klassen

---

### Run 3 (experimental): Blackout für Defect 2
```bash
python code/train.py \
  augmentation.defect_blackout.enabled=true \
  augmentation.defect_blackout.defect_indices_to_blackout=[2] \
  augmentation.defect_blackout.instance_blackout_prob=0.4
```

**Hinweis:** Benötigt Dataset-Erweiterung (siehe BLACKOUT_AUGMENTATION_INTEGRATION.md)

---

## Warum KEIN Random Search?

**Random Search macht Sinn wenn:**
- ✅ Schnelles Feedback möglich (z.B. 5-10 Epochen ausreichend)
- ✅ Viele Hyperparameter zu tunen (10+)
- ✅ Keine Intuition über gute Werte

**Bei euch trifft NICHTS davon zu:**
- ❌ Langsames Feedback (Epoch 83 für Best Model)
- ❌ Wenige kritische Hyperparameter (gamma, lr, blackout_prob)
- ✅ Ihr habt bereits gute Intuition (Baseline Run funktioniert!)

**Random Search würde bedeuten:**
- 20 Runs × 20 Epochen = 400 Epochen Rechenzeit
- vs.
- 5 Runs × 100 Epochen = 500 Epochen Rechenzeit

**Für nur 25% mehr Rechenzeit bekommst du:**
- Aussagekräftige Ergebnisse
- Verständnis der Hyperparameter-Effekte
- Reproduzierbare Best Practices

---

## Learning Curve Analysis

Schau dir deine Training History an:

```python
# Aus results.json
val_f1_macro = [
    0.360,  # Epoch 1
    0.195,  # Epoch 2  ← Dip!
    0.401,  # Epoch 3
    ...
    0.561,  # Epoch 14
    0.577,  # Epoch 15
    ...
    0.710,  # Epoch 50
    ...
    0.837,  # Epoch 83 ← Best!
]
```

**Beobachtungen:**
1. **Early Instability (Epoch 1-15):** Große Schwankungen
2. **Steady Improvement (Epoch 15-60):** Kontinuierlicher Anstieg
3. **Fine-tuning (Epoch 60-83):** Langsamerer, aber wichtiger Progress
4. **Convergence (Epoch 83-98):** Relativ stabil, leichter Overfit

**Minimum sinnvolle Training-Zeit: ~50 Epochen**
- Ab da: Kurve wird stabiler
- Kannst du "Trend" erkennen
- Aber: Würdest trotzdem ~30% Performance verschenken

---

## Hyperparameter Impact Prediction

Basierend auf Literatur + deinen Daten:

| Hyperparameter | Expected Impact on Defect 2 F1 | Confidence |
|----------------|--------------------------------|------------|
| **gamma: 2.5→3.0** | +2% bis +5% | High ✅ |
| **use_effective_num=true** | +1% bis +3% | Medium ⚠️ |
| **batch_size: 16→8** | +1% bis +4% | Medium-High ⚠️ |
| **batch_size: 16→4** | +2% bis +6% | Medium ⚠️ |
| **lr: 0.001→0.0005** | +0% bis +2% | Low-Medium |
| **Blackout (prob=0.4)** | +3% bis +8% | Medium-High ⚠️ |

### 🎯 Warum kleinere Batch Sizes helfen können

**Theorie (aus Literatur):**

1. **"Noisy Gradients" = Bessere Generalisierung**
   - Kleinere Batches → weniger repräsentativ → mehr Rauschen
   - Rauschen hilft, aus flachen Local Minima zu entkommen
   - Führt zu "breiteren" Minima = bessere Generalisierung auf Test Set
   - Paper: "On Large-Batch Training for Deep Learning" (Keskar et al., 2017)

2. **Mehr Weight Updates pro Epoch**
   - Batch Size 16: 12000/16 = 750 Updates pro Epoch
   - Batch Size 8: 12000/8 = 1500 Updates pro Epoch (+100%)
   - Batch Size 4: 12000/4 = 3000 Updates pro Epoch (+300%)
   - Mehr Updates = feingranulareres Lernen

3. **Bessere Performance bei Imbalanced Classes**
   - Kleine Batches = höhere Wahrscheinlichkeit, seltene Klassen zu sehen
   - Bei Batch Size 4: Jeder Batch könnte 1 defect_2 Sample enthalten
   - Bei Batch Size 16: Defect_2 könnte in vielen Batches fehlen
   - **Wichtig für eure Situation!**

**Trade-offs:**

❌ **Nachteile kleiner Batch Sizes:**
- **Langsameres Training:** 4× mehr Updates = 4× länger pro Epoch
- **Instabilere Konvergenz:** Loss-Kurve wird "zackiger"
- **Höherer Memory Overhead:** Mehr Backward Passes
- **Batch Normalization Problems:** Bei BS=4 sehr kleine Statistiken

✅ **Vorteile kleiner Batch Sizes:**
- **Bessere Generalisierung:** Oft 1-5% F1 improvement
- **Hilft bei rare classes:** Defect_2 könnte profitieren
- **Robuster gegen Overfitting**

**Empfehlung für euch:**

```
Priorität 1: Batch Size 8
  - Guter Kompromiss: 2× mehr Updates, nicht zu langsam
  - Sollte stabil konvergieren
  - Erwartete Trainingszeit: ~1.5× länger als BS=16

Priorität 2: Batch Size 4 (experimentell)
  - Maximaler Effekt auf rare classes
  - Könnte 2-3× länger dauern
  - Könnte instabil werden → evtl. LR anpassen
  
Falls BS=4 gut ist:
  - Learning Rate reduzieren (0.0005 oder 0.0001)
  - Warmup Epochs erhöhen (5 → 10)
  - Gradient Clipping aktivieren (max_norm=1.0)
```

**Best Case Szenario:**
```
Current Defect 2 F1: 0.681 (Test Set)

Mit gamma=3.0 + Class-Balanced:     0.681 → 0.710 (+4%)
+ Batch Size 8:                      0.710 → 0.740 (+4%)
+ Blackout:                          0.740 → 0.785 (+6%)
+ LR tuning:                         0.785 → 0.795 (+1%)

Target F1 für Defect 2: ~0.78-0.80
```

---

## Fazit: Empfohlener Plan

**Keine Random Search. Sequential Experimentation.**

### **Phase 1: Focal Loss (Woche 1)**

**Run 1a: Gamma 3.0 + Class-Balanced Loss (PRIORITÄT 1)**
```bash
python code/train.py
# Nutzt bereits aktualisierte train_config.yaml
```

**Erwartung:**
- Defect 2 F1: 0.681 → 0.71-0.73
- Overall F1 Macro: 0.837 → 0.85-0.86
- **Entscheidung:** Wenn Verbesserung > 2%, nutze dies als neue Baseline

**Run 1b (optional): Gamma 3.5 (aggressive)**
```bash
python code/train.py loss.gamma=3.5
```
- Nur falls Run 1a nicht genug bringt
- Riskanter (könnte overfitting verursachen)

---

### **Phase 2: Batch Size (Woche 1-2)**

**Run 2a: Batch Size 8**
```bash
python code/train.py data.batch_size=8
# Mit best gamma/alpha aus Phase 1
```

**Erwartung:**
- Training Duration: ~1.5× länger (ca. 150 Epochen in gleicher Zeit wie 100 @ BS=16)
- Defect 2 F1: +2-4% improvement
- Stabilere Konvergenz als BS=4

**Run 2b: Batch Size 4 (experimentell)**
```bash
python code/train.py data.batch_size=4
```

**Erwartung:**
- Training Duration: ~2-3× länger
- Defect 2 F1: +3-6% improvement (best case)
- **Watchout:** Könnte instabil werden → Monitor erste 20 Epochen

**Entscheidung nach Phase 2:**
- Vergleiche F1 Macro & Defect 2 F1 für BS=16, BS=8, BS=4
- Wähle beste Batch Size für Phase 3

---

### **Phase 3: Learning Rate (Woche 2)**

**Nur ausführen falls Phase 2 zeigte: BS=8 oder BS=4 ist besser!**

**Run 3a: Lower LR mit best Batch Size**
```bash
python code/train.py data.batch_size=<BEST_BS> optimizer.lr=0.0005
```

**Warum:**
- Kleinere Batches profitieren oft von niedrigerer LR
- Feineres Tuning möglich
- Könnte weitere 1-2% bringen

**Run 3b (optional): Sehr niedrige LR**
```bash
python code/train.py data.batch_size=<BEST_BS> optimizer.lr=0.0001
```
- Nur testen falls 3a schlechter als erwartet
- Langsames aber sehr präzises Lernen

---

### **Phase 4: Blackout Augmentation (Woche 3)**

**Voraussetzung:** Dataset muss erweitert werden (siehe BLACKOUT_AUGMENTATION_INTEGRATION.md)

**Run 4a: Moderate Blackout**
```bash
python code/train.py \
  data.batch_size=<BEST_BS> \
  optimizer.lr=<BEST_LR> \
  augmentation.defect_blackout.enabled=true \
  augmentation.defect_blackout.defect_indices_to_blackout=[2] \
  augmentation.defect_blackout.instance_blackout_prob=0.3
```

**Run 4b: Aggressive Blackout**
```bash
python code/train.py \
  data.batch_size=<BEST_BS> \
  optimizer.lr=<BEST_LR> \
  augmentation.defect_blackout.enabled=true \
  augmentation.defect_blackout.defect_indices_to_blackout=[2] \
  augmentation.defect_blackout.instance_blackout_prob=0.5
```

**Erwartung:**
- Defect 2 F1: +3-8% (größter einzelner Impact!)
- Könnte leichten Trade-off bei anderen Klassen geben
- Modell lernt, Defekt-Kontext zu nutzen

---

## Erwartetes Endresultat

**Konservative Schätzung:**
```
Phase 1 (Gamma+CB-Loss):    Defect 2: 0.681 → 0.710 (+4%)
Phase 2 (Batch Size 8):     Defect 2: 0.710 → 0.740 (+4%)
Phase 3 (LR=0.0005):        Defect 2: 0.740 → 0.750 (+1%)
Phase 4 (Blackout):         Defect 2: 0.750 → 0.785 (+5%)

Final: Defect 2 F1 = 0.78-0.80
Overall F1 Macro = 0.87-0.89
```

**Optimistische Schätzung:**
```
Alle Phasen synergieren gut:
Defect 2 F1 = 0.82-0.85
Overall F1 Macro = 0.89-0.91
```

**Worst Case:**
```
Nur Phase 1 hilft wirklich:
Defect 2 F1 = 0.72-0.74
Overall F1 Macro = 0.85-0.86
```

---

## Zeitplan & Ressourcen

**Total Investment:**
- 10-13 vollständige Trainingsläufe
- @ 100 Epochen each
- @ ca. 8-12 Stunden pro Run (je nach Batch Size)
- **Total Zeit: 2-3 Wochen** auf RunPod (RTX 4090)

**Kosten (grobe Schätzung):**
- RunPod RTX 4090: ~$0.50-0.70/Stunde
- 13 Runs × 10 Stunden = 130 GPU-Stunden
- **Total: ~$65-90**

**Schnellerer Plan (falls Zeit knapp):**
- Phase 1: 2 Runs (gamma 3.0 + CB-Loss)
- Phase 2: 2 Runs (BS=8, BS=4)
- Phase 4: 1 Run (Blackout @ best config)
- **Total: 5 Runs, 1 Woche, ~$35-50**

---

## 📋 Quick Reference: Training Commands

| Phase | Run | Command | Expected Time | Priority |
|-------|-----|---------|---------------|----------|
| **1a** | Gamma 3.0 + CB | `python code/train.py` | 8-10h | 🔥 HIGH |
| **1b** | Gamma 3.5 | `python code/train.py loss.gamma=3.5` | 8-10h | MEDIUM |
| **2a** | Batch Size 8 | `python code/train.py data.batch_size=8` | 12-15h | 🔥 HIGH |
| **2b** | Batch Size 4 | `python code/train.py data.batch_size=4` | 20-25h | MEDIUM |
| **3a** | LR 0.0005 + BS8 | `python code/train.py data.batch_size=8 optimizer.lr=0.0005` | 12-15h | MEDIUM |
| **3b** | LR 0.0001 + BS8 | `python code/train.py data.batch_size=8 optimizer.lr=0.0001` | 12-15h | LOW |
| **4a** | Blackout 30% | `python code/train.py augmentation.defect_blackout.enabled=true augmentation.defect_blackout.instance_blackout_prob=0.3` | 12-15h | 🔥 HIGH |
| **4b** | Blackout 50% | `python code/train.py augmentation.defect_blackout.enabled=true augmentation.defect_blackout.instance_blackout_prob=0.5` | 12-15h | MEDIUM |

**Note:** Run 4a/4b requires dataset integration (see BLACKOUT_AUGMENTATION_INTEGRATION.md)

---

## 🎯 Monitoring & Decision Points

### Nach Phase 1:
```python
# Vergleiche in results.json:
baseline_f1_defect2 = 0.681
run1a_f1_defect2 = ?

if run1a_f1_defect2 > baseline_f1_defect2 + 0.02:
    print("✅ Phase 1 erfolgreich! Weiter zu Phase 2")
    use_gamma = 3.0
else:
    print("⚠️ Gamma 3.0 bringt wenig. Teste 3.5 oder überspringe Phase 1")
```

### Nach Phase 2:
```python
results = {
    "BS=16": {"f1_macro": 0.85, "defect_2_f1": 0.71},
    "BS=8":  {"f1_macro": 0.87, "defect_2_f1": 0.74},
    "BS=4":  {"f1_macro": 0.88, "defect_2_f1": 0.76},
}

best_bs = max(results, key=lambda x: results[x]["defect_2_f1"])
print(f"Best Batch Size: {best_bs}")

# Entscheidung:
if best_bs == "BS=4":
    print("BS=4 ist am besten → Phase 3 mit LR=0.0005 oder 0.0001")
    recommended_lr = 0.0005
elif best_bs == "BS=8":
    print("BS=8 ist am besten → Phase 3 mit LR=0.0005 optional")
    recommended_lr = 0.0005
else:
    print("BS=16 bleibt beste → Überspringe Phase 3, direkt zu Phase 4")
```

### Nach Phase 3:
```python
# Finale Config für Phase 4 (Blackout)
final_config = {
    "gamma": 3.0,
    "use_effective_num": True,
    "batch_size": best_bs,
    "lr": best_lr
}

print(f"Finale Config für Blackout-Tests: {final_config}")
```

### Nach Phase 4:
```python
# Vergleiche alle Runs
all_runs = {
    "Baseline (BS=16, gamma=2.5)": 0.681,
    "Phase 1 (gamma=3.0)": 0.710,
    "Phase 2 (BS=8)": 0.740,
    "Phase 3 (LR=0.0005)": 0.750,
    "Phase 4 (Blackout 30%)": 0.785,
    "Phase 4 (Blackout 50%)": 0.775  # könnte weniger sein wenn zu aggressive
}

best_run = max(all_runs, key=all_runs.get)
print(f"🏆 Best Configuration: {best_run}")
print(f"   Defect 2 F1: {all_runs[best_run]:.3f}")
print(f"   Improvement: +{(all_runs[best_run] - 0.681)*100:.1f}%")
```

# Defect Blackout Augmentation - Integration Guide

## Overview

Die **DefectBlackoutTransform** ist implementiert und konfigurierbar, aber **standardmäßig deaktiviert** (enabled: false).

## Was wurde implementiert?

✅ **Defect Blackout Transform Klasse** (`code/core/augmentation/defect_blackout.py`)
- Selektives Blackout: Zufälliges Maskieren einzelner Defekt-Instanzen
- Complete Blackout: Alle Defekte entfernen (synthetische "clean" Samples)
- Konfigurierbar per Defekt-Typ (z.B. nur defect_2)

✅ **Configuration** (`config/train_config.yaml`)
```yaml
augmentation:
  defect_blackout:
    enabled: false  # Standardmäßig AUS
    instance_blackout_prob: 0.5
    defect_indices_to_blackout: null  # null = alle, oder [2] für nur defect_2
    min_pixels_to_blackout: 15
    complete_blackout_prob: 0.0
    verbose: false
```

## ⚠️ Wichtige Einschränkung

**Aktuell verwendet die `SeverstalFullImageDataset` KEINE Ground-Truth Masken während des Trainings.**

Die Dataset-Klasse lädt nur:
- Bilder (PIL Image)
- Labels (5-dimensionaler Vektor: [no_defect, defect_1, defect_2, defect_3, defect_4])

Für Blackout-Augmentation werden aber benötigt:
- Pixelgenaue Masken (H×W numpy array mit class indices)
- Instanz-Separation (scipy.ndimage.label für connected components)

## Wie Blackout aktivieren?

### Option 1: Separate Wide-Strip Dataset erstellen (empfohlen)

Erstelle eine neue Dataset-Klasse `SeverstalWideStripDataset`, die:
1. Masken aus Annotations lädt (wie in `Archive/datasets_5_classes_blackout.py`)
2. `DefectBlackoutTransform` im `__getitem__` aufruft
3. Blackout VOR den Torchvision-Transforms anwendet

**Beispiel-Integration:**

```python
class SeverstalWideStripDataset(Dataset):
    def __init__(self, ..., blackout_transform=None):
        self.blackout_transform = blackout_transform
        # ...
    
    def __getitem__(self, idx):
        # 1. Load image
        image_pil = Image.open(...)
        
        # 2. Load mask from annotation
        gt_mask_np = self._load_mask_from_annotation(...)
        
        # 3. Apply blackout BEFORE other transforms
        if self.blackout_transform:
            image_pil, gt_mask_np, modified = self.blackout_transform(
                image_pil, gt_mask_np
            )
        
        # 4. Create label from modified mask
        label = self._create_label_from_mask(gt_mask_np)
        
        # 5. Apply standard transforms (ColorJitter, Normalize, etc.)
        if self.transform:
            image_tensor = self.transform(image_pil)
        
        return image_tensor, label
```

### Option 2: Aktuelle Dataset erweitern

Erweitere `SeverstalFullImageDataset` um:
- `load_masks: bool` Parameter
- `_load_mask_from_annotation()` Methode (aus Archive übernehmen)
- Conditional blackout application

## Testen der Blackout-Klasse

Die Klasse kann standalone getestet werden:

```bash
cd "c:\Users\User\PycharmProjects\DL & SE 2"
python code/core/augmentation/defect_blackout.py
```

Dies erstellt ein synthetisches Testbeispiel und zeigt:
- Wie viele Instanzen gefunden wurden
- Wie viele Pixel geschwärzt wurden
- Ob die Transformation erfolgreich war

## Konfigurationsbeispiele

### Nur Defect 2 schwärzen (50% der Instanzen)
```yaml
defect_blackout:
  enabled: true
  instance_blackout_prob: 0.5
  defect_indices_to_blackout: [2]  # Nur defect_2
  min_pixels_to_blackout: 15
```

### Aggressive Blackout für alle Defekte
```yaml
defect_blackout:
  enabled: true
  instance_blackout_prob: 0.7  # 70% der Instanzen
  defect_indices_to_blackout: null  # Alle Defekte
  complete_blackout_prob: 0.1  # 10% komplett clean machen
  verbose: true  # Debug logging
```

### Nur für Experimente: Viele clean Samples generieren
```yaml
defect_blackout:
  enabled: true
  complete_blackout_prob: 0.3  # 30% komplett geschwärzt
  defect_indices_to_blackout: [1, 2, 3, 4]
```

## Nächste Schritte zum Aktivieren

1. **Entscheide:** Wide-Strip Dataset erstellen ODER aktuelle Dataset erweitern?

2. **Implementiere Mask-Loading:**
   - Siehe `Archive/datasets_5_classes_blackout.py` für Referenz
   - Methode: `create_full_mask_from_png_object()` aus `utils.py`

3. **Integriere Blackout in Dataset:**
   - Instantiate `DefectBlackoutTransform` aus Config
   - Apply VOR torchvision transforms
   - Update Labels basierend auf modifizierter Mask

4. **Teste:**
   ```bash
   python code/validate_system.py  # Sollte funktionieren
   python code/train.py augmentation.defect_blackout.enabled=true augmentation.defect_blackout.verbose=true
   ```

5. **Experimentiere:**
   - Start: nur defect_2, prob=0.3
   - Erhöhe graduell: prob=0.5, dann 0.7
   - Teste complete_blackout_prob wenn sinnvoll

## Warum standardmäßig deaktiviert?

- Benötigt zusätzliche Implementierung in Dataset
- Scipy dependency (schon in requirements.txt, aber optional)
- Performance: Mask-Loading + Instance-Labeling ist langsamer
- Soll bewusst aktiviert werden nach Testing

## Referenzen

- Implementierung: `code/core/augmentation/defect_blackout.py`
- Konfiguration: `config/train_config.yaml` (augmentation.defect_blackout)
- Beispiel-Dataset mit Blackout: `Archive/datasets_5_classes_blackout.py`

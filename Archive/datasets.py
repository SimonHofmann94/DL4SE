"""
Severstal Steel Defect Dataset Classes
Multi-label classification for 4 defect classes with patch and strip processing modes.
"""

import os
import json
from typing import Optional, Tuple, List, Dict
from matplotlib import pyplot as plt

from scipy import ndimage

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig, ListConfig

import numpy as np
import base64
import zlib

class DefectBlackoutTransform:
    def __init__(self, 
                 instance_blackout_prob: float = 0.5, 
                 min_pixels_to_blackout: int = 10, # Mindestgröße einer Defektinstanz, um sie zu berücksichtigen
                 fill_value: int = 0, # Womit geschwärzt wird (0 = schwarz)
                 verbose: bool = False): # Für Debugging-Ausgaben
        self.instance_blackout_prob = instance_blackout_prob
        self.min_pixels_to_blackout = min_pixels_to_blackout
        self.fill_value = fill_value
        self.verbose = verbose
        
        # Defektklassen-Indizes, die für Blackout relevant sind (1-4)
        self.relevant_defect_indices = [1, 2, 3, 4] 

    def __call__(self, image_pil: Image.Image, gt_mask_np: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """
        Args:
            image_pil (PIL.Image): Das Eingabebild (Streifen).
            gt_mask_np (np.ndarray): Die Ground-Truth-Maske für den Streifen (gleiche Größe wie image_pil).
                                     Werte sind Klassenindizes (0=kein Defekt, 1-4=Defekte).
        Returns:
            Tuple[PIL.Image, np.ndarray]: Das modifizierte Bild und die modifizierte Maske.
        """
        if self.verbose:
            print(f"\n[DefectBlackout] Starte Augmentierung. Prob: {self.instance_blackout_prob}")

        if torch.rand(1).item() >= self.instance_blackout_prob:
            if self.verbose:
                print("[DefectBlackout] Augmentierung nicht angewendet (Wahrscheinlichkeit).")
            return image_pil, gt_mask_np # Keine Änderung

        # Kopien erstellen, um Originale nicht zu verändern
        img_out_pil = image_pil.copy()
        mask_out_np = gt_mask_np.copy()
        
        draw = ImageDraw.Draw(img_out_pil)

        num_blacked_out_instances = 0
        original_defect_pixels_before_blackout = np.sum(np.isin(mask_out_np, self.relevant_defect_indices))

        if self.verbose:
            print(f"[DefectBlackout] Originale Defektpixel (Klassen 1-4) im Streifen: {original_defect_pixels_before_blackout}")

        if original_defect_pixels_before_blackout == 0:
            if self.verbose:
                print("[DefectBlackout] Keine Defekte im Streifen vorhanden. Keine Aktion.")
            return img_out_pil, mask_out_np

        # Iteriere über die relevanten Defektklassen
        for defect_idx in self.relevant_defect_indices:
            # Erstelle eine binäre Maske nur für die aktuelle Defektklasse
            binary_mask_for_class = (mask_out_np == defect_idx).astype(np.int32)
            
            # Finde zusammenhängende Komponenten (Instanzen) dieser Klasse
            # structure = np.ones((3,3)) # 8er-Nachbarschaft
            labeled_array, num_features = ndimage.label(binary_mask_for_class) # structure=structure
            
            if self.verbose and num_features > 0:
                print(f"[DefectBlackout] Klasse {defect_idx}: {num_features} Instanz(en) gefunden.")

            if num_features == 0:
                continue # Keine Instanzen dieser Klasse

            # Iteriere über jede gefundene Instanz dieser Klasse
            for i in range(1, num_features + 1): # Label-Werte gehen von 1 bis num_features
                instance_mask_pixels = (labeled_array == i)
                num_instance_pixels = np.sum(instance_mask_pixels)

                if self.verbose:
                    print(f"[DefectBlackout]  - Instanz {i} (Klasse {defect_idx}): Größe {num_instance_pixels} Pixel.")

                if num_instance_pixels < self.min_pixels_to_blackout:
                    if self.verbose:
                        print(f"[DefectBlackout]    - Instanz zu klein (min: {self.min_pixels_to_blackout}). Überspringe.")
                    continue
                
                # Entscheidung, ob diese spezifische Instanz geschwärzt wird
                # Hier könnte man komplexere Logik einbauen (z.B. nur einen Teil der Instanzen schwärzen)
                # Fürs Erste: Jede qualifizierte Instanz wird mit 50% Wahrscheinlichkeit geschwärzt
                if torch.rand(1).item() < 0.5: # 50% Chance pro Instanz
                    if self.verbose:
                        print(f"[DefectBlackout]    - Schwärze Instanz {i} (Klasse {defect_idx}).")
                    
                    # Finde Koordinaten der Pixel dieser Instanz
                    ys, xs = np.where(instance_mask_pixels)
                    
                    # Schwärze im Bild
                    for y_coord, x_coord in zip(ys, xs):
                        # ImageDraw.point erwartet (x,y)
                        draw.point((x_coord, y_coord), fill=self.fill_value)
                    
                    # Entferne aus Maske (setze auf 0 = kein Defekt)
                    mask_out_np[instance_mask_pixels] = 0 # NO_DEFECT_IDX
                    num_blacked_out_instances += 1
                elif self.verbose:
                    print(f"[DefectBlackout]    - Instanz {i} (Klasse {defect_idx}) nicht geschwärzt (Wahrscheinlichkeit).")
        
        if self.verbose:
            final_defect_pixels_after_blackout = np.sum(np.isin(mask_out_np, self.relevant_defect_indices))
            print(f"[DefectBlackout] Augmentierung angewendet. {num_blacked_out_instances} Defektinstanzen geschwärzt.")
            print(f"[DefectBlackout] Defektpixel nach Blackout: {final_defect_pixels_after_blackout}")
            if original_defect_pixels_before_blackout > 0 and final_defect_pixels_after_blackout == 0:
                print("[DefectBlackout] ALLE Defekte wurden aus dem Streifen entfernt!")
        
        return img_out_pil, mask_out_np


def find_multilabel_severstal_examples(cfg: DictConfig, max_to_find: int = 5):
    print("\nSuche nach Severstal JSON-Beispielen mit mehreren Defektklassen...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_dir_path = os.path.join(project_root, cfg.data.severstal.img_dir)
    ann_dir_path = os.path.join(project_root, cfg.data.severstal.ann_dir)

    if not os.path.exists(img_dir_path) or not os.path.exists(ann_dir_path):
        print("Fehler: Bild- oder Annotationsverzeichnis nicht gefunden.")
        return

    # Minimal-Transform, nur um das Laden zu ermöglichen
    minimal_transform = transforms.Compose([transforms.ToTensor()])

    # Lade das gesamte Dataset (oder einen großen Teil davon) ohne Debug-Limit für die Suche
    full_dataset = SeverstalDataset(
        img_dir=img_dir_path,
        ann_dir=ann_dir_path,
        transform=minimal_transform, # Nur für die Bildladung, nicht relevant für JSON-Inhalt
        debug_limit=None # Alle Bilder durchsuchen
    )
    
    print(f"Durchsuche {len(full_dataset)} Samples...")
    found_count = 0
    
    for i in range(len(full_dataset)):
        # Wir greifen direkt auf die interne Logik zu, um das JSON zu laden,
        # ohne das Bild komplett zu prozessieren, wenn es nicht nötig ist.
        img_name = full_dataset.image_names[i]
        base_name_no_ext = os.path.splitext(img_name)[0]
        
        ann_path_variants = [
            os.path.join(full_dataset.ann_dir, f"{img_name}.json"),
            os.path.join(full_dataset.ann_dir, f"{base_name_no_ext}.json")
        ]
        ann_path = None
        for p_variant in ann_path_variants:
            if os.path.exists(p_variant):
                ann_path = p_variant
                break
        
        if ann_path:
            try:
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                
                present_defect_indices = []
                if "objects" in annotation and annotation["objects"]:
                    for obj in annotation["objects"]:
                        class_title = obj.get("classTitle")
                        if class_title in SeverstalDataset.DEFECT_CLASS_TO_IDX:
                            class_idx = SeverstalDataset.DEFECT_CLASS_TO_IDX[class_title]
                            if class_idx not in present_defect_indices: # Nur eindeutige Indizes
                                present_defect_indices.append(class_idx)
                
                if len(present_defect_indices) > 1: # Mehr als eine Defektklasse gefunden
                    print(f"\n--- Multilabel Beispiel gefunden für Bild: {img_name} ---")
                    print(f"Annotationsdatei: {ann_path}")
                    print(f"Gefundene Defekt-Indizes (1-4): {sorted(present_defect_indices)}")
                    # Gib das gesamte JSON aus
                    print("JSON Inhalt:")
                    print(json.dumps(annotation, indent=2))
                    found_count += 1
                    if found_count >= max_to_find:
                        break
            except Exception as e:
                print(f"Fehler beim Verarbeiten von {ann_path}: {e}")
        if i % 500 == 0 and i > 0 :
            print(f"  ... {i} Samples geprüft ...")

    if found_count == 0:
        print("Keine Beispiele mit mehreren unterschiedlichen Defektklassen gefunden.")
    else:
        print(f"\nSuche abgeschlossen. {found_count} Beispiele gefunden.")

# --- Utility Functions for Bitmap Processing ---
def base64_zlib_to_pil_image(base64_str: str) -> Optional[Image.Image]:
    """
    Convert Base64 encoded zlib-compressed bitmap to PIL Image.
    """
    try:
        decoded_bytes = base64.b64decode(base64_str)
        decompressed_bytes = zlib.decompress(decoded_bytes)
        # Assuming it's a PNG format after decompression
        from io import BytesIO
        return Image.open(BytesIO(decompressed_bytes))
    except Exception as e:
        print(f"Error decoding base64 bitmap: {e}")
        return None

def create_full_mask_from_png_object(b64_data: str, origin: List[int], full_h: int, full_w: int) -> Optional[np.ndarray]:
    """
    Create full-size mask from base64 encoded PNG object.
    
    Args:
        b64_data: Base64 encoded bitmap data
        origin: [x, y] coordinates of the top-left corner
        full_h: Full image height
        full_w: Full image width
    
    Returns:
        Binary mask array (0s and 1s) of size (full_h, full_w)
    """
    try:
        # Decode the base64 bitmap
        decoded_bytes = base64.b64decode(b64_data)
        decompressed_bytes = zlib.decompress(decoded_bytes)
        
        # Convert to PIL Image
        from io import BytesIO
        mask_pil = Image.open(BytesIO(decompressed_bytes)).convert('L')
        mask_np = np.array(mask_pil)
        
        # Create full-size mask
        full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        
        # Get mask dimensions and origin
        mask_h, mask_w = mask_np.shape
        orig_x, orig_y = origin
        
        # Ensure we don't go out of bounds
        end_y = min(orig_y + mask_h, full_h)
        end_x = min(orig_x + mask_w, full_w)
        
        # Place the mask in the full image
        # Convert non-zero pixels to 1 (binary mask)
        mask_binary = (mask_np > 0).astype(np.uint8)
        full_mask[orig_y:end_y, orig_x:end_x] = mask_binary[:end_y-orig_y, :end_x-orig_x]
        
        return full_mask
        
    except Exception as e:
        print(f"Error creating full mask from PNG object: {e}")
        return None


class SeverstalDataset(Dataset): # Wird jetzt zum Patch-Dataset
    NO_DEFECT_IDX = 0
    DEFECT_CLASS_TO_IDX = {
        "defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4
    }
    NUM_TOTAL_CLASSES = 5

    def __init__(self,
                 img_dir: str,
                 ann_dir: str,
                 patch_size_hw: Tuple[int, int], # (H_patch, W_patch)
                 stride_hw: Tuple[int, int],     # (H_stride, W_stride)
                 # img_size_tuple_hw: Tuple[int, int], # Wird durch patch_size_hw ersetzt für den Output
                 transform: Optional[transforms.Compose] = None,
                 max_neg_patches_per_image: Optional[int] = 5, # Max. Anzahl negativer Patches pro Bild
                 min_positive_pixel_percentage: float = 0.01, # Mindestanteil an Defektpixeln, um als positiver Patch zu gelten
                 debug_limit_images: Optional[int] = None): # Limit für Anzahl der zu verarbeitenden Originalbilder
        
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.patch_h, self.patch_w = patch_size_hw
        self.stride_h, self.stride_w = stride_hw
        self.transform = transform
        self.max_neg_patches_per_image = max_neg_patches_per_image
        self.min_positive_pixel_percentage = min_positive_pixel_percentage * self.patch_h * self.patch_w # Absolute Pixelanzahl

        self.patches_info = [] # Hier speichern wir Infos zu jedem Patch

        all_original_image_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        images_to_process = all_original_image_names
        if debug_limit_images is not None and debug_limit_images > 0: # Diese Bedingung greift jetzt nicht, wenn es null ist
            images_to_process = all_original_image_names[:debug_limit_images]
            print(f"SeverstalDataset (Patch Mode): Debug-Limit aktiv, verarbeite die ersten {len(images_to_process)} Originalbilder für Patch-Extraktion.")
        else:
            print(f"SeverstalDataset (Patch Mode): Verarbeite alle {len(images_to_process)} Originalbilder für Patch-Extraktion.")

        print(f"Starte Patch-Extraktion für {len(images_to_process)} Bilder...")
        for i, img_name in enumerate(images_to_process):
            if i % 100 == 0 and i > 0:
                print(f"  ... Patches für {i} Bilder extrahiert ...")

            img_path = os.path.join(self.img_dir, img_name)
            base_name_no_ext = os.path.splitext(img_name)[0]
            
            try:
                image_pil_original = Image.open(img_path).convert("RGB")
                original_h, original_w = image_pil_original.height, image_pil_original.width
            except FileNotFoundError:
                print(f"Warnung: Originalbild {img_path} nicht gefunden. Überspringe für Patch-Extraktion.")
                continue

            # Lade Annotation und erstelle kombinierte GT-Maske in Originalgröße
            combined_gt_mask_np_original_size = np.zeros((original_h, original_w), dtype=np.uint8)
            current_image_has_annotated_defect = False # Um zu wissen, ob wir negative Patches suchen sollen

            ann_path_variants = [
                os.path.join(self.ann_dir, f"{img_name}.json"),
                os.path.join(self.ann_dir, f"{base_name_no_ext}.json")
            ]
            ann_path = None
            for p_variant in ann_path_variants:
                if os.path.exists(p_variant):
                    ann_path = p_variant
                    break
            
            if ann_path:
                try:
                    with open(ann_path, 'r') as f:
                        annotation = json.load(f)
                    if "objects" in annotation and annotation["objects"]:
                        for obj in annotation["objects"]:
                            class_title = obj.get("classTitle")
                            if class_title in self.DEFECT_CLASS_TO_IDX:
                                current_image_has_annotated_defect = True # Zumindest ein bekannter Defekt ist da
                                class_idx_for_mask = self.DEFECT_CLASS_TO_IDX[class_title]
                                if "bitmap" in obj:
                                    b64_data = obj["bitmap"].get("data")
                                    origin = obj["bitmap"].get("origin")
                                    if b64_data and origin:
                                        single_obj_mask_np = create_full_mask_from_png_object(
                                            b64_data, origin, original_h, original_w
                                        )
                                        if single_obj_mask_np is not None:
                                            combined_gt_mask_np_original_size[single_obj_mask_np == 1] = class_idx_for_mask
                except Exception as e_json:
                    print(f"Fehler beim Verarbeiten von JSON {ann_path} für {img_name} bei Patch-Extraktion: {e_json}")
            
            # --- Patch-Extraktion für das aktuelle Bild ---
            neg_patches_count_for_this_image = 0
            for y in range(0, original_h - self.patch_h + 1, self.stride_h):
                for x in range(0, original_w - self.patch_w + 1, self.stride_w):
                    patch_coords_in_original = (x, y) # Obere linke Ecke des Patches
                    
                    gt_mask_patch = combined_gt_mask_np_original_size[
                        y : y + self.patch_h, 
                        x : x + self.patch_w
                    ]
                    
                    patch_label_vector = torch.zeros(self.NUM_TOTAL_CLASSES, dtype=torch.float32)
                    unique_classes_in_patch = np.unique(gt_mask_patch)
                    
                    has_defect_in_this_patch = False
                    # Zähle Pixel pro Defektklasse im Patch
                    positive_pixel_counts = 0

                    for class_val in unique_classes_in_patch:
                        if class_val >= 1 and class_val <= 4: # Defektklassen 1-4
                            num_pixels_for_class = np.sum(gt_mask_patch == class_val)
                            if num_pixels_for_class >= self.min_positive_pixel_percentage:
                                patch_label_vector[class_val] = 1.0
                                has_defect_in_this_patch = True
                                positive_pixel_counts += num_pixels_for_class # Für Debug oder erweiterte Logik
                    
                    if has_defect_in_this_patch:
                        patch_label_vector[self.NO_DEFECT_IDX] = 0.0
                        self.patches_info.append({
                            "image_path": img_path, # Ganzer Pfad zum Originalbild
                            "coords_in_original": patch_coords_in_original, # (x,y) der oberen linken Ecke
                            "label_vector": patch_label_vector
                        })
                    elif current_image_has_annotated_defect and np.all(gt_mask_patch == 0):
                        # Nur Patches aus ANNOTIERTEN Bildern als "kein Defekt" nehmen,
                        # wenn sie WIRKLICH keine Defektpixel gemäß Annotation enthalten.
                        if self.max_neg_patches_per_image is None or \
                        neg_patches_count_for_this_image < self.max_neg_patches_per_image:
                            
                            patch_label_vector[self.NO_DEFECT_IDX] = 1.0
                            self.patches_info.append({
                                "image_path": img_path,
                                "coords_in_original": patch_coords_in_original,
                                "label_vector": patch_label_vector
                            })
                            neg_patches_count_for_this_image += 1
        
        if not self.patches_info:
            print("WARNUNG: Keine Patches konnten extrahiert werden! Bitte Konfiguration und Daten prüfen.")
        else:
            print(f"Patch-Extraktion abgeschlossen. {len(self.patches_info)} Patches indiziert.")


    def __len__(self) -> int:
        return len(self.patches_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]: # Gibt Patch, Label, Patch-Identifier zurück
        patch_info = self.patches_info[idx]
        
        image_path = patch_info["image_path"]
        coords = patch_info["coords_in_original"] # (x_start, y_start)
        label_vector = patch_info["label_vector"]
        
        try:
            image_pil_original = Image.open(image_path).convert("RGB")
        except FileNotFoundError: # Sollte nicht passieren, wenn __init__ erfolgreich war
            # Fallback
            dummy_patch_tensor = torch.zeros((3, self.patch_h, self.patch_w))
            if self.transform:
                dummy_patch_pil = Image.new('RGB', (self.patch_w, self.patch_h))
                try: dummy_patch_tensor = self.transform(dummy_patch_pil)
                except: pass
            return dummy_patch_tensor, label_vector, f"ERROR_PATCH_IMG_NOT_FOUND_{os.path.basename(image_path)}"

        # Extrahiere den Patch
        x_start, y_start = coords[0], coords[1]
        patch_pil = image_pil_original.crop((x_start, y_start, 
                                             x_start + self.patch_w, y_start + self.patch_h))
        
        # Wende Transformationen auf den Patch an
        transformed_patch_tensor = patch_pil # Initialisiere mit PIL Image
        if self.transform:
            transformed_patch_tensor = self.transform(patch_pil)
        else: # Fallback, falls keine Transformationen definiert sind
            transformed_patch_tensor = transforms.ToTensor()(patch_pil)

        # Erstelle einen eindeutigen Identifier für den Patch (optional)
        patch_identifier = f"{os.path.splitext(os.path.basename(image_path))[0]}_x{x_start}_y{y_start}"
        
        # Für die Klassifikation brauchen wir die GT-Maske des Patches hier nicht unbedingt zurückzugeben,
        # da das Label schon auf Patch-Ebene bestimmt wurde.
        # Wenn du sie doch brauchst (z.B. für eine Segmentierungs-Loss auf Patch-Ebene):
        # gt_mask_patch = combined_gt_mask_np_original_size[y_start : y_start + self.patch_h, x_start : x_start + self.patch_w]
        # gt_mask_patch_tensor = torch.from_numpy(gt_mask_patch).long()
        # return transformed_patch_tensor, label_vector, gt_mask_patch_tensor, patch_identifier

        return transformed_patch_tensor, label_vector, patch_identifier

# NEUE KLASSE für den Streifen-Modus
class SeverstalStripDataset(Dataset):
    NO_DEFECT_IDX = 0
    DEFECT_CLASS_TO_IDX = {
        "defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4
    }
    NUM_TOTAL_CLASSES = 5

    def __init__(self,
                 img_dir: str,
                 ann_dir: str,
                 strip_height: int,
                 transform: Optional[transforms.Compose] = None,
                 use_defect_blackout: bool = False,
                 defect_blackout_prob: float = 0.5,
                 target_no_defect_ratio: Optional[float] = None,
                 instance_blackout_prob_selective: float = 0.5,
                 blackout_min_pixels: int = 10,
                 verbose_strip_dataset_debug: bool = False,
                 debug_limit_images: Optional[int] = None,
                 image_filenames_to_use: Optional[List[str]] = None): # <<<< NEUER PARAMETER HIER
        
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.strip_height = strip_height
        self.transform = transform
        self.use_defect_blackout = use_defect_blackout
        self.defect_blackout_prob = defect_blackout_prob
        self.target_no_defect_ratio = target_no_defect_ratio
        # self.blackout_all_defects_prob = blackout_all_defects_prob # Veraltet durch target_no_defect_ratio
        self.verbose_debug = verbose_strip_dataset_debug

        self.defect_blackout_transform = None
        if self.use_defect_blackout:
            self.defect_blackout_transform = DefectBlackoutTransform(
                instance_blackout_prob=instance_blackout_prob_selective,
                min_pixels_to_blackout=blackout_min_pixels,
                verbose=self.verbose_debug
            )
            if self.defect_blackout_transform is not None and self.verbose_debug: # Nur printen wenn auch initialisiert UND verbose
                 print(f"[DEBUG SeverstalStripDataset] DefectBlackoutTransform.verbose ist auf: {self.defect_blackout_transform.verbose}")


        if image_filenames_to_use is not None:
            # Wenn eine spezifische Liste von Dateinamen übergeben wird, verwende diese.
            # Das Debug-Limit wird in diesem Fall ignoriert, da die Liste bereits "final" ist.
            self.image_files = sorted(list(set(image_filenames_to_use))) # set zur Sicherheit gegen Duplikate, dann sortieren
            print(f"SeverstalStripDataset: Initialisiert mit einer spezifischen Liste von {len(self.image_files)} Bilddateinamen.")
            if debug_limit_images is not None and debug_limit_images > 0:
                 print(f"  (Hinweis: debug_limit_images ({debug_limit_images}) wird ignoriert, da image_filenames_to_use bereitgestellt wurde.)")
        else:
            # Standardverhalten: Lade alle Bilder aus img_dir und wende ggf. debug_limit an.
            all_image_files_in_dir = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if debug_limit_images is not None and debug_limit_images > 0:
                self.image_files = all_image_files_in_dir[:debug_limit_images]
                print(f"SeverstalStripDataset: Debug-Limit ({debug_limit_images}) aktiv. Verwende die ersten {len(self.image_files)} Bilder aus {img_dir}.")
            else:
                self.image_files = all_image_files_in_dir
                print(f"SeverstalStripDataset: Verarbeite alle {len(self.image_files)} Originalbilder aus {img_dir}.")
        
        # Annahme: Dein img_dir für wide_strips enthält nur Bilder, die bereits Defekte haben.
        # Daher ist kein weiterer Scan der Annotationen hier in __init__ nötig, um die Liste zu filtern.
        
        if not self.image_files:
            print("WARNUNG: SeverstalStripDataset - Keine Bilddateien zum Verarbeiten vorhanden!")
        else:
            print(f"SeverstalStripDataset: Finale Anzahl zu verarbeitender Bilder: {len(self.image_files)}.")
            

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_combined_gt_mask(self, img_name: str, original_h: int, original_w: int) -> np.ndarray:
        """
        Hilfsfunktion zum Laden der kombinierten Ground-Truth-Maske für ein Bild.
        (Fast identisch zur Logik in SeverstalDataset init)
        """
        combined_gt_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        base_name_no_ext = os.path.splitext(img_name)[0]
        ann_path_variants = [
            os.path.join(self.ann_dir, f"{img_name}.json"),
            os.path.join(self.ann_dir, f"{base_name_no_ext}.json")
        ]
        ann_path = None
        for p_variant in ann_path_variants:
            if os.path.exists(p_variant):
                ann_path = p_variant
                break
        
        if ann_path:
            try:
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                if "objects" in annotation and annotation["objects"]:
                    for obj in annotation["objects"]:
                        class_title = obj.get("classTitle")
                        if class_title in self.DEFECT_CLASS_TO_IDX:
                            class_idx_for_mask = self.DEFECT_CLASS_TO_IDX[class_title]
                            if "bitmap" in obj:
                                b64_data = obj["bitmap"].get("data")
                                origin = obj["bitmap"].get("origin")
                                if b64_data and origin:
                                    single_obj_mask_np = create_full_mask_from_png_object(
                                        b64_data, origin, original_h, original_w
                                    )
                                    if single_obj_mask_np is not None:
                                        combined_gt_mask[single_obj_mask_np == 1] = class_idx_for_mask
            except Exception as e_json:
                if self.verbose_debug: print(f"Fehler beim Verarbeiten von JSON für {img_name} in _load_combined_gt_mask: {e_json}")
        else:
            if self.verbose_debug: print(f"WARNUNG: Für Bild {img_name} wurde keine Annotationsdatei gefunden, obwohl erwartet.")
        return combined_gt_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image_pil_original = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback, falls Bild trotz Auflistung nicht da ist
            print(f"WARNUNG (StripDataset): Bild {img_path} nicht gefunden in __getitem__.")
            # Erzeuge Dummy-Daten, um einen Crash zu vermeiden
            dummy_strip_tensor = torch.zeros((3, self.strip_height, 1600)) # Annahme Standardbreite
            # if self.target_strip_width: dummy_strip_tensor = torch.zeros((3, self.strip_height, self.target_strip_width))
            dummy_label_vector = torch.zeros(self.NUM_TOTAL_CLASSES, dtype=torch.float32)
            dummy_label_vector[self.NO_DEFECT_IDX] = 1.0 # Sicher als "kein Defekt"
            return dummy_strip_tensor, dummy_label_vector, f"ERROR_IMG_NOT_FOUND_{img_name}"

        original_w, original_h = image_pil_original.size # PIL size ist (width, height)

        # Lade die volle GT-Maske für das Bild
        gt_mask_original_np = self._load_combined_gt_mask(img_name, original_h, original_w)

        # --- Streifenextraktion (Random Vertical Crop) ---
        # Die Breite des Streifens ist erstmal die Originalbreite des Bildes.
        # Höhe ist self.strip_height.
        # TODO: Wenn self.target_strip_width gesetzt ist, hier horizontal croppen/padden.
        
        current_strip_width = original_w # Fürs Erste
        # if self.target_strip_width:
        #     if original_w > self.target_strip_width:
        #         # Random horizontal crop
        #         x_start_crop = random.randint(0, original_w - self.target_strip_width)
        #         image_pil_original = image_pil_original.crop((x_start_crop, 0, x_start_crop + self.target_strip_width, original_h))
        #         gt_mask_original_np = gt_mask_original_np[:, x_start_crop : x_start_crop + self.target_strip_width]
        #         current_strip_width = self.target_strip_width
        #     elif original_w < self.target_strip_width:
        #         # Padding (komplexer, erstmal vermeiden)
        #         print(f"WARNUNG: Bildbreite {original_w} ist kleiner als Zielbreite {self.target_strip_width}. Kein Padding implementiert.")
        #         pass


        if original_h <= self.strip_height:
            # Bild ist niedriger oder gleich der Streifenhöhe, nehme das ganze Bild vertikal
            y_start_strip = 0
            img_strip_pil = image_pil_original.copy() # Kopiere das (ggf. horizontal gecroppte) Bild
            gt_mask_strip_np = gt_mask_original_np.copy()
            # Hier könnte man noch Padding hinzufügen, wenn strip_height > original_h
        else:
            # Random vertical crop
            y_start_strip = torch.randint(0, original_h - self.strip_height + 1, (1,)).item()
            img_strip_pil = image_pil_original.crop((0, y_start_strip, current_strip_width, y_start_strip + self.strip_height))
            gt_mask_strip_np = gt_mask_original_np[y_start_strip : y_start_strip + self.strip_height, :]
        
        # --- Defect Blackout Augmentierung (Platzhalter) ---
        # Diese Funktion müsste das img_strip_pil und gt_mask_strip_np modifizieren
        # und das resultierende (möglicherweise geänderte) gt_mask_strip_np zurückgeben.
        final_gt_mask_for_labeling_strip_np = gt_mask_strip_np # Vorerst keine Änderung
        # Defect Blackout Augmentierung anwenden
        if self.use_defect_blackout and self.defect_blackout_transform is not None: # Sicherstellen, dass es initialisiert wurde
            # Die Transformation erwartet PIL Image und NumPy Maske
            img_strip_pil, final_gt_mask_for_labeling_strip_np = self.defect_blackout_transform(
                img_strip_pil, 
                gt_mask_strip_np # Übergebe die originale GT-Maske des Streifens
            )
            # Wichtig: img_strip_pil wurde durch die Transformation modifiziert
            # final_gt_mask_for_labeling_strip_np ist die modifizierte Maske

        # --- Label-Generierung für den Streifen ---
        strip_label_vector = torch.zeros(self.NUM_TOTAL_CLASSES, dtype=torch.float32)
        unique_classes_in_final_strip = np.unique(final_gt_mask_for_labeling_strip_np)
        
        has_defect_in_final_strip = False
        min_pixels_for_defect_in_strip = 1 
    
        for class_val in unique_classes_in_final_strip:
            if class_val >= 1 and class_val <= 4: # Defektklassen 1-4
                num_pixels = np.sum(final_gt_mask_for_labeling_strip_np == class_val)
                if num_pixels >= min_pixels_for_defect_in_strip:
                    strip_label_vector[class_val] = 1.0
                    has_defect_in_final_strip = True
        
        if not has_defect_in_final_strip: 
            # Da das Originalbild garantiert Defekte hatte,
            # bedeutet dies, dass Blackout erfolgreich *alle* Defekte 
            # im aktuellen Streifen entfernt hat.
            strip_label_vector[self.NO_DEFECT_IDX] = 1.0
            if self.verbose_debug: print(f"[Labeling {img_name}] Streifen als 'Kein Defekt' gelabelt (alle Defekte entfernt durch Blackout).")
        else: 
            strip_label_vector[self.NO_DEFECT_IDX] = 0.0
            if self.verbose_debug: print(f"[Labeling {img_name}] Streifen mit Defekten gelabelt: {strip_label_vector.numpy().astype(int)}")


        # --- Standard-Transformationen anwenden ---
        # Wie Hflip, Vflip, BrightnessContrast, ToTensor, Normalize
        # Diese werden über das 'transform'-Argument des Datasets gesteuert
        transformed_strip_tensor = img_strip_pil
        if self.transform:
            transformed_strip_tensor = self.transform(img_strip_pil)
        else: # Fallback
            transformed_strip_tensor = transforms.ToTensor()(img_strip_pil)
            # Man müsste hier noch Normalisierung hinzufügen, wenn kein transform übergeben wird

        identifier = f"{os.path.splitext(img_name)[0]}_strip_y{y_start_strip}"
        return transformed_strip_tensor, strip_label_vector, identifier

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig): # cfg kommt jetzt von Hydra
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Korrigierter Projekt-Root: {project_root}")

    # Pfade für unsere Severstal-Datenstruktur
    img_dir_path_cfg = cfg.data.severstal.get("img_dir", "data/images/img_dmg")
    ann_dir_path_cfg = cfg.data.severstal.get("ann_dir", "data/annotations/ann_dmg")

    # Absolute Pfade basierend auf unserem Projekt-Root
    img_dir_abs_path = os.path.join(project_root, img_dir_path_cfg)
    ann_dir_abs_path = os.path.join(project_root, ann_dir_path_cfg)
    
    print(f"Bildverzeichnis: {img_dir_abs_path}")
    print(f"Annotationsverzeichnis: {ann_dir_abs_path}")
    
    # Überprüfe, ob Verzeichnisse existieren
    if not os.path.exists(img_dir_abs_path):
        print(f"WARNUNG: Bildverzeichnis nicht gefunden: {img_dir_abs_path}")
        return
    if not os.path.exists(ann_dir_abs_path):
        print(f"WARNUNG: Annotationsverzeichnis nicht gefunden: {ann_dir_abs_path}")
        return


    current_training_mode = cfg.data.severstal.get("training_mode", "patches") # Standard auf "patches"
    print(f"\nAktueller Testmodus (aus cfg.data.severstal.training_mode): {current_training_mode}")

    if current_training_mode == "patches":
        print("\n--- Teste SeverstalDataset (Patch-Modus) ---")
        patch_proc_cfg = cfg.data.severstal.patch_processing
        if not patch_proc_cfg.get("enabled", False):
            print("Patch-Verarbeitung (patch_processing.enabled) nicht in Config aktiviert für Patch-Modus-Test. Stoppe.")
            return

        patch_size_hw = tuple(patch_proc_cfg.patch_size_hw)
        stride_hw = tuple(patch_proc_cfg.stride_hw)
        max_neg = patch_proc_cfg.get("max_neg_patches_per_image", 5)
        min_pos_perc = patch_proc_cfg.get("min_positive_pixel_percentage", 0.01)
        # debug_img_limit_patches = patch_proc_cfg.get("debug_limit_images_patching", None)
        # Für den Test hier setzen wir ein kleines Limit, um es schnell zu halten
        debug_img_limit_patches_test = patch_proc_cfg.get("debug_limit_images_patching", 10)


        print(f"Patch-Parameter: Size={patch_size_hw}, Stride={stride_hw}, MaxNegPerImg={max_neg}, MinPosPerc={min_pos_perc}")

        patch_test_transform = transforms.Compose([
            transforms.Resize(patch_size_hw), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        sever_patch_dataset = SeverstalDataset(
            img_dir=img_dir_abs_path,
            ann_dir=ann_dir_abs_path,
            patch_size_hw=patch_size_hw,
            stride_hw=stride_hw,
            transform=patch_test_transform,
            max_neg_patches_per_image=max_neg,
            min_positive_pixel_percentage=min_pos_perc,
            debug_limit_images=debug_img_limit_patches_test # Test-Limit verwenden
        )
        
        if len(sever_patch_dataset) == 0:
            print("Severstal Patch Dataset ist leer.")
            return

        # Verwende eine kleine Batch-Größe für den Test
        test_batch_size_patches = min(4, len(sever_patch_dataset)) 
        if test_batch_size_patches == 0: return

        sever_loader_patches = torch.utils.data.DataLoader(
            sever_patch_dataset, batch_size=test_batch_size_patches, shuffle=True, num_workers=0
        )

        print(f"\nTeste Severstal Patch DataLoader ({len(sever_patch_dataset)} Patches insgesamt aus {debug_img_limit_patches_test} Bildern):")
        for i, batch_data in enumerate(sever_loader_patches):
            patches, labels, patch_ids = batch_data
            print(f"Batch {i+1}: Patch Tensor Shape: {patches.shape}, Labels Shape: {labels.shape}")
            if i == 0: # Visualisiere den ersten Patch des ersten Batches
                plt.figure(figsize=(6,6))
                patch_to_show = patches[0].permute(1,2,0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
                patch_to_show = std * patch_to_show + mean
                patch_to_show = np.clip(patch_to_show, 0, 1)
                plt.imshow(patch_to_show)
                plt.title(f"Patch Mode Sample: {patch_ids[0]}\nLabel: {labels[0].numpy().astype(int)}")
                plt.savefig("patch_dataset_sample_via_main.png")
                print("Beispiel-Patch als 'patch_dataset_sample_via_main.png' gespeichert.")
                plt.close()
            if i >= 1: break # Zeige nur die ersten paar Batches für den Test
        print("--- Test SeverstalDataset (Patch-Modus) abgeschlossen ---")

    elif current_training_mode == "wide_strips":
        print("\n--- Teste SeverstalStripDataset (Wide Strip-Modus) ---")
        strip_proc_cfg = cfg.data.severstal.wide_strip_processing
        if not strip_proc_cfg.get("enabled", False):
            print("Wide Strip-Verarbeitung (wide_strip_processing.enabled) nicht in Config für Test aktiviert. Stoppe.")
            return

        strip_height = strip_proc_cfg.strip_height
        use_blackout = strip_proc_cfg.get("use_defect_blackout", False)
        blackout_prob = strip_proc_cfg.get("defect_blackout_prob", 0.5)
        # debug_img_limit_strips = cfg.data.severstal.patch_processing.get("debug_limit_images_patching", None) # Kann denselben Debug-Limit verwenden
        # Für den Test hier setzen wir ein kleines Limit, um es schnell zu halten
        debug_img_limit_strips_test = cfg.data.severstal.patch_processing.get("debug_limit_images_patching", 10)


        print(f"Strip-Parameter: Höhe={strip_height}, UseBlackout={use_blackout}, BlackoutProb={blackout_prob}")

        # Annahme: Für Streifen wollen wir oft die volle Breite, daher Resize auf (strip_height, original_width)
        # Die Normalisierung ist wichtig. Die Breite hier ist ein Platzhalter.
        # In der echten __getitem__ wird die Breite des Bildes genommen.
        # Für den Test-Transform muss die Breite bekannt sein, oder man macht es komplexer.
        # Einfachheitshalber nehmen wir eine typische Breite für die Visualisierung des Transforms.
        example_strip_width_for_transform = 1600 # Typische Breite
        strip_test_transform = transforms.Compose([
            transforms.Resize((strip_height, example_strip_width_for_transform)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        strip_dataset = SeverstalStripDataset(
            img_dir=img_dir_abs_path,
            ann_dir=ann_dir_abs_path,
            strip_height=strip_height,
            transform=strip_test_transform,
            use_defect_blackout=use_blackout,
            defect_blackout_prob=blackout_prob,
            debug_limit_images=debug_img_limit_strips_test # Test-Limit verwenden
        )

        if len(strip_dataset) == 0:
            print("Severstal Strip Dataset ist leer.")
            return

        test_batch_size_strips = min(2, len(strip_dataset)) # Kleinere Batch-Größe für Streifen
        if test_batch_size_strips == 0: return
            
        sever_loader_strips = torch.utils.data.DataLoader(
            strip_dataset, batch_size=test_batch_size_strips, shuffle=True, num_workers=0
        )

        print(f"\nTeste Severstal Strip DataLoader ({len(strip_dataset)} Bilder für Streifenextraktion):")
        for i, batch_data in enumerate(sever_loader_strips):
            strip_tensors, label_vectors, identifiers = batch_data
            print(f"Batch {i+1}: Strip Tensor Shape: {strip_tensors.shape}, Label Vector Shape: {label_vectors.shape}")
            print(f"  Label (erstes Sample): {label_vectors[0].numpy().astype(int)}")
            print(f"  Identifier (erstes Sample): {identifiers[0]}")

            if i == 0: # Visualisiere den ersten Streifen des ersten Batches
                plt.figure(figsize=(12, 4)) # Breiter für Streifen
                strip_to_show = strip_tensors[0].permute(1,2,0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
                strip_to_show = std * strip_to_show + mean
                strip_to_show = np.clip(strip_to_show, 0, 1)
                plt.imshow(strip_to_show)
                plt.title(f"Strip Mode Sample: {identifiers[0]}\nLabel: {label_vectors[0].numpy().astype(int)}")
                plt.savefig("strip_dataset_sample_via_main.png")
                print("Beispiel-Streifen als 'strip_dataset_sample_via_main.png' gespeichert.")
                plt.close()
            if i >= 0: break # Zeige nur den ersten Batch für den Test
        print("--- Test SeverstalStripDataset (Wide Strip-Modus) abgeschlossen ---")
    else:
        print(f"Unbekannter training_mode: {current_training_mode}. Bitte 'patches' oder 'wide_strips' in der Config setzen.")

if __name__ == "__main__":
    # Diese main-Funktion wird jetzt durch Hydra aufgerufen.
    # Wenn du sie direkt ausführen willst (ohne Hydra), müsstest du die cfg manuell erstellen.
    # z.B. python project/datasets.py data.severstal.training_mode=wide_strips data.severstal.wide_strip_processing.enabled=true
    main()

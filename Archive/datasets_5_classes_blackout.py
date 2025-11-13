# project/datasets.py

import os
import json
from typing import Optional, Tuple, List, Dict
import random # Für __getitem__ in StripDataset und DefectBlackout

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw # ImageDraw für DefectBlackout
import torchvision.transforms as transforms
import hydra # Für die main Testfunktion
from omegaconf import DictConfig # Für die main Testfunktion

import numpy as np
# scipy.ndimage wird für DefectBlackout benötigt
try:
    from scipy import ndimage
except ImportError:
    print("WARNUNG: scipy.ndimage konnte nicht importiert werden. DefectBlackoutTransform wird nicht voll funktionsfähig sein.")
    ndimage = None 

# Annahme: utils.py ist im PYTHONPATH oder selben Verzeichnis
try:
    from utils import create_full_mask_from_png_object
except ImportError:
    print("WARNUNG: utils.create_full_mask_from_png_object konnte nicht importiert werden. Masken können nicht geladen werden.")
    def create_full_mask_from_png_object(b64_data, origin, target_h, target_w): # Dummy
        return np.zeros((target_h, target_w), dtype=np.uint8)

# Für Visualisierung in der main Testfunktion
from matplotlib import pyplot as plt


# --- DefectBlackoutTransform Klasse ---
class DefectBlackoutTransform:
    def __init__(self, 
                 instance_blackout_prob: float = 0.5,
                 min_pixels_to_blackout: int = 10,
                 fill_value: int = 0,
                 verbose: bool = False):
        self.instance_blackout_prob = instance_blackout_prob
        self.min_pixels_to_blackout = min_pixels_to_blackout
        self.fill_value = fill_value
        self.verbose = verbose
        self.relevant_defect_indices = [1, 2, 3, 4] 
        if ndimage is None and verbose:
            print("WARNUNG [DefectBlackoutTransform]: scipy.ndimage nicht verfügbar. Transformation kann nicht korrekt arbeiten.")

    def _blackout_selected_instances(self, image_pil: Image.Image, gt_mask_np: np.ndarray, instances_to_blackout: List[np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        img_out_pil = image_pil.copy()
        mask_out_np = gt_mask_np.copy()
        draw = ImageDraw.Draw(img_out_pil)
        
        for instance_mask_pixels in instances_to_blackout:
            ys, xs = np.where(instance_mask_pixels)
            for y_coord, x_coord in zip(ys, xs):
                draw.point((x_coord, y_coord), fill=self.fill_value)
            mask_out_np[instance_mask_pixels] = 0 # NO_DEFECT_IDX (sollte 0 sein)
        return img_out_pil, mask_out_np

    def blackout_all_instances(self, image_pil: Image.Image, gt_mask_np: np.ndarray) -> Tuple[Image.Image, np.ndarray, bool]:
        if ndimage is None: return image_pil, gt_mask_np, False
        if self.verbose: print(f"[DefectBlackout - ALL] Starte 'blackout_all_instances'.")
        
        mask_for_modification = gt_mask_np.copy()
        all_instances_to_blackout_masks = []
        original_defect_pixels = np.sum(np.isin(gt_mask_np, self.relevant_defect_indices))

        if original_defect_pixels == 0:
            if self.verbose: print("[DefectBlackout - ALL] Keine Defekte vorhanden. Keine Aktion.")
            return image_pil, gt_mask_np, False

        for defect_idx in self.relevant_defect_indices:
            binary_mask_for_class = (mask_for_modification == defect_idx).astype(np.int32)
            labeled_array, num_features = ndimage.label(binary_mask_for_class)
            if num_features == 0: continue
            for i in range(1, num_features + 1):
                instance_mask_pixels = (labeled_array == i)
                if np.sum(instance_mask_pixels) >= self.min_pixels_to_blackout:
                    all_instances_to_blackout_masks.append(instance_mask_pixels)
        
        if not all_instances_to_blackout_masks:
            if self.verbose: print("[DefectBlackout - ALL] Keine qualifizierten Instanzen zum Schwärzen gefunden.")
            return image_pil, gt_mask_np, False

        img_out, mask_out = self._blackout_selected_instances(image_pil, gt_mask_np, all_instances_to_blackout_masks)
        if self.verbose:
            final_defect_pixels = np.sum(np.isin(mask_out, self.relevant_defect_indices))
            print(f"[DefectBlackout - ALL] {len(all_instances_to_blackout_masks)} Instanzen geschwärzt. Verbleibende Defektpixel: {final_defect_pixels}")
        return img_out, mask_out, True

    def __call__(self, image_pil: Image.Image, gt_mask_np: np.ndarray) -> Tuple[Image.Image, np.ndarray, bool]:
        if ndimage is None: return image_pil, gt_mask_np, False
        if self.verbose: print(f"\n[DefectBlackout - SELECTIVE] Starte instanzbasiertes Blackout. Instanz-Prob: {self.instance_blackout_prob}")
        
        mask_for_modification = gt_mask_np.copy()
        instances_randomly_selected_for_blackout = []
        num_total_qualified_instances = 0
        original_defect_pixels = np.sum(np.isin(gt_mask_np, self.relevant_defect_indices))

        if original_defect_pixels == 0:
            if self.verbose: print("[DefectBlackout - SELECTIVE] Keine Defekte vorhanden. Keine Aktion.")
            return image_pil, gt_mask_np, False

        for defect_idx in self.relevant_defect_indices:
            binary_mask_for_class = (mask_for_modification == defect_idx).astype(np.int32)
            labeled_array, num_features = ndimage.label(binary_mask_for_class)
            if num_features == 0: continue
            for i in range(1, num_features + 1):
                instance_mask_pixels = (labeled_array == i)
                if np.sum(instance_mask_pixels) >= self.min_pixels_to_blackout:
                    num_total_qualified_instances +=1
                    if torch.rand(1).item() < self.instance_blackout_prob:
                        instances_randomly_selected_for_blackout.append(instance_mask_pixels)
                        if self.verbose: print(f"[DefectBlackout - SELECTIVE]  - Instanz ({defect_idx},{i}) wird geschwärzt.")
                    elif self.verbose: print(f"[DefectBlackout - SELECTIVE]  - Instanz ({defect_idx},{i}) NICHT geschwärzt.")
        
        if not instances_randomly_selected_for_blackout:
            if self.verbose: print(f"[DefectBlackout - SELECTIVE] Keine Instanzen zufällig zum Schwärzen ausgewählt (aus {num_total_qualified_instances} qualifizierten).")
            return image_pil, gt_mask_np, False

        img_out, mask_out = self._blackout_selected_instances(image_pil, gt_mask_np, instances_randomly_selected_for_blackout)
        if self.verbose:
            final_defect_pixels = np.sum(np.isin(mask_out, self.relevant_defect_indices))
            print(f"[DefectBlackout - SELECTIVE] {len(instances_randomly_selected_for_blackout)} Instanzen geschwärzt. Verbleibende Defektpixel: {final_defect_pixels}")
        return img_out, mask_out, True


# --- SeverstalDataset (Patch-Modus) ---
class SeverstalDataset(Dataset):
    NO_DEFECT_IDX = 0
    DEFECT_CLASS_TO_IDX = {"defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4}
    NUM_TOTAL_CLASSES = 5

    def __init__(self, # ... (Parameter wie in deinem letzten funktionierenden Stand) ...
                 img_dir: str, ann_dir: str, patch_size_hw: Tuple[int, int], stride_hw: Tuple[int, int],
                 transform: Optional[transforms.Compose] = None,
                 max_neg_patches_per_image: Optional[int] = 5,
                 min_positive_pixel_percentage: float = 0.01,
                 debug_limit_images: Optional[int] = None):
        
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.patch_h, self.patch_w = patch_size_hw
        self.stride_h, self.stride_w = stride_hw
        self.transform = transform
        self.max_neg_patches_per_image = max_neg_patches_per_image
        self.min_positive_pixel_abs = min_positive_pixel_percentage * self.patch_h * self.patch_w
        self.patches_info = []

        all_original_image_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        images_to_process = all_original_image_names
        if debug_limit_images is not None and debug_limit_images > 0:
            images_to_process = all_original_image_names[:debug_limit_images]
        
        print(f"SeverstalDataset (Patch Mode): Verarbeite {len(images_to_process)} Originalbilder für Patch-Extraktion.")
        # ... (Rest deiner __init__ Logik für Patch-Extraktion, inklusive der korrigierten
        #      Logik für "kein Defekt"-Patches: elif current_image_has_annotated_defect and np.all(gt_mask_patch == 0): )
        for i_img, img_name in enumerate(images_to_process):
            # ... (Bild laden, Annotationspfad finden) ...
            img_path = os.path.join(self.img_dir, img_name)
            try:
                image_pil_original = Image.open(img_path).convert("RGB")
                original_h, original_w = image_pil_original.height, image_pil_original.width
            except FileNotFoundError: continue # Überspringe, wenn Bild nicht da

            combined_gt_mask_np_original_size = np.zeros((original_h, original_w), dtype=np.uint8)
            current_image_has_annotated_defect = False
            base_name_no_ext = os.path.splitext(img_name)[0]
            ann_path_variants = [ os.path.join(self.ann_dir, f"{img_name}.json"), os.path.join(self.ann_dir, f"{base_name_no_ext}.json")]
            ann_path = None
            for p_variant in ann_path_variants:
                if os.path.exists(p_variant): ann_path = p_variant; break
            
            if ann_path:
                try:
                    with open(ann_path, 'r') as f: annotation = json.load(f)
                    if "objects" in annotation and annotation["objects"]:
                        for obj in annotation["objects"]:
                            class_title = obj.get("classTitle")
                            if class_title in self.DEFECT_CLASS_TO_IDX:
                                current_image_has_annotated_defect = True
                                class_idx_for_mask = self.DEFECT_CLASS_TO_IDX[class_title]
                                if "bitmap" in obj: # ... (Maskenerstellung)
                                    b64_data = obj["bitmap"].get("data"); origin = obj["bitmap"].get("origin")
                                    if b64_data and origin:
                                        single_obj_mask_np = create_full_mask_from_png_object(b64_data, origin, original_h, original_w)
                                        if single_obj_mask_np is not None: combined_gt_mask_np_original_size[single_obj_mask_np == 1] = class_idx_for_mask
                except Exception: pass # Ignoriere Fehler beim Laden einzelner Annotationen für __init__ Robustheit

            neg_patches_count_for_this_image = 0
            for y in range(0, original_h - self.patch_h + 1, self.stride_h):
                for x in range(0, original_w - self.patch_w + 1, self.stride_w):
                    patch_coords_in_original = (x, y)
                    gt_mask_patch = combined_gt_mask_np_original_size[y : y + self.patch_h, x : x + self.patch_w]
                    patch_label_vector = torch.zeros(self.NUM_TOTAL_CLASSES, dtype=torch.float32)
                    unique_classes_in_patch = np.unique(gt_mask_patch)
                    has_defect_in_this_patch = False
                    for class_val in unique_classes_in_patch:
                        if class_val >= 1 and class_val <= 4:
                            num_pixels_for_class = np.sum(gt_mask_patch == class_val)
                            if num_pixels_for_class >= self.min_positive_pixel_abs:
                                patch_label_vector[class_val] = 1.0
                                has_defect_in_this_patch = True
                    
                    if has_defect_in_this_patch:
                        patch_label_vector[self.NO_DEFECT_IDX] = 0.0
                        self.patches_info.append({"image_path": img_path, "coords_in_original": patch_coords_in_original, "label_vector": patch_label_vector})
                    elif current_image_has_annotated_defect and np.all(gt_mask_patch == 0):
                        if self.max_neg_patches_per_image is None or neg_patches_count_for_this_image < self.max_neg_patches_per_image:
                            patch_label_vector[self.NO_DEFECT_IDX] = 1.0
                            self.patches_info.append({"image_path": img_path, "coords_in_original": patch_coords_in_original, "label_vector": patch_label_vector})
                            neg_patches_count_for_this_image += 1
        if not self.patches_info: print("WARNUNG [SeverstalDataset]: Keine Patches konnten extrahiert werden!")
        else: print(f"SeverstalDataset: Patch-Extraktion abgeschlossen. {len(self.patches_info)} Patches indiziert.")

    def __len__(self) -> int: # ... (bleibt gleich) ...
        return len(self.patches_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]: # ... (bleibt gleich) ...
        patch_info = self.patches_info[idx]
        image_path = patch_info["image_path"]; coords = patch_info["coords_in_original"]; label_vector = patch_info["label_vector"]
        try: image_pil_original = Image.open(image_path).convert("RGB")
        except FileNotFoundError: # Fallback
            dummy_patch_tensor = torch.zeros((3, self.patch_h, self.patch_w))
            if self.transform: dummy_patch_tensor = self.transform(Image.new('RGB', (self.patch_w, self.patch_h)))
            return dummy_patch_tensor, label_vector, f"ERROR_PATCH_IMG_NOT_FOUND_{os.path.basename(image_path)}"
        x_start, y_start = coords[0], coords[1]
        patch_pil = image_pil_original.crop((x_start, y_start, x_start + self.patch_w, y_start + self.patch_h))
        transformed_patch_tensor = self.transform(patch_pil) if self.transform else transforms.ToTensor()(patch_pil)
        patch_identifier = f"{os.path.splitext(os.path.basename(image_path))[0]}_x{x_start}_y{y_start}"
        return transformed_patch_tensor, label_vector, patch_identifier


# --- SeverstalStripDataset (Wide Strip-Modus) ---
class SeverstalStripDataset(Dataset):
    NO_DEFECT_IDX = 0
    DEFECT_CLASS_TO_IDX = {"defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4}
    NUM_TOTAL_CLASSES = 5

    def __init__(self,
                 img_dir: str, ann_dir: str, strip_height: int,
                 transform: Optional[transforms.Compose] = None,
                 use_defect_blackout: bool = False, defect_blackout_prob: float = 0.5,
                 target_no_defect_ratio: Optional[float] = None,
                 instance_blackout_prob_selective: float = 0.5,
                 blackout_min_pixels: int = 10,
                 verbose_strip_dataset_debug: bool = False,
                 debug_limit_images: Optional[int] = None,
                 image_filenames_to_use: Optional[List[str]] = None):
        
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.strip_height = strip_height # Korrekte Zuweisung
        self.transform = transform
        self.use_defect_blackout = use_defect_blackout
        self.defect_blackout_prob = defect_blackout_prob
        self.target_no_defect_ratio = target_no_defect_ratio
        self.verbose_debug = verbose_strip_dataset_debug

        self.defect_blackout_transform = None
        if self.use_defect_blackout:
            self.defect_blackout_transform = DefectBlackoutTransform(
                instance_blackout_prob=instance_blackout_prob_selective,
                min_pixels_to_blackout=blackout_min_pixels,
                verbose=self.verbose_debug
            )
            if self.defect_blackout_transform is not None and self.verbose_debug:
                 print(f"[DEBUG SeverstalStripDataset] DefectBlackoutTransform.verbose ist auf: {self.defect_blackout_transform.verbose}")

        if image_filenames_to_use is not None:
            self.image_files = sorted(list(set(image_filenames_to_use)))
            if self.verbose_debug: print(f"SeverstalStripDataset: Initialisiert mit spezifischer Liste von {len(self.image_files)} Bildern.")
        else:
            all_image_files_in_dir = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if debug_limit_images is not None and debug_limit_images > 0:
                self.image_files = all_image_files_in_dir[:debug_limit_images]
                if self.verbose_debug: print(f"SeverstalStripDataset: Debug-Limit ({debug_limit_images}) aktiv. Verwende die ersten {len(self.image_files)} Bilder aus {img_dir}.")
            else:
                self.image_files = all_image_files_in_dir
                if self.verbose_debug: print(f"SeverstalStripDataset: Verarbeite alle {len(self.image_files)} Originalbilder aus {img_dir}.")
        
        if not self.image_files: print("WARNUNG [SeverstalStripDataset]: Keine Bilddateien zum Verarbeiten!")
        else: print(f"SeverstalStripDataset: Finale Anzahl Bilder: {len(self.image_files)} (Annahme: img_dir enthält nur Bilder mit annotierten Defekten).")

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_combined_gt_mask(self, ann_path: str, original_h: int, original_w: int) -> np.ndarray:
        combined_gt_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f: annotation = json.load(f)
                if "objects" in annotation and annotation["objects"]:
                    for obj in annotation["objects"]:
                        class_title = obj.get("classTitle")
                        if class_title in self.DEFECT_CLASS_TO_IDX:
                            class_idx_for_mask = self.DEFECT_CLASS_TO_IDX[class_title]
                            if "bitmap" in obj:
                                b64_data = obj["bitmap"].get("data"); origin = obj["bitmap"].get("origin")
                                if b64_data and origin:
                                    single_obj_mask_np = create_full_mask_from_png_object(b64_data, origin, original_h, original_w)
                                    if single_obj_mask_np is not None: combined_gt_mask[single_obj_mask_np == 1] = class_idx_for_mask
            except Exception as e_json:
                if self.verbose_debug: print(f"Fehler beim Verarbeiten von JSON {ann_path} in _load_combined_gt_mask: {e_json}")
        elif self.verbose_debug: print(f"WARNUNG: Annotationsdatei {ann_path} nicht gefunden in _load_combined_gt_mask.")
        return combined_gt_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Konstruiere Annotationspfad
        base_name_no_ext = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.ann_dir, f"{base_name_no_ext}.json")
        if not os.path.exists(ann_path): # Fallback für img_name.json (z.B. bild.jpg.json)
             ann_path = os.path.join(self.ann_dir, f"{img_name}.json")

        try: image_pil_original = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"WARNUNG (StripDataset): Bild {img_path} nicht gefunden in __getitem__.")
            dummy_strip_tensor = torch.zeros((3, self.strip_height, 1600)) 
            dummy_label_vector = torch.zeros(self.NUM_TOTAL_CLASSES, dtype=torch.float32); dummy_label_vector[self.NO_DEFECT_IDX] = 1.0
            return dummy_strip_tensor, dummy_label_vector, f"ERROR_IMG_NOT_FOUND_{img_name}"

        original_w, original_h = image_pil_original.size
        gt_mask_original_np = self._load_combined_gt_mask(ann_path, original_h, original_w)

        current_strip_width = original_w
        if original_h <= self.strip_height:
            y_start_strip = 0; img_strip_pil = image_pil_original.copy(); gt_mask_strip_np = gt_mask_original_np.copy()
        else:
            y_start_strip = torch.randint(0, original_h - self.strip_height + 1, (1,)).item()
            img_strip_pil = image_pil_original.crop((0, y_start_strip, current_strip_width, y_start_strip + self.strip_height))
            gt_mask_strip_np = gt_mask_original_np[y_start_strip : y_start_strip + self.strip_height, :]
        
        img_after_blackout_pil = img_strip_pil
        final_gt_mask_for_labeling_strip_np = gt_mask_strip_np.copy()
        attempt_blackout = False
        if self.use_defect_blackout and self.defect_blackout_transform is not None:
            if torch.rand(1).item() < self.defect_blackout_prob: attempt_blackout = True

        if attempt_blackout:
            if self.target_no_defect_ratio is not None and torch.rand(1).item() < self.target_no_defect_ratio:
                if self.verbose_debug: print(f"[Blackout __getitem__] {img_name}: Versuche, ALLE Defekte zu schwärzen.")
                temp_img, temp_mask, applied_all = self.defect_blackout_transform.blackout_all_instances(img_strip_pil, final_gt_mask_for_labeling_strip_np)
                if applied_all and np.sum(np.isin(temp_mask, self.defect_blackout_transform.relevant_defect_indices)) == 0:
                    img_after_blackout_pil, final_gt_mask_for_labeling_strip_np = temp_img, temp_mask
                elif self.verbose_debug: print(f"[Blackout __getitem__] {img_name}: 'Alle schwärzen' hat nicht alle Defekte entfernt oder wurde nicht angewendet. Fallback zu selektiv.")
                if not (applied_all and np.sum(np.isin(temp_mask, self.defect_blackout_transform.relevant_defect_indices)) == 0) : # Wenn "alle schwärzen" nicht erfolgreich war, mache selektiv
                    img_after_blackout_pil, final_gt_mask_for_labeling_strip_np, _ = self.defect_blackout_transform(img_strip_pil, final_gt_mask_for_labeling_strip_np)
            else:
                 if self.verbose_debug: print(f"[Blackout __getitem__] {img_name}: Führe instanzbasierten selektiven Blackout durch.")
                 img_after_blackout_pil, final_gt_mask_for_labeling_strip_np, _ = self.defect_blackout_transform(img_strip_pil, final_gt_mask_for_labeling_strip_np)
        
        strip_label_vector = torch.zeros(self.NUM_TOTAL_CLASSES, dtype=torch.float32)
        unique_classes_in_final_strip = np.unique(final_gt_mask_for_labeling_strip_np)
        has_defect_in_final_strip = False; min_pixels_for_defect_in_strip = 1
        for class_val in unique_classes_in_final_strip:
            if 1 <= class_val <= 4:
                num_pixels = np.sum(final_gt_mask_for_labeling_strip_np == class_val)
                if num_pixels >= min_pixels_for_defect_in_strip:
                    strip_label_vector[class_val] = 1.0; has_defect_in_final_strip = True
        
        if not has_defect_in_final_strip: 
            strip_label_vector[self.NO_DEFECT_IDX] = 1.0
            if self.verbose_debug: print(f"[Labeling {img_name}] Streifen als 'Kein Defekt' gelabelt (nach Blackout).")
        else: 
            strip_label_vector[self.NO_DEFECT_IDX] = 0.0
            if self.verbose_debug: print(f"[Labeling {img_name}] Streifen mit Defekten gelabelt: {strip_label_vector.numpy().astype(int)}")

        transformed_strip_tensor = self.transform(img_after_blackout_pil) if self.transform else transforms.ToTensor()(img_after_blackout_pil)
        identifier = f"{os.path.splitext(img_name)[0]}_strip_y{y_start_strip}"
        return transformed_strip_tensor, strip_label_vector, identifier


# --- main Funktion zum Testen der DataLoaders ---
@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # ... (Rest der main-Funktion wie im letzten funktionierenden Entwurf für den Test) ...
    # Stelle sicher, dass die Pfade (img_dir_abs, ann_dir_abs) und Parameter
    # korrekt aus cfg für den jeweiligen Modus gelesen werden.
    # Die Visualisierungslogik kann beibehalten werden.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_dir_abs = os.path.join(project_root, cfg.data.severstal.img_dir)
    ann_dir_abs = os.path.join(project_root, cfg.data.severstal.ann_dir)
    current_training_mode = cfg.data.severstal.get("training_mode", "patches")
    print(f"\nTestmodus: {current_training_mode}")

    if current_training_mode == "patches":
        # ... (Dein Testcode für SeverstalDataset) ...
        print("Patch-Modus Test wird ausgeführt (Logik von dir).")
        patch_proc_cfg = cfg.data.severstal.patch_processing
        patch_size_hw = tuple(patch_proc_cfg.patch_size_hw); stride_hw = tuple(patch_proc_cfg.stride_hw)
        max_neg = patch_proc_cfg.get("max_neg_patches_per_image", 5); min_pos_perc = patch_proc_cfg.get("min_positive_pixel_percentage", 0.01)
        debug_img_limit_patches_test = cfg.data.severstal.patch_processing.get("debug_limit_images_patching", 10)
        patch_test_transform = transforms.Compose([transforms.Resize(patch_size_hw), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        sever_patch_dataset = SeverstalDataset(img_dir=img_dir_abs, ann_dir=ann_dir_abs, patch_size_hw=patch_size_hw, stride_hw=stride_hw, transform=patch_test_transform, max_neg_patches_per_image=max_neg, min_positive_pixel_percentage=min_pos_perc, debug_limit_images=debug_img_limit_patches_test)
        if len(sever_patch_dataset) > 0:
            loader = torch.utils.data.DataLoader(sever_patch_dataset, batch_size=min(4, len(sever_patch_dataset)), shuffle=True)
            patches, labels, ids = next(iter(loader))
            print(f"Patch-Modus: Batch Shapes - Patches: {patches.shape}, Labels: {labels.shape}")
            # (Visualisierung für Patch-Modus hier, falls gewünscht)

    elif current_training_mode == "wide_strips":
        print("Wide Strip-Modus Test wird ausgeführt.")
        strip_cfg = cfg.data.severstal.wide_strip_processing
        if not strip_cfg.get("enabled", True): print("Strip-Modus nicht aktiviert in Config."); return # Default true für Test hier
        
        strip_height = strip_cfg.strip_height
        debug_limit_strips_test = cfg.data.severstal.patch_processing.get("debug_limit_images_patching", 5) # Nimm ein gemeinsames Debug-Limit für den Test
        
        # Parameter für StripDataset direkt aus der strip_cfg holen
        use_blackout = strip_cfg.get("use_defect_blackout", True) # Default True für Test
        blackout_prob = strip_cfg.get("defect_blackout_prob", 1.0) # Default 1.0 für Test
        target_no_defect = strip_cfg.get("target_no_defect_ratio", 0.5) # Default 0.5 für Test
        instance_blackout_p = strip_cfg.get("instance_blackout_prob_selective", 0.5)
        blackout_min_px = strip_cfg.get("blackout_min_pixels", 10)
        verbose_strip_debug = strip_cfg.get("verbose_strip_dataset_debug", True) # Default True für Test

        example_strip_width = 1600 # Für Resize in Test-Transform
        strip_test_transform = transforms.Compose([
            transforms.Resize((strip_height, example_strip_width)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Teste mit einer expliziten Liste von Bildern (wenn vorhanden) oder allen (limitierten)
        # Für den __main__ Test nehmen wir an, wir verwenden debug_limit_images und nicht image_filenames_to_use
        strip_dataset = SeverstalStripDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs,
            strip_height=strip_height, transform=strip_test_transform,
            use_defect_blackout=use_blackout, defect_blackout_prob=blackout_prob,
            target_no_defect_ratio=target_no_defect,
            instance_blackout_prob_selective=instance_blackout_p,
            blackout_min_pixels=blackout_min_px,
            verbose_strip_dataset_debug=verbose_strip_debug,
            debug_limit_images=debug_limit_strips_test
        )

        if len(strip_dataset) > 0:
            loader = torch.utils.data.DataLoader(strip_dataset, batch_size=min(2, len(strip_dataset)), shuffle=True) # Kleiner Batch für Streifen
            strip_tensors, label_vectors, identifiers = next(iter(loader)) # Hole einen Batch
            print(f"Strip-Modus: Batch Shapes - Strips: {strip_tensors.shape}, Labels: {label_vectors.shape}")
            print(f"  Erstes Label: {label_vectors[0].numpy().astype(int)}, ID: {identifiers[0]}")

            # Visualisiere den ersten Streifen des Batches
            plt.figure(figsize=(12, 4))
            strip_to_show = strip_tensors[0].permute(1,2,0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
            strip_to_show = std * strip_to_show + mean; strip_to_show = np.clip(strip_to_show, 0, 1)
            plt.imshow(strip_to_show); plt.title(f"Strip Sample: {identifiers[0]}\nLabel: {label_vectors[0].numpy().astype(int)}")
            plt.savefig("strip_dataset_sample_via_main.png"); print("Beispiel-Streifen als 'strip_dataset_sample_via_main.png' gespeichert.")
            plt.close()
        else: print("StripDataset ist leer.")
    else: print(f"Unbekannter training_mode: {current_training_mode}")

if __name__ == "__main__":
    main()
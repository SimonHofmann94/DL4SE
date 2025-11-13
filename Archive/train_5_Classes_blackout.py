# project/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import hydra
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler # Subset, WeightedRandomSampler importieren
import torchvision.transforms as transforms 
from omegaconf import DictConfig, OmegaConf 
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score 
from PIL import Image


# Importiere beide Dataset-Klassen und ggf. FocalLoss
from project.datasets import SeverstalDataset, SeverstalStripDataset 
from project.models import ClassifierModel
from project.utils import set_seed, EarlyStopping
from collections import Counter, defaultdict 

# Optional: Focal Loss (wenn du sie in einer separaten Datei hast)
try:
    from utils import FocalLoss
except ImportError:
    print("WARNUNG: FocalLoss konnte nicht von project.losses importiert werden. Stelle sicher, dass die Datei und Klasse existieren.")
    FocalLoss = None # Setze auf None, wenn nicht verfügbar

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


def analyze_patch_dataset_composition(subset_dataset: Subset, dataset_name: str, underlying_patch_dataset_info: list):
    """ Analysiert Patch-Dataset Zusammensetzung. """
    print(f"\n--- Analyse der Zusammensetzung des {dataset_name} (Patch-Modus) ---")
    image_patch_counts = defaultdict(lambda: {"total": 0, "no_defect": 0, "defect_1": 0, "defect_2": 0, "defect_3": 0, "defect_4": 0})
    if not subset_dataset.indices: print(f"Das {dataset_name} enthält keine Patches."); return
    for patch_idx_in_original_list in subset_dataset.indices:
        patch_info = underlying_patch_dataset_info[patch_idx_in_original_list]
        image_basename = os.path.basename(patch_info["image_path"])
        label_vector = patch_info["label_vector"]
        image_patch_counts[image_basename]["total"] += 1
        if label_vector[SeverstalDataset.NO_DEFECT_IDX].item() == 1.0: image_patch_counts[image_basename]["no_defect"] += 1
        for i in range(1, 5): # Defektklassen 1-4
             if label_vector[i].item() == 1.0: image_patch_counts[image_basename][f"defect_{i}"] += 1
    print(f"Das {dataset_name} wurde aus {len(image_patch_counts)} einzigartigen Originalbildern generiert.")
    print(f"Gesamtzahl der Patches im {dataset_name}: {len(subset_dataset.indices)}")
    for img_name, counts in sorted(image_patch_counts.items()):
        print(f"  {img_name}: {counts['total']} Patches [{counts['no_defect']},{counts['defect_1']},{counts['defect_2']},{counts['defect_3']},{counts['defect_4']}]")
    print(f"--- Ende der Analyse für {dataset_name} ---\n")


def get_transforms(cfg_augment_dict: dict, img_h_config: int, img_w_config: int, is_train: bool = True, for_strip_mode: bool = False):
    aug = cfg_augment_dict 
    pil_transforms_list = []
    verbose = aug.get("verbose_transforms", False)

    # Resize-Logik
    resize_applied = False
    if is_train and aug.get("use_resizedcrop", False) and not for_strip_mode: # RRC eher für Patches
        if img_h_config > 0 and img_w_config > 0:
            pil_transforms_list.append(transforms.RandomResizedCrop((img_h_config, img_w_config), scale=(0.8, 1.0), ratio=(0.9, 1.1)))
            if verbose: print(f"TRAIN TF: RandomResizedCrop to ({img_h_config},{img_w_config})")
            resize_applied = True
        elif img_h_config > 0 and verbose:
            print(f"TRAIN TF WARN: use_resizedcrop=True, aber ungültige Breite ({img_w_config}). Kein RRC angewendet.")
    
    if not resize_applied: # Wenn RRC nicht angewendet wurde (oder nicht aktiv war)
        if img_h_config > 0 and img_w_config > 0 and not for_strip_mode: # Feste (h,w) für Patches
            pil_transforms_list.append(transforms.Resize((img_h_config, img_w_config)))
            if verbose: print(f"{'TRAIN' if is_train else 'VAL/TEST'} TF: Resize to ({img_h_config},{img_w_config})")
        elif img_h_config > 0 : # Nur Höhe resizen (gut für Streifen oder wenn Breite variabel bleiben soll)
            pil_transforms_list.append(transforms.Resize(img_h_config)) # Passt Höhe an, behält Aspekt
            if verbose: print(f"{'TRAIN' if is_train else 'VAL/TEST'} TF: Resize Höhe auf {img_h_config}, Breite proportional.")
    
    # Standard Augmentierungen
    if is_train:
        if aug.get("use_hflip", False): pil_transforms_list.append(transforms.RandomHorizontalFlip())
        if not for_strip_mode: # Vflip und Rotation eher für Patches
            if aug.get("use_vflip", False): pil_transforms_list.append(transforms.RandomVerticalFlip())
            if aug.get("use_rotation", False): pil_transforms_list.append(transforms.RandomRotation(aug.get("rotation_degree", 15)))
        if aug.get("use_colorjitter", False):
            pil_transforms_list.append(transforms.ColorJitter(brightness=aug.get("cj_brightness",0.2), contrast=aug.get("cj_contrast",0.2), saturation=aug.get("cj_saturation",0.2), hue=aug.get("cj_hue",0.1)))
    
    pil_transforms_list.append(transforms.ToTensor())
    tensor_transforms_list = []
    if is_train:
        if aug.get("use_blur", False): tensor_transforms_list.append(transforms.GaussianBlur(kernel_size=tuple(aug.get("blur_kernel",[3,3])), sigma=tuple(aug.get("blur_sigma",[0.1,2.0]))))
        if aug.get("use_erasing", False): tensor_transforms_list.append(transforms.RandomErasing(p=aug.get("erasing_p",0.5), scale=tuple(aug.get("erasing_scale",[0.02,0.33])), ratio=tuple(aug.get("erasing_ratio",[0.3,3.3]))))
    
    tensor_transforms_list.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    if verbose: 
        print(f"  {'TRAIN' if is_train else 'VAL/TEST'} PIL Transforms: {[type(t).__name__ for t in pil_transforms_list]}")
        print(f"  {'TRAIN' if is_train else 'VAL/TEST'} Tensor Transforms: {[type(t).__name__ for t in tensor_transforms_list]}")
    return transforms.Compose(pil_transforms_list + tensor_transforms_list)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Verwende Device: {device}")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_dir_abs = os.path.join(project_root, cfg.data.severstal.img_dir)
    ann_dir_abs = os.path.join(project_root, cfg.data.severstal.ann_dir)

    current_training_mode = cfg.data.severstal.get("training_mode", "patches")
    tb_log_subdir = f"{cfg.model.backbone}/{cfg.data.name}_{current_training_mode}" # cfg.data.name verwenden
    tb_log_abs_dir = os.path.join(project_root, cfg.train.tb_log_dir, tb_log_subdir)
    os.makedirs(tb_log_abs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_abs_dir)
    print(f"TensorBoard Logs in: {tb_log_abs_dir}")

    debug_orig_imgs = cfg.data.severstal.patch_processing.get("debug_limit_images_patching", None)
    if current_training_mode == "wide_strips":
        debug_orig_imgs = cfg.data.severstal.wide_strip_processing.get("debug_limit_images_strips", debug_orig_imgs)

    cfg_augment_dict = OmegaConf.to_container(cfg.augment, resolve=True) if cfg.augment else {}
    train_dataset, val_dataset = None, None
    num_labels_for_model = 5 # Standard
    class_positive_counts_train = torch.zeros(num_labels_for_model, device=device) # Initialisiere hier
    n_train_total_items = 0 # Initialisiere hier


    if current_training_mode == "patches":
        print("\n--- Modus: Training mit Patches ---")
        patch_cfg = cfg.data.severstal.patch_processing
        if not patch_cfg.get("enabled", True): print("Patch-Verarbeitung nicht aktiviert. Abbruch."); writer.close(); return
        patch_size_hw = tuple(patch_cfg.patch_size_hw); stride_hw = tuple(patch_cfg.stride_hw)
        patch_h, patch_w = patch_size_hw[0], patch_size_hw[1]
        train_tf = get_transforms(cfg_augment_dict, patch_h, patch_w, is_train=True, for_strip_mode=False)
        val_tf = get_transforms(cfg_augment_dict, patch_h, patch_w, is_train=False, for_strip_mode=False)
        
        dataset_for_splitting = SeverstalDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs, patch_size_hw=patch_size_hw, stride_hw=stride_hw, 
            transform=train_tf, # Wichtig: train_tf für das Dataset, das gesplittet wird
            max_neg_patches_per_image=patch_cfg.max_neg_patches_per_image, 
            min_positive_pixel_percentage=patch_cfg.min_positive_pixel_percentage,
            debug_limit_images=debug_orig_imgs
        )
        if not dataset_for_splitting: print("Severstal Patch Dataset ist leer. Abbruch."); writer.close(); return

        n_total = len(dataset_for_splitting); n_train = int(n_total * cfg.data.severstal.severstal_train_split_ratio); n_val = n_total - n_train
        train_indices_obj, val_indices_obj = random_split(range(n_total), [n_train, n_val], generator=torch.Generator().manual_seed(cfg.train.seed))
        
        train_dataset = Subset(dataset_for_splitting, train_indices_obj.indices)
        # Val-Dataset mit val_tf - erzeuge neue Instanz für saubere Transforms
        val_dataset_instance = SeverstalDataset(
             img_dir=img_dir_abs, ann_dir=ann_dir_abs, patch_size_hw=patch_size_hw, stride_hw=stride_hw, 
            transform=val_tf, # Hier val_tf
            max_neg_patches_per_image=patch_cfg.max_neg_patches_per_image, 
            min_positive_pixel_percentage=patch_cfg.min_positive_pixel_percentage,
            debug_limit_images=debug_orig_imgs
        )
        val_dataset = Subset(val_dataset_instance, val_indices_obj.indices)
        
        analyze_patch_dataset_composition(train_dataset, "Trainingsset (Patches)", dataset_for_splitting.patches_info)
        
        # pos_weight für Patches
        num_labels_for_model = SeverstalDataset.NUM_TOTAL_CLASSES
        for idx in train_dataset.indices: class_positive_counts_train += dataset_for_splitting.patches_info[idx]["label_vector"].to(device)
        n_train_total_items = len(train_dataset)

    elif current_training_mode == "wide_strips":
        print("\n--- Modus: Training mit Wide Strips ---")
        strip_cfg = cfg.data.severstal.wide_strip_processing
        if not strip_cfg.get("enabled", True): print("Wide Strip-Verarbeitung nicht aktiviert. Abbruch."); writer.close(); return
        
        strip_h = strip_cfg.strip_height
        strip_w_for_tf = strip_cfg.get("random_crop_width", -1) # -1 für variable Breite in get_transforms
        train_tf = get_transforms(cfg_augment_dict, strip_h, strip_w_for_tf, is_train=True, for_strip_mode=True)
        val_tf = get_transforms(cfg_augment_dict, strip_h, strip_w_for_tf, is_train=False, for_strip_mode=True)

        # Hole Liste aller Bilddateien (Annahme: img_dir ist vor-gefiltert)
        initial_strip_ds_for_list = SeverstalStripDataset(img_dir=img_dir_abs, ann_dir=ann_dir_abs, strip_height=strip_h, debug_limit_images=debug_orig_imgs)
        all_image_filenames = initial_strip_ds_for_list.image_files
        if not all_image_filenames: print("Keine Bilder für Strip-Modus gefunden. Abbruch."); writer.close(); return

        n_total = len(all_image_filenames); n_train = int(n_total * cfg.data.severstal.severstal_train_split_ratio); n_val = n_total - n_train
        indices_to_split = list(range(n_total))
        train_idx_list_obj, val_idx_list_obj = random_split(indices_to_split, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.train.seed))
        
        train_files = [all_image_filenames[i] for i in train_idx_list_obj.indices]
        val_files = [all_image_filenames[i] for i in val_idx_list_obj.indices]

        print(f"Erstelle Trainings-Dataset (Strips) mit {len(train_files)} Bildern...")
        train_dataset = SeverstalStripDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs, strip_height=strip_h, transform=train_tf,
            use_defect_blackout=strip_cfg.use_defect_blackout, defect_blackout_prob=strip_cfg.defect_blackout_prob,
            target_no_defect_ratio=strip_cfg.target_no_defect_ratio,
            instance_blackout_prob_selective=strip_cfg.instance_blackout_prob_selective,
            blackout_min_pixels=strip_cfg.blackout_min_pixels,
            verbose_strip_dataset_debug=strip_cfg.verbose_strip_dataset_debug,
            image_filenames_to_use=train_files
        )
        print(f"Erstelle Validierungs-Dataset (Strips) mit {len(val_files)} Bildern...")
        val_dataset = SeverstalStripDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs, strip_height=strip_h, transform=val_tf,
            use_defect_blackout=False, # Kein Blackout für Validierung
            verbose_strip_dataset_debug=strip_cfg.verbose_strip_dataset_debug,
            image_filenames_to_use=val_files
        )
        num_labels_for_model = SeverstalStripDataset.NUM_TOTAL_CLASSES
        # pos_weight für Streifen (basierend auf Original-Annotationen der Trainingsbilder)
        print("Berechne pos_weight für Streifen-Modus...")
        for img_fname in train_files:
            ann_path = os.path.join(ann_dir_abs, os.path.splitext(img_fname)[0] + ".json")
            if not os.path.exists(ann_path): ann_path = os.path.join(ann_dir_abs, img_fname + ".json") # Fallback
            if not os.path.exists(ann_path): continue
            try:
                with Image.open(os.path.join(img_dir_abs, img_fname)) as temp_img: o_h, o_w = temp_img.height, temp_img.width
                # Verwende _load_combined_gt_mask der train_dataset Instanz (oder initial_strip_ds_for_list)
                original_gt_mask = train_dataset.dataset._load_combined_gt_mask(ann_path, o_h, o_w) if isinstance(train_dataset, Subset) else train_dataset._load_combined_gt_mask(ann_path, o_h, o_w)
                
                unique_classes_in_orig_img = np.unique(original_gt_mask)
                for class_val in unique_classes_in_orig_img:
                    if 1 <= class_val <= 4 and np.sum(original_gt_mask == class_val) > 0:
                        class_positive_counts_train[class_val] += 1
            except Exception as e: print(f"WARN (pos_weight strip): Fehler bei {img_fname}: {e}")
        
        if strip_cfg.use_defect_blackout and strip_cfg.target_no_defect_ratio and strip_cfg.defect_blackout_prob > 0:
            prob_no_defect = strip_cfg.defect_blackout_prob * strip_cfg.target_no_defect_ratio
            class_positive_counts_train[SeverstalStripDataset.NO_DEFECT_IDX] = len(train_files) * prob_no_defect
        n_train_total_items = len(train_files)
    else:
        print(f"FEHLER: Unbekannter training_mode: {current_training_mode}. Abbruch."); writer.close(); return

    if not train_dataset or not val_dataset or len(train_dataset) == 0: print("Leeres Dataset. Abbruch."); writer.close(); return
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)
    print(f"DataLoaders erstellt: Train {len(train_loader)} Batches, Val {len(val_loader)} Batches.")

    # --- Loss-Gewichte ---
    pos_weight_tensor = torch.ones(num_labels_for_model, device=device) # Default
    if n_train_total_items > 0:
        for c in range(num_labels_for_model):
            pos_c = class_positive_counts_train[c].item(); neg_c = n_train_total_items - pos_c
            if pos_c > 0 and neg_c > 0: pos_weight_tensor[c] = neg_c / pos_c
        pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=cfg.train.get("max_pos_weight", 30.0))
    print(f"Positive Zählungen (für Loss): {class_positive_counts_train.cpu().numpy()}")
    print(f"Finaler pos_weight Tensor: {pos_weight_tensor.cpu().numpy()}")

    # --- Modell, Verlustfunktion, Optimizer ---
    model = ClassifierModel(backbone_name=cfg.model.backbone, pretrained=cfg.model.pretrained, num_classes=num_labels_for_model).to(device)
    criterion = None
    if cfg.train.loss_function == "focal" and FocalLoss is not None:
        focal_cfg = cfg.train.focal_loss; alpha_tensor = None
        if focal_cfg.alpha_mode == "fixed" and focal_cfg.get("alpha_fixed"):
            alpha_list = OmegaConf.to_container(focal_cfg.alpha_fixed); 
            if len(alpha_list) == num_labels_for_model: alpha_tensor = torch.tensor(alpha_list, device=device, dtype=torch.float32)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=focal_cfg.gamma)
        print(f"Verwende FocalLoss (gamma={criterion.gamma}, alpha={criterion.alpha.cpu().numpy() if criterion.alpha is not None else 'None'})")
    else:
        if cfg.train.loss_function == "focal": print("WARNUNG: FocalLoss angefordert, aber nicht verfügbar/konfiguriert. Fallback zu BCE.")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Verwende BCEWithLogitsLoss (pos_weight={pos_weight_tensor.cpu().numpy()})")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.max_epochs)
    
    # --- Trainings-Loop ---
    best_f1_micro_val = 0.0
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max", verbose=True, path=os.path.join(tb_log_abs_dir, "best_model.pt")) # Pfad für ES anpassen

    print(f"\nStarte Training für {cfg.train.max_epochs} Epochen...")
    for epoch in range(cfg.train.max_epochs):
        model.train()
        train_loss_epoch = 0.0
        # ... (Rest des Trainings- und Validierungsloops bleibt identisch) ...
        # Stelle sicher, dass die Batch-Entpackung für beide Dataset-Typen passt
        # (beide geben image, label, identifier zurück)
        for batch_idx, (images, labels, _) in enumerate(train_loader): # 'patches' zu 'images' umbenannt
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images) # 'images' verwenden
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            if (batch_idx + 1) % cfg.train.get("log_interval", 100) == 0: # Log-Intervall erhöht
                print(f"  Epoch {epoch+1}/{cfg.train.max_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate", current_lr, epoch)
        print(f"Epoch {epoch+1} abgeschlossen. Avg Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")

        # Validierung
        model.eval()
        val_loss_epoch = 0.0
        all_labels_val_list, all_probs_val_list = [], []
        with torch.no_grad():
            for images_val, labels_val, _ in val_loader: # 'patches' zu 'images_val'
                images_val, labels_val = images_val.to(device), labels_val.to(device).float()
                outputs_val = model(images_val) # 'images_val' verwenden
                loss_val = criterion(outputs_val, labels_val)
                val_loss_epoch += loss_val.item()
                all_probs_val_list.append(torch.sigmoid(outputs_val).cpu())
                all_labels_val_list.append(labels_val.cpu())
        
        avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        if not all_labels_val_list: # Verhindere Fehler, wenn val_loader leer ist
             print("Validierungsset leer, überspringe Metrikberechnung.")
             scheduler.step(); continue


        all_labels_val_np = torch.cat(all_labels_val_list).numpy()
        all_probs_val_np = torch.cat(all_probs_val_list).numpy()
        
        # Threshold-Optimierung und Metriken (bleibt gleich)
        thresholds = np.arange(0.1, 0.91, 0.05); current_epoch_best_f1_micro = 0.0; current_epoch_best_threshold = 0.5
        for t in thresholds:
            preds_val = (all_probs_val_np > t).astype(int)
            f1_t = f1_score(all_labels_val_np, preds_val, average='micro', zero_division=0)
            if f1_t > current_epoch_best_f1_micro: current_epoch_best_f1_micro = f1_t; current_epoch_best_threshold = t
        
        final_preds_val = (all_probs_val_np > current_epoch_best_threshold).astype(int)
        precision_val_micro = precision_score(all_labels_val_np, final_preds_val, average='micro', zero_division=0)
        recall_val_micro = recall_score(all_labels_val_np, final_preds_val, average='micro', zero_division=0)
        f1_per_class_val = f1_score(all_labels_val_np, final_preds_val, average=None, zero_division=0)

        print(f"[Val] Epoch {epoch+1} - Loss: {avg_val_loss:.4f}, BestTh: {current_epoch_best_threshold:.2f} -> F1_μ: {current_epoch_best_f1_micro:.4f}, P_μ: {precision_val_micro:.4f}, R_μ: {recall_val_micro:.4f}")
        f1_class_strings = [f"{f1:.3f}" for f1 in f1_per_class_val]
        print(f"  F1 per class (0..{num_labels_for_model-1}): [{', '.join(f1_class_strings)}]")

        writer.add_scalar("Val/F1_micro", current_epoch_best_f1_micro, epoch); writer.add_scalar("Val/Precision_micro", precision_val_micro, epoch)
        writer.add_scalar("Val/Recall_micro", recall_val_micro, epoch); writer.add_scalar("Val/Best_Threshold", current_epoch_best_threshold, epoch)
        for i, f1_class in enumerate(f1_per_class_val): writer.add_scalar(f"Val_F1_Class/Class_{i}", f1_class, epoch)

        if current_epoch_best_f1_micro > best_f1_micro_val:
            best_f1_micro_val = current_epoch_best_f1_micro
            # Pfad für bestes Modell anpassen, um Modus im Namen zu haben
            best_model_filename = f"best_model_severstal_{current_training_mode}.pt"
            best_model_path = os.path.join(tb_log_abs_dir, best_model_filename) # Speichere im TB-Ordner
            torch.save(model.state_dict(), best_model_path)
            print(f"Neues bestes Modell gespeichert: {best_model_path} (F1_micro: {best_f1_micro_val:.4f})")
            # Update EarlyStopping path as well
            stopper.path = best_model_path 
        
        if stopper.step(current_epoch_best_f1_micro, model): # Modell für ES übergeben, falls es dort speichert
            print("Early stopping ausgelöst."); break
        scheduler.step()

    print(f"\nTraining abgeschlossen. Bestes F1_micro auf Validierung: {best_f1_micro_val:.4f}")
    writer.close()

if __name__ == "__main__":
    main()
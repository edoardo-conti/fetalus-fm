import sys
import os
import random
import numpy as np
import json
import torch
import torch.nn as nn
from pathlib import Path
from augmentations import geometric_transforms, color_transforms
from datasets.unified import UnifiedFetalDataset
from datetime import datetime
from matplotlib import pyplot as plt
from utils.utils import DEFAULT_IMAGE_SIZE
from utils.utils import FUS_STRUCTS_COLORS, mask2d_to_rgb
from foundation.model import DINOSegmentator

image_size = (224, 224)
train_geo_tfms = geometric_transforms(image_size)
train_color_tfms = color_transforms()

fus_train = UnifiedFetalDataset(
    root='/leonardo_work/IscrC_FoSAM-X/fetalus-fm',
    data_path='/leonardo_scratch/fast/IscrC_FoSAM-X/datasets',
    datasets=["HC18", "FABD", "FPDB", "IPSFH", "FPLR", "ACSLC"],
    split='train',
    supervised=True,
    target_size=image_size,
    augmentations=(train_geo_tfms, train_color_tfms)
)
fus_val = UnifiedFetalDataset(
    root='/leonardo_work/IscrC_FoSAM-X/fetalus-fm',
    data_path='/leonardo_scratch/fast/IscrC_FoSAM-X/datasets',
    datasets=fus_train.datasets,
    split='val',
    supervised=True,
    target_size=image_size,
)
fus_test = UnifiedFetalDataset(
    root='/leonardo_work/IscrC_FoSAM-X/fetalus-fm',
    data_path='/leonardo_scratch/fast/IscrC_FoSAM-X/datasets',
    datasets=fus_train.datasets,
    split='test',
    supervised=True,
    target_size=image_size,
)

total_images = len(fus_train) + len(fus_val) + len(fus_test)
print(f"Total images in sets: {total_images}")
print(f"Train dataset size: {len(fus_train)} ({len(fus_train) / total_images:.2%})")
print(f"Validation dataset size: {len(fus_val)} ({len(fus_val) / total_images:.2%})")
print(f"Test dataset size: {len(fus_test)} ({len(fus_test) / total_images:.2%})")

# model weights paths
model_path_base = '/leonardo_work/IscrC_FoSAM-X/fetalus-fm/outputs/hffia_20250915_215625'
model_weights = 'best_model_iou.pth'
model_checkpoint = os.path.join(model_path_base, model_weights)

# Load experiment config
with open('/leonardo_work/IscrC_FoSAM-X/fetalus-fm/configs/experiments.json') as f:
    experiments = json.load(f)
config = experiments[0]
num_classes = 11
device = 'cpu'

model = DINOSegmentator(nc=num_classes, config=config['dino'], image_size=DEFAULT_IMAGE_SIZE, device=device)
model.to(device)

checkpoint = torch.load(model_checkpoint, weights_only=False, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict']),
model.eval()

# Numero di immagini da testare
num_images = 50
test_imgs_ids = [13, 50, 150, 200, 360, 460, 563, 601, 700, 804, 900, 1003, 1199, 1218, 1360, 1424, 1542, 1692, 1716, 1862, 1956, 2066, 2101, 2200, 2300, 2444, 2575, 2630, 2700, 2846, 2907, 3000, 3100, 3150]

# Cartella di output
output_dir = f"/leonardo_work/IscrC_FoSAM-X/fetalus-fm/testing_preds/{os.path.dirname(model_weights)}"
os.makedirs(output_dir, exist_ok=True)

now_str = datetime.now().strftime("%Y%m%d")

with torch.no_grad():
    # for i in range(num_images):
    for i in test_imgs_ids:
        #idx = random.randint(0, len(fus_test)-1)
        idx = i
        image, mask = fus_test[idx]
        pixel_values = image.unsqueeze(0)  # [1, 3, H, W]
        outputs = model(pixel_values)
        upsampled_logits = nn.functional.interpolate(
            outputs, size=mask.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        # Salva immagini
        base_name = f"{now_str}_idx{idx}"
        # cartella di output per questa immagine
        sub_dir = os.path.join(output_dir, base_name)
        os.makedirs(sub_dir, exist_ok=True)
        
        orig_path = os.path.join(sub_dir, f"image.png") # Immagine originale
        gt_path = os.path.join(sub_dir, f"mask_gt.png") # Maschera GT
        pred_path = os.path.join(sub_dir, f"mask_pred.png") # Maschera predetta
        
        print(f"Processing image {i+1}/{num_images} (dataset idx: {idx})")

        # Salva immagine originale
        img_np = image.permute(1, 2, 0).cpu().numpy()
        plt.imsave(orig_path, img_np)
        # Salva maschera GT
        mask_rgb = mask2d_to_rgb(mask, FUS_STRUCTS_COLORS)
        plt.imsave(gt_path, mask_rgb)
        # Salva maschera predetta
        pred_mask = upsampled_logits.argmax(1).squeeze(0).cpu()
        pred_mask_rgb = mask2d_to_rgb(pred_mask, FUS_STRUCTS_COLORS)
        plt.imsave(pred_path, pred_mask_rgb)

        # Overlay tra immagine originale e maschera GT
        overlay_gt_path = os.path.join(sub_dir, "overlay_gt.png")
        overlay_gt = (0.6 * img_np + 0.4 * mask_rgb / 255.0)
        plt.imsave(overlay_gt_path, np.clip(overlay_gt, 0, 1))

        # Overlay tra immagine originale e maschera predetta
        overlay_pred_path = os.path.join(sub_dir, "overlay_pred.png")
        overlay_pred = (0.6 * img_np + 0.4 * pred_mask_rgb / 255.0)
        plt.imsave(overlay_pred_path, np.clip(overlay_pred, 0, 1))

        # Overlay tra maschera GT e maschera predetta
        overlay_masks_path = os.path.join(sub_dir, "overlay_masks.png")
        overlay_masks = (0.5 * mask_rgb / 255.0 + 0.5 * pred_mask_rgb / 255.0)
        plt.imsave(overlay_masks_path, np.clip(overlay_masks, 0, 1))

        # Calcola mIoU per tutte le classi
        def compute_miou(gt, pred, num_classes):
            ious = []
            gt = gt.cpu().numpy() if hasattr(gt, 'cpu') else gt
            pred = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
            for cls in range(num_classes):
                gt_cls = (gt == cls)
                pred_cls = (pred == cls)
                intersection = np.logical_and(gt_cls, pred_cls).sum()
                union = np.logical_or(gt_cls, pred_cls).sum()
                if union == 0:
                    ious.append(np.nan)
                else:
                    ious.append(intersection / union)
            miou = np.nanmean(ious)
            return miou

        miou = compute_miou(mask, pred_mask, num_classes)

        # Salva la figura invece di mostrarla
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis('off')

        axes[1].imshow(mask_rgb)
        axes[1].set_title("Mask GT")
        axes[1].axis('off')

        axes[2].imshow(pred_mask_rgb)
        axes[2].set_title("Mask PRED")
        axes[2].axis('off')

        axes[3].imshow(overlay_masks)
        axes[3].set_title(f"Overlay Masks\nmIoU: {miou:.3f}")
        axes[3].axis('off')

        plt.tight_layout()
        fig_path = os.path.join(sub_dir, "image_gt_pred_overlay.png")
        plt.savefig(fig_path)
        plt.close(fig)
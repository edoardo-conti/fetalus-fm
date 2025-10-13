# Disable Albumentations update check for faster imports
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Import required libraries
import albumentations as A
import cv2
import numpy as np


def geometric_transforms(image_size, task='seg'):
    """
    Safe geometric transformations for fetal ultrasound data augmentation.
    Designed for medical imaging where anatomical structures must remain realistic.

    Args:
        image_size: Tuple (height, width) for image dimensions
        task: 'seg' or 'cls' - affects mask synchronization

    Returns:
        Albumentations Compose object
    """
    transforms = [
        # Orientation changes - safe for fetal planes
        A.HorizontalFlip(p=0.5),  # Mirror left/right - anatomical realistic

        # Position variations - simulate scanning variations
        A.Affine(scale=(0.9, 1.1),  # ±10% zoom
                translate_percent=(-0.1, 0.1),  # ±10% translation
                rotate=(-15, 15),  # ±15° rotation (safe for fetal)
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,  # Reflect for anatomical continuity
                p=0.4),

        # Light elastic deformation to simulate tissue variations
        A.ElasticTransform(alpha=1, sigma=5,
                          interpolation=cv2.INTER_LINEAR,
                          p=0.2),

        # Small random crops to focus on different anatomical regions
        A.RandomCrop(width=image_size[0], height=image_size[1],
                    p=0.3 if task == 'seg' else 0.1),  # Less aggressive for cls
    ]

    # For segmentation, synchronize with masks
    additional_targets = {'mask': 'image'} if task == 'seg' else {}
    return A.Compose(transforms, additional_targets=additional_targets)


def color_transforms(intensity='moderate'):
    """
    Color and intensity transformations that simulate realistic ultrasound variations.
    Moderately aggressive to simulate different scanning conditions while staying realistic.

    Args:
        intensity: 'light', 'moderate', or 'strong' - controls augmentation strength

    Returns:
        Albumentations Compose object
    """
    # Adjust parameters based on intensity
    if intensity == 'light':
        brightness_limit = 0.1
        contrast_limit = 0.1
        gamma_limit = 1.1  # Light gamma variation (gamma ~0.9-1.1)
        clahe_clip = 2.0
        noise_var = 0.001
        blur_limit = 3
    elif intensity == 'strong':
        brightness_limit = 0.3
        contrast_limit = 0.3
        gamma_limit = 1.2  # Stronger gamma variation (gamma ~0.8-1.2)
        clahe_clip = 4.0
        noise_var = 0.005
        blur_limit = 5
    else:  # moderate (default)
        brightness_limit = 0.2
        contrast_limit = 0.2
        gamma_limit = 1.15  # Moderate gamma variation (gamma ~0.87-1.15)
        clahe_clip = 3.0
        noise_var = 0.003
        blur_limit = 4

    transforms = [
        # Brightness and contrast variations - simulate different gain settings
        A.RandomBrightnessContrast(brightness_limit=brightness_limit,
                                 contrast_limit=contrast_limit,
                                 p=0.4),

        # Gamma correction - simulate different display curves
        A.RandomGamma(gamma_limit=gamma_limit, p=0.3),

        # CLAHE - automatic contrast enhancement (common in US processing)
        A.CLAHE(clip_limit=clahe_clip,
               tile_grid_size=(8, 8),
               p=0.3),

        # Gaussian noise - simulate sensor noise
        A.GaussNoise(p=0.2),  # Note: parameters depend on Albumentations version
    ]

    return A.Compose(transforms)


def get_cls_augmentations(image_size):
    """
    Complete augmentation pipeline for classification tasks.
    Combines geometric and color transformations for images only.

    Args:
        image_size: Tuple (height, width) for image dimensions

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Orientation & position
        A.HorizontalFlip(p=0.5),

        # Realistic scanning variations
        A.Affine(scale=(0.92, 1.08),  # ±8% zoom
                translate_percent=(-0.08, 0.08),  # ±8% translation
                rotate=(-12, 12),  # ±12° rotation (safe for fetal)
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,  # Reflect for anatomical continuity
                p=0.4),

        # Subtle tissue deformation
        A.ElasticTransform(alpha=1, sigma=5,
                          interpolation=cv2.INTER_LINEAR,
                          p=0.15),

        # Brightness/contrast (moderate)
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                 contrast_limit=0.15,
                                 p=0.35),

        # Gamma variation
        A.RandomGamma(gamma_limit=1.176, p=0.25),  # Gamma ~0.85-1.176

        # CLAHE enhancement
        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.25),

        # Subtle noise
        A.GaussNoise(p=0.15),
    ])


def get_val_augmentations(image_size):
    """
    Minimal augmentations for validation/test - only essential preprocessing.

    Args:
        image_size: Tuple (height, width) for image dimensions

    Returns:
        Albumentations Compose object with minimal transforms
    """
    return A.Compose([
        # Only center crop if needed for consistency
        A.CenterCrop(width=image_size[0], height=image_size[1]),
    ])

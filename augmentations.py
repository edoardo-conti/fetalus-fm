# Disable Albumentations update check for faster imports
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Import required libraries
import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torchvision.transforms import v2


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
        # A.CenterCrop(width=image_size[0], height=image_size[1]),
    ])


def make_imagenet_transform(resize_size: int = 224):
    """
    DINOv3 standard ImageNet evaluation transform for pretrained models.
    This should be applied after Albumentations transforms for proper normalization.

    Args:
        resize_size: Target size for resizing (should match model input size)

    Returns:
        torchvision Compose object with ImageNet normalization
    """
    return v2.Compose([
        v2.ToImage(),  # Convert to tensor if needed
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])


class CombinedTransform:
    """
    Combines Albumentations transforms with PyTorch ImageNet normalization.
    This ensures proper preprocessing for DINOv3 models pretrained on ImageNet.
    """
    def __init__(self, albumentations_transform, imagenet_transform=None, task='seg'):
        """
        Args:
            albumentations_transform: Albumentations Compose object
            imagenet_transform: torchvision transform for ImageNet normalization (optional)
            task: 'seg' or 'cls' to handle masks appropriately
        """
        self.albumentations_transform = albumentations_transform
        self.imagenet_transform = imagenet_transform
        self.task = task

    def __call__(self, **kwargs):
        # Apply Albumentations transforms
        result = self.albumentations_transform(**kwargs)

        # Apply ImageNet normalization if provided (for classification with pretrained models)
        if self.imagenet_transform is not None and 'image' in result:
            # Convert numpy array back to PIL for torchvision transforms
            if isinstance(result['image'], np.ndarray):
                # Convert from numpy (H, W, C) to PIL Image
                if result['image'].dtype != np.uint8:
                    # Scale to 0-255 if needed
                    img_min, img_max = result['image'].min(), result['image'].max()
                    if img_min < 0 or img_max > 1:
                        result['image'] = ((result['image'] - img_min) / (img_max - img_min) * 255).astype(np.uint8)

                pil_image = T.ToPILImage()(result['image'])
                # Apply ImageNet normalization
                normalized_image = self.imagenet_transform(pil_image)
                result['image'] = normalized_image

        return result

    def __repr__(self):
        return f"CombinedTransform(albumentations={self.albumentations_transform}, imagenet={self.imagenet_transform is not None})"

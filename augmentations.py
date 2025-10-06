# Disable Albumentations update check for faster imports
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Import required libraries
import albumentations as A
import cv2


def geometric_transforms(image_size):
    """
    Transforms/augmentations for training images and masks.
    
    :param img_size: Integer, for image resize.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(width=image_size[0], 
                     height=image_size[1], 
                     p=0.2),
        A.Affine(scale=1.0,
                 translate_percent=(-0.2, 0.2),
                 rotate=(-20, 20),
                 shear=1.0,
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT,
                 p=0.5),
        # A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.)
    ], additional_targets={'mask': 'image'})  # Abilita sync image/mask


def color_transforms():
    """
    Transforms/augmentations for training images and masks.
    
    :param img_size: Integer, for image resize.
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                   contrast_limit=0.2, 
                                   p=0.3),
        A.CLAHE(clip_limit=3.0, 
                tile_grid_size=(8, 8), 
                p=0.3),
        # A.Blur(blur_limit=[3, 5], p=0.3),
    ]) 
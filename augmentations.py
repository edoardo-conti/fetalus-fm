import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import albumentations as A

def geometric_transforms(image_size):
    """
    Transforms/augmentations for training images and masks.
    
    :param img_size: Integer, for image resize.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.2),
        A.RandomCrop(width=image_size[0], height=image_size[1], p=0.2),
        # A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.)
    ], additional_targets={'mask': 'image'})  # Abilita sync image/mask

def color_transforms():
    """
    Transforms/augmentations for training images and masks.
    
    :param img_size: Integer, for image resize.
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        A.Blur(blur_limit=[3, 5], p=0.3),
    ]) 

# def val_transforms(img_size):
#     """
#     Transforms/augmentations for validation images and masks.

#     :param img_size: Integer, for image resize.
#     """
#     val_image_transform = A.Compose([
#         A.Resize(img_size[1], img_size[0]),
#         # A.PadIfNeeded(
#         #     min_height=img_size[1]+4, 
#         #     min_width=img_size[0]+4,
#         #     position='center',
#         # ),
#         # A.Normalize(max_pixel_value=255.)
#         # A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.)
#     ], is_check_shapes=False)

#     return val_image_transform
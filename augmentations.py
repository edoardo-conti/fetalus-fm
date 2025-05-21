import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import albumentations as A

def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0]),
        # A.PadIfNeeded(
        #     min_height=img_size[1]+4, 
        #     min_width=img_size[0]+4,
        #     position='center'
        # ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25),
        A.Normalize(max_pixel_value=255.)
        # A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.)
    ], is_check_shapes=False)

    return train_image_transform

def val_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    val_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0]),
        # A.PadIfNeeded(
        #     min_height=img_size[1]+4, 
        #     min_width=img_size[0]+4,
        #     position='center',
        # ),
        A.Normalize(max_pixel_value=255.)
        # A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.)
    ], is_check_shapes=False)

    return val_image_transform
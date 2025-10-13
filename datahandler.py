from augmentations import geometric_transforms, color_transforms
from datasets.unified import UnifiedFetalDataset
from torch.utils.data import Subset


def get_fetalus_dataloaders(
    root,
    data_path,
    datasets,
    image_size,
    supervised=True,
    debug=False,
    task='seg',
    eval_augmentation=False,
):
    """
    Creates and returns train, validation, and test data loaders for fetal ultrasound data.
    Args:
        root (str): Root directory path for the data.
        data_path (str): Path to the data files relative to root.
        datasets (list): List of dataset names to include.
        image_size (tuple): Target size for image resizing (height, width).
        supervised (bool, optional): Whether the dataset is for supervised learning. Defaults to True.
        debug (bool, optional): If True, limits datasets to minimal samples for debugging. Defaults to False.
    Returns:
        tuple: Three dataset objects (train, val, test) wrapped in Subset if debug=True.
               Each dataset is an instance of UnifiedFetalDataset with appropriate transforms applied.
               Train set includes both geometric and color augmentations, while val/test sets have no augmentations.
    Note:
        - Geometric and color augmentations are only applied to the training set
        - In debug mode, datasets are limited to 2 training samples and 1 sample each for val/test
    """
    # Data augmentation trasformations
    geometric_augs = geometric_transforms(image_size, task=task)
    color_augs = color_transforms()

    # Create UnifiedFetalDataset instances for train, val, and test splits
    fus_train = UnifiedFetalDataset(
        root=root,
        data_path=data_path,
        datasets=datasets,
        split='train',
        supervised=supervised,
        target_size=image_size,
        augmentations=(geometric_augs, color_augs),
        task=task,
    )
    fus_val = UnifiedFetalDataset(
        root=root,
        data_path=data_path,
        datasets=datasets,
        split='val',
        supervised=supervised,
        target_size=image_size,
        task=task,
        eval_augmentation=eval_augmentation,
    )
    fus_test = UnifiedFetalDataset(
        root=root,
        data_path=data_path,
        datasets=datasets,
        split='test',
        supervised=supervised,
        target_size=image_size,
        task=task,
        eval_augmentation=eval_augmentation,
    )
    
    # If debug mode, limit the datasets to a small number of samples
    if debug:
        fus_train = Subset(fus_train, range(min(2, len(fus_train))))
        fus_val = Subset(fus_val, range(min(1, len(fus_val))))
        fus_test = Subset(fus_test, range(min(1, len(fus_test))))
    
    return fus_train, fus_val, fus_test

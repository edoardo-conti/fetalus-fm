import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from tqdm import tqdm
from metrics import IOUEval
from utils.utils import draw_translucent_seg_maps


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting by monitoring validation loss.
    Stops training if validation loss doesn't improve for a specified number of epochs.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping
        min_delta (float): Minimum change to qualify as an improvement
        verbose (bool): If True, prints messages when improving or stopping
    """
    def __init__(self, patience=7, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")


def get_lr_scheduler(optimizer, config):
    """
    Creates a learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary

    Returns:
        scheduler: Learning rate scheduler

    Supported schedulers:
    - 'step': StepLR - decays LR by gamma every step_size epochs
    - 'cosine': CosineAnnealingLR - cosine annealing schedule
    - 'plateau': ReduceLROnPlateau - reduces LR when metric stops improving
    """
    scheduler_type = config.get('lr_scheduler', 'step')  # default step

    if scheduler_type == 'step':
        # Decays LR by gamma every step_size epochs
        # Common: step_size=10, gamma=0.1 -> LR halved every 10 epochs
        step_size = config.get('lr_step_size', 10)
        gamma = config.get('lr_gamma', 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == 'cosine':
        # Cosine annealing: starts high, decreases following cosine curve
        # Better for fine-tuning, smoother decay
        T_max = config.get('lr_T_max', config['epochs'])  # Total cycles
        eta_min = config.get('lr_eta_min', 1e-6)  # Minimum LR
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == 'plateau':
        # Reduces LR by factor when metric stops improving
        # Good for unstable training, reacts to validation loss
        mode = config.get('lr_plateau_mode', 'min')  # 'min' for loss, 'max' for acc
        factor = config.get('lr_plateau_factor', 0.5)
        patience = config.get('lr_plateau_patience', 5)
        min_lr = config.get('lr_min', 1e-6)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                    patience=patience, min_lr=min_lr)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def train(
    model,
    train_dataloader,
    optimizer,
    criterion,
    task='seg',
    num_classes=None,
    logger=None,
    device='cpu',
    scheduler=None
):
    """
    Unified training function for both segmentation and classification tasks.

    Args:
        model: Neural network model
        train_dataloader: Training data loader
        optimizer: Optimizer (AdamW recommended)
        criterion: Loss function
        task: 'seg' for segmentation or 'cls' for classification
        num_classes: Number of classes (for segmentation metrics)
        logger: Logger for progress messages
        device: Device to run on ('cuda' or 'cpu')
        scheduler: Optional LR scheduler to step per epoch

    Returns:
        For 'seg': (train_loss, overall_acc, mIOU)
        For 'cls': (train_loss, train_acc)
    """
    if task not in ['seg', 'cls']:
        raise ValueError("Task must be 'seg' or 'cls'")

    logger.info(f'Training {task.upper()}')

    model.train()

    if task == 'seg':
        train_running_loss = 0.0
        iou_eval = IOUEval(num_classes)
    else:  # cls
        train_running_loss = 0.0
        train_running_acc = 0.0

    prog_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    counter = 0

    for data in prog_bar:
        counter += 1
        pixel_values = data[0].to(device)
        if task == 'seg':
            target = data[1].to(device)
        else:  # cls
            labels = data[1].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values)

        if task == 'seg':
            # Upsample logits to match target size for segmentation
            upsampled_logits = nn.functional.interpolate(
                outputs, size=target.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            loss = criterion(upsampled_logits, target.squeeze(1))
            loss.backward()
            optimizer.step()
            iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)

        else:  # cls
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Classification accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean().item()
            train_running_acc += acc

        train_running_loss += loss.item()

    # Epoch-level metrics
    train_loss = train_running_loss / counter

    if task == 'seg':
        overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
        return train_loss, overall_acc, mIOU
    else:  # cls
        train_acc = train_running_acc / counter
        return train_loss, train_acc


def validate(
    model,
    dataloader,
    criterion,
    task='seg',
    num_classes=None,
    logger=None,
    phase='val',
    device='cpu'
):
    """
    Unified validation function for both segmentation and classification tasks.

    Args:
        model: Neural network model
        dataloader: Validation/test data loader
        criterion: Loss function
        task: 'seg' for segmentation or 'cls' for classification
        num_classes: Number of classes (for segmentation metrics)
        logger: Logger for progress messages
        phase: 'val' or 'test'
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        For 'seg': (valid_loss, overall_acc, mIOU)
        For 'cls': (valid_loss, valid_acc)
    """
    if task not in ['seg', 'cls']:
        raise ValueError("Task must be 'seg' or 'cls'")

    logger.info(f'Validating {task.upper()} ({phase})')

    model.eval()

    if task == 'seg':
        valid_running_loss = 0.0
        iou_eval = IOUEval(num_classes)
    else:  # cls
        valid_running_loss = 0.0
        valid_running_acc = 0.0

    with torch.no_grad():
        prog_bar = tqdm(
            dataloader,
            total=len(dataloader),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        counter = 0

        for data in prog_bar:
            counter += 1
            pixel_values = data[0].to(device)
            if task == 'seg':
                target = data[1].to(device)
            else:  # cls
                labels = data[1].to(device)

            outputs = model(pixel_values)

            if task == 'seg':
                # Upsample logits for segmentation
                upsampled_logits = nn.functional.interpolate(
                    outputs, size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = criterion(upsampled_logits, target.squeeze(1))
                iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)

            else:  # cls
                loss = criterion(outputs, labels)
                # Classification accuracy
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean().item()
                valid_running_acc += acc

            valid_running_loss += loss.item()

    # Epoch-level metrics
    valid_loss = valid_running_loss / counter

    if task == 'seg':
        overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
        return valid_loss, overall_acc, mIOU
    else:  # cls
        valid_acc = valid_running_acc / counter
        return valid_loss, valid_acc

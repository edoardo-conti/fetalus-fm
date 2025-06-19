import torch
import torch.nn as nn


class DiceLoss(nn.Module):
        """
        A Dice Loss implementation for multi-class segmentation tasks.
        This loss function computes the Dice coefficient between predicted probabilities
        and target labels, which measures the overlap between two samples. The Dice
        coefficient is commonly used in segmentation tasks to evaluate the similarity
        between predicted and ground truth masks.
        Args:
            smooth (float, optional): Smoothing factor to avoid division by zero.
                Defaults to 1.0.
        Inputs:
            inputs (torch.Tensor): Raw unnormalized scores (logits) with shape (N, C, H, W),
                where N is batch size, C is number of classes, H and W are spatial dimensions.
            targets (torch.Tensor): Ground truth labels with shape (N, H, W) where each
                value is an integer in [0, C-1].
        Returns:
            torch.Tensor: Computed Dice loss (1 - mean Dice coefficient) as a scalar tensor.
        Note:
            - The input tensor is automatically softmax-normalized along the channel dimension.
            - Targets are automatically converted to one-hot encoding.
            - The Dice coefficient is computed per-class and then averaged.
        """
        def __init__(self, smooth=1.0):
            super(DiceLoss, self).__init__()
            self.smooth = smooth

        def forward(self, inputs, targets):
            # inputs: (N, C, H, W), targets: (N, H, W)
            inputs = torch.softmax(inputs, dim=1)
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
            intersection = torch.sum(inputs * targets_one_hot, dims)
            cardinality = torch.sum(inputs + targets_one_hot, dims)
            dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            return 1 - dice.mean()

class CEDiceLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss for segmentation tasks.
    This loss function combines Cross-Entropy loss and Dice loss, weighted by user-defined parameters.
    It's commonly used in segmentation tasks to leverage both pixel-wise classification (CE) and
    region-based similarity (Dice) metrics.
    Args:
        weight_dice (float, optional): Weight for Dice loss component (0-1). Default: 0.5.
        smooth (float, optional): Smoothing factor for Dice loss. Default: 1.0.
    Inputs:
        inputs (torch.Tensor): Raw, unnormalized scores for each class (N x C x H x W).
        targets (torch.Tensor): Ground truth labels (N x H x W) with class indices.
    Returns:
        torch.Tensor: Combined loss value (weighted sum of CE and Dice losses).
    Note:
        The weight for Cross-Entropy loss is automatically set as (1 - weight_dice) to ensure
        the weights sum to 1.
    """
    def __init__(self, weight_dice=0.5, smooth=1.0):
        super(CEDiceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.weight_dice = weight_dice
        self.weight_ce = 1. - self.weight_dice
        
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.weight_ce * ce + self.weight_dice * dice
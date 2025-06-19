import torch
import torch.nn as nn
import ssl
from functools import partial
from collections import OrderedDict
from configs.dinov2_backbone_cfg import model as model_dict


class LinearClassifierToken(torch.nn.Module):
    """
    A linear classifier module that operates on tokenized input features.
    This module applies a 1x1 convolution to classify tokenized input features into `nc` classes.
    The input is expected to be in a flattened form which is reshaped to (batch_size, in_channels, H, W)
    before applying the convolution.
    Args:
        in_channels (int): Number of input channels/features per token.
        nc (int, optional): Number of output classes. Defaults to 1.
        tokenW (int, optional): Width of each token. Defaults to 46.
        tokenH (int, optional): Height of each token. Defaults to 46.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, in_channels) 
                          or similar that can be reshaped to (batch_size, in_channels, H, W).
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, nc, H, W) after applying 1x1 convolution.
    """
    def __init__(self, in_channels, nc=1, tokenW=46, tokenH=46):
        super(LinearClassifierToken, self).__init__()
        self.in_channels = in_channels
        self.W = tokenW
        self.H = tokenH
        self.nc = nc
        self.conv = torch.nn.Conv2d(in_channels, nc, (1, 1))
    
    def forward(self,x):
        outputs =  self.conv(
            x.reshape(-1, self.in_channels, self.H, self.W)
        )
        return outputs


class Dinov2Segmentation(nn.Module):
    """
    A PyTorch module for segmentation using DINOv2 as a backbone.
    This module combines the DINOv2 vision transformer backbone with a linear classifier
    head for segmentation tasks. The backbone can be fine-tuned or kept frozen during training.
        nc (int): Number of classes for the segmentation task.
        config (dict): Configuration dictionary containing:
            - backbone_size (str): Size of DINOv2 backbone ('small', 'base', 'large', or 'giant')
            - intermediate_layers (list): List of layer indices to extract features from
        image_size (tuple): Input image size (height, width). Default: (644, 644)
        fine_tune (bool): Whether to fine-tune the backbone. Default: False
        device (str): Device to run the model on ('cpu', 'cuda', etc.). Default: 'cpu'
    Methods:
        load_backbone(backbone_size, int_layers, device):
            Loads and configures the DINOv2 backbone model.
                backbone_size (str): Size of backbone model
                int_layers (list): Intermediate layers to extract features from
                device (str): Device to load model on
                torch.nn.Module: Configured backbone model
        forward(x):
            Forward pass through the model.
                x (torch.Tensor): Input tensor
                torch.Tensor: Segmentation output
    The model architecture consists of:
    1. DINOv2 backbone (with configurable size and feature extraction layers)
    2. Linear classifier head for segmentation
    """
    def __init__(self, nc, config, image_size=(644,644), fine_tune=False, device="cpu"):
        super(Dinov2Segmentation, self).__init__()

        # Load the DINOv2 backbone model based on the configuration
        self.backbone_model = self.load_backbone(backbone_size=config['backbone_size'], 
                                                 int_layers=config['intermediate_layers'],
                                                 device=device)
        
        # Set the backbone model to training mode if fine-tuning is enabled
        for param in self.backbone_model.parameters():
            param.requires_grad = fine_tune
        
        # Extract the number of channels from the backbone model and calculate token dimensions
        self.channels = model_dict['decode_head']['channels']
        self.tokenWH = image_size[0] // 14 

        # Define the decode head for segmentation
        self.decode_head = LinearClassifierToken(in_channels=self.channels, nc=nc, tokenW=self.tokenWH, tokenH=self.tokenWH)

        # Create the model as a sequential container
        self.model = nn.Sequential(OrderedDict([
            ('backbone', self.backbone_model),
            ('decode_head', self.decode_head)
        ]))

    def load_backbone(self, backbone_size="small", int_layers=[8, 9, 10, 11], device="cpu"):        
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"

        # Bypass SSL verification (TODO: Remove this)
        ssl._create_default_https_context = ssl._create_unverified_context  
        
        # Load the DINOv2 backbone model from Facebook's research repository
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        backbone_model = backbone_model.to(device)
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n=int_layers,
            reshape=True,
        )

        return backbone_model

    def forward(self, x):
        features = self.model.backbone(x)

        # `features` is a tuple.
        concatenated_features = torch.cat(features, 1)

        # Pass the concatenated features through the decode head
        classifier_out = self.decode_head(concatenated_features)

        return classifier_out
import torch
import torch.nn as nn
import ssl
from functools import partial
from collections import OrderedDict
from configs.dinov2_backbone_cfg import model as model_dict

ssl._create_default_https_context = ssl._create_unverified_context


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool, use_cls=False):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    if use_cls:
        # Use CLS token from intermediate layers (assuming first token is CLS)
        output = torch.cat([layer[:, 0, :] for layer in intermediate_output], dim=-1)  # CLS tokens from layers
    else:
        output = torch.cat([torch.mean(layer, dim=1) for layer in intermediate_output], dim=-1)  # global mean of layers
    if use_avgpool:
        last_layer = intermediate_output[-1]
        if use_cls:
            # Already using CLS, no need for avgpool
            pass
        else:
            if last_layer.shape[1] > 1:  # if more than 1 token
                avg_patch = torch.mean(last_layer, dim=1)  # (batch, dim)
                output = torch.cat((output, avg_patch), dim=-1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000, use_cls=False):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.use_cls = use_cls
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool, self.use_cls)
        return self.linear(output)


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


class DINOSegmentator(nn.Module):
    """
    A PyTorch module for segmentation using DINO as a backbone.
    This module combines the DINO vision transformer backbone with a linear classifier
    head for segmentation tasks. The backbone can be fine-tuned or kept frozen during training.
        nc (int): Number of classes for the segmentation task.
        config (dict): Configuration dictionary containing:
            - backbone_size (str): Size of DINO backbone ('small', 'base', 'large', or 'giant')
            - intermediate_layers (list): List of layer indices to extract features from
        image_size (tuple): Input image size (height, width). Default: (644, 644)
        fine_tune (bool): Whether to fine-tune the backbone. Default: False
        device (str): Device to run the model on ('cpu', 'cuda', etc.). Default: 'cpu'
    Methods:
        load_backbone(backbone_size, int_layers, device):
            Loads and configures the DINO backbone model.
                backbone_size (str): Size of backbone model
                int_layers (list): Intermediate layers to extract features from
                device (str): Device to load model on
                torch.nn.Module: Configured backbone model
        forward(x):
            Forward pass through the model.
                x (torch.Tensor): Input tensor
                torch.Tensor: Segmentation output
    The model architecture consists of:
    1. DINO backbone (with configurable size and feature extraction layers)
    2. Linear classifier head for segmentation
    """
    def __init__(self, nc, config, image_size=(644,644), fine_tune=False, device="cpu"):
        super(DINOSegmentator, self).__init__()

        # Load the DINOv2 backbone model based on the configuration
        self.backbone_model = self.load_backbone(dinov=config['version'],
                                                 backbone_size=config['backbone_size'], 
                                                 pretrained_weights=config['pretrained_weights'],
                                                 int_layers=config['intermediate_layers'],
                                                 device=device)
        
        # Set the backbone model to training mode if fine-tuning is enabled
        for param in self.backbone_model.parameters():
            param.requires_grad = fine_tune
        
        # Extract the number of channels from the backbone model and calculate token dimensions
        patch_size = 14 if config['version'] == 'v2' else 16
        self.channels = model_dict['decode_head']['channels']
        self.tokenWH = image_size[0] // patch_size 

        # Define the decode head for segmentation
        self.decode_head = LinearClassifierToken(in_channels=self.channels, nc=nc, tokenW=self.tokenWH, tokenH=self.tokenWH)

        # Create the model as a sequential container
        self.model = nn.Sequential(OrderedDict([
            ('backbone', self.backbone_model),
            ('decode_head', self.decode_head)
        ]))

    def load_backbone(self, dinov="v2", backbone_size="small", pretrained_weights=None, int_layers=[8, 9, 10, 11], device="cpu"):    
        if dinov != "v2":   
            backbone_archs = {
                "small": "dinov3_vits16",
                "base": "dinov3_vitb16",
                "large": "dinov3_vitl16",
                "giant": "dinov3_vit7b16",
            }
            backbone_arch = backbone_archs[backbone_size]
            
            # Load the DINOv2 backbone model
            fwork_path = '/leonardo_work/IscrC_FoSAM-X/fetalus-fm/dinov3/'
            dinov3_weights = f"{fwork_path}/weights/{backbone_arch}_pretrain_lvd1689m-73cec8be.pth"
            backbone_model = torch.hub.load(fwork_path, backbone_arch, source='local', weights=dinov3_weights)

        else:
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
            
            # Load the DINOv2 backbone model
            backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
            
            if pretrained_weights:
                checkpoint = torch.load(pretrained_weights, weights_only=False, map_location=device)

                # Extract student backbone weights
                student_weights = {}
                for key, value in checkpoint["model"].items():
                    if key.startswith("student.backbone."):
                        new_key = key[17:]  # Remove "student.backbone." prefix
                        student_weights[new_key] = value

                missing_keys, unexpected_keys = backbone_model.load_state_dict(student_weights, strict=False)
                print(f"[DEBUG] Missing keys: {missing_keys}")
                print(f"[DEBUG] Unexpected keys: {unexpected_keys}")
                
                print(f"Loaded DINOv2 {backbone_arch} backbone from local weights: {pretrained_weights}")
                
        backbone_model = backbone_model.to(device)
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n=int_layers,
            reshape=False,
        )

        return backbone_model

    def forward(self, x):
        features = self.model.backbone(x)

        # `features` is a tuple.
        concatenated_features = torch.cat(features, 1)

        # Pass the concatenated features through the decode head
        classifier_out = self.decode_head(concatenated_features)

        return classifier_out


class DINOClassifier(nn.Module):
    """
    A PyTorch module for classification using DINO as a backbone.
    This module combines the DINO vision transformer backbone with a linear classifier
    head for classification tasks. The backbone can be fine-tuned or kept frozen during training.
        nc (int): Number of classes for the classification task.
        config (dict): Configuration dictionary containing:
            - backbone_size (str): Size of DINO backbone ('small', 'base', 'large', or 'giant')
            - intermediate_layers (list): List of layer indices to extract features from
        image_size (tuple): Input image size (height, width). Default: (644, 644)
        fine_tune (bool): Whether to fine-tune the backbone. Default: False
        device (str): Device to run the model on ('cpu', 'cuda', etc.). Default: 'cpu'
    Methods:
        load_backbone(backbone_size, int_layers, device):
            Loads and configures the DINO backbone model.
                backbone_size (str): Size of backbone model
                int_layers (list): Intermediate layers to extract features from
                device (str): Device to load model on
                torch.nn.Module: Configured backbone model
        forward(x):
            Forward pass through the model.
                x (torch.Tensor): Input tensor
                torch.Tensor: Classification output
    The model architecture consists of:
    1. DINO backbone (with configurable size and feature extraction layers)
    2. Linear classifier head for classification
    """
    def __init__(self, nc, config, image_size=(644,644), fine_tune=False, device="cpu"):
        super(DINOClassifier, self).__init__()

        # Load the DINO backbone model based on the configuration
        self.backbone_model = self.load_backbone(dinov=config['version'],
                                                 backbone_size=config['backbone_size'], 
                                                 pretrained_weights=config['pretrained_weights'],
                                                 int_layers=config['intermediate_layers'],
                                                 device=device)
        
        # Set the backbone model to training mode if fine-tuning is enabled
        for param in self.backbone_model.parameters():
            param.requires_grad = fine_tune
        
        # Extract the number of channels from the backbone model
        self.channels = model_dict['decode_head']['channels']
        self.use_n_blocks = 1
        self.use_cls = config.get('use_cls_token', True)    # Usa CLS token per migliore rappresentazione globale
        self.use_avgpool = not self.use_cls                 # Opposto di self.use_cls

        # Create dummy input to calculate out_dim
        dummy_input = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
        sample_output = self.backbone_model(dummy_input)
        out_dim = create_linear_input(sample_output, self.use_n_blocks, self.use_avgpool, self.use_cls).shape[1]

        # Define the decode head for classification
        self.decode_head = LinearClassifier(out_dim, self.use_n_blocks, self.use_avgpool, num_classes=nc, use_cls=self.use_cls)
        
        # Create the model as a sequential container
        self.model = nn.Sequential(OrderedDict([
            ('backbone', self.backbone_model),
            ('decode_head', self.decode_head)
        ]))

    def load_backbone(self, dinov="v2", backbone_size="small", pretrained_weights=None, int_layers=[8, 9, 10, 11], device="cpu"):    
        if dinov != "v2":   
            backbone_archs = {
                "small": "dinov3_vits16",
                "base": "dinov3_vitb16",
                "large": "dinov3_vitl16",
                "giant": "dinov3_vit7b16",
            }
            backbone_arch = backbone_archs[backbone_size]

            # Load the DINOv3 backbone model
            dinov3_repo_path = '/Users/edoardoconti/PhD/projects/fetalUS_FM/dinov3'
            # pretrained_weights = "/Users/edoardoconti/PhD/projects/fetalUS_FM/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            backbone_model = torch.hub.load(dinov3_repo_path, backbone_arch, source='local', weights=pretrained_weights)

        else:
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

            # Load the DINOv2 backbone model
            backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)

            if pretrained_weights:
                checkpoint = torch.load(pretrained_weights, weights_only=False, map_location=device)

                # Extract student backbone weights
                student_weights = {}
                for key, value in checkpoint["model"].items():
                    if key.startswith("student.backbone."):
                        new_key = key[17:]  # Remove "student.backbone." prefix
                        student_weights[new_key] = value

                missing_keys, unexpected_keys = backbone_model.load_state_dict(student_weights, strict=False)
                print(f"[DEBUG] Missing keys: {missing_keys}")
                print(f"[DEBUG] Unexpected keys: {unexpected_keys}")

                print(f"Loaded DINOv2 {backbone_arch} backbone from local weights: {pretrained_weights}")

        backbone_model = backbone_model.to(device)
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n=int_layers,
            reshape=False,
        )

        return backbone_model

    def forward(self, x):
        features = self.model.backbone(x)

        # Pass the features through the decode head
        classifier_out = self.decode_head(features)

        return classifier_out

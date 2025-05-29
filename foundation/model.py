import torch
import torch.nn as nn
import ssl

from functools import partial
from collections import OrderedDict
from torchinfo import summary
from utils import IMAGE_SIZE

from configs.dinov2_backbone_cfg import model as model_dict

class LinearClassifierToken(torch.nn.Module):
    def __init__(self, in_channels, nc=1, tokenW=32, tokenH=32):
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
    def __init__(self, nc, backbone="small", fine_tune=False, device="cpu"):
        super(Dinov2Segmentation, self).__init__()
        
        self.backbone_model = self._load_backbone(backbone_size=backbone, device=device)
        
        if fine_tune:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False

        self.decode_head = LinearClassifierToken(in_channels=1536, nc=nc, tokenW=46, tokenH=46)

        self.model = nn.Sequential(OrderedDict([
            ('backbone', self.backbone_model),
            ('decode_head', self.decode_head)
        ]))

    def _load_backbone(self, backbone_size="small", device="cpu"):
        """
        Load the DINOv2 backbone model.
        Args:
            backbone_size (str): Size of the backbone model. Options are "small", "base", "large", or "giant".
            device (str): Device to load the model on. Options are "cpu", "cuda", or "mps".
        Returns:
            backbone_model (torch.nn.Module): The DINOv2 backbone model.
        """
        
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"

        ssl._create_default_https_context = ssl._create_unverified_context  # Bypass SSL verification (TODO: Remove this)
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        
        backbone_model = backbone_model.to(device)
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n=model_dict['backbone']['out_indices'],
            reshape=True,
        )

        return backbone_model

    def forward(self, x):
        features = self.model.backbone(x)

        # `features` is a tuple.
        concatenated_features = torch.cat(features, 1)

        classifier_out = self.decode_head(concatenated_features)

        return classifier_out
    
if __name__ == '__main__':
    model = Dinov2Segmentation()
    
    summary(
        model, 
        (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )

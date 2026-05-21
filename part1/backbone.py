import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    """
    Module backbone use ResNet to extract feature maps
    """
    def __init__(self, name = "resnet50", train_backbone: bool = True, d_model: int = 256):
        super().__init__()
        # Load ResNet
        backbone = getattr(models, name)(pretrained=True)
        
        # Remove avgpool and fc
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        
        #Freeze backbone
        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
                
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        
        # Project to d_model
        self.conv_proj = nn.Conv2d(num_channels, d_model, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: X first tensor shape [batch_size, 3, H, W]
        Output: feature Tensor shape [batch_size, d_model, H/32, W/32]
        """
        feature = self.body(x)
        return self.conv_proj(feature)

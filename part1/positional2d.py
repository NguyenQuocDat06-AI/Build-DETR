import math
import torch
import torch.nn as nn


class Positional2DEncoding(nn.Module):
    """
    2D Sinusoidal Positional Encoding
    """
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        INPUT: X shape [Batch, C, H, W] with C = d_model = 256
        OUTPUT:  Tensor with shape [Batch, C, H, W]
        """
        b,c,h,w = x.shape

        y_embed = torch.ones(b, h, w, device=x.device).cumsum(1, dtype=torch.float32)
        x_embed = torch.ones(b, h, w, device=x.device).cumsum(2, dtype=torch.float32)
        
        # Standard position encod [0,2*pi]

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

            # Create vector omega
            # dim_t = [0,1,2,...,num_pos_feats-1]
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** ( 2 * (dim_t // 2) / self.num_pos_feats)

            # y_embed: [H,W] -> [H,W,1] / [128] -> [H,W,128]
            pos_y = y_embed.unsqueeze(-1) / dim_t
            # x_embed: [H,W] -> [H,1,W] / [128] -> [H,1,128]
            pos_x = x_embed.unsqueeze(-1) / dim_t

            # Sine encoding
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            
            # Concatenate
            pos = torch.cat((pos_y, pos_x), dim=3)

            # [B,H,W,C] -> [B,C,H,W]
            pos = pos.permute(0, 3, 1, 2)
            
        return pos
        
        
        

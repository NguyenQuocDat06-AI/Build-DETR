import torch
from backbone import Backbone
from positional2d import Positional2DEncoding

if __name__ == '__main__':
    # Ex: Batch_size = 2, 3 channel, H = 600, W = 800
    dummy_images = torch.randn(2, 3, 600, 800)
    print(f"Shape of the dummy images: {dummy_images.shape}")

    backbone = Backbone()
    feature_maps = backbone(dummy_images)
    print(f"Shape of feature maps: {feature_maps.shape}")

    pos_encoding = Positional2DEncoding()
    pos = pos_encoding(feature_maps)
    print(f"Shape of positional encodings: {pos.shape}")

    # Add positional encodings to feature maps
    output = feature_maps + pos
    print(f"Shape of output: {output.shape}")
import torch.nn as nn
import torch

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class SpatialPath(nn.Sequential):
    def __init__(self, in_channels=3):
        super().__init__(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )


if __name__ == '__main__':
    spatial_path = SpatialPath()
    dummy = torch.rand(8, 3, 224, 224) # 1/8 downsample
    assert spatial_path(dummy).shape == torch.Size([8, 256, 28, 28]), "Invalid output shape"
    print("Test Successful")


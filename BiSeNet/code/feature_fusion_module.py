import torch
import torch.nn as nn


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # in_channels = sum of x1.size(1) + x2.size(2)
        self.in_channels = in_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1), # same padding
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_1x1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        assert x.size(1) == self.in_channels, "in_channels and concatenated feature channel must be the same"

        feature = self.conv_block(x)

        attn = self.avg_pool(feature)
        attn = self.conv_1x1(attn)
        attn = self.relu(attn)
        attn = self.conv_1x1(attn)
        attn = self.relu(attn)

        mul = torch.mul(feature, attn)
        add = torch.add(feature, mul)

        return add


if __name__ == '__main__':
    FFM = FeatureFusionModule(64, 10)
    dummy1 = torch.randn(8, 32, 224, 224)
    dummy2 = torch.randn(8, 32, 224, 224)

    out = FFM(dummy1, dummy2)
    assert out.shape == torch.Size([8, 10, 224, 224]), "Invalid output shape"
    print("Test Successful")

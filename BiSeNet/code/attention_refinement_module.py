import torch
import torch.nn as nn

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels == out_channels, "in_channels and out_channels must be the same"
        self.in_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert x.size(1) == self.in_channels, "in_channels and channel for x must be the same"
        x_copy = x.detach().clone()

        x = self.avg_pool(x)
        x = self.conv_1x1(x)
        x = self.batch_norm(x)
        x = self.sigmoid(x)

        out = torch.mul(x_copy, x)
        return out


if __name__ == '__main__':
    ARM = AttentionRefinementModule(64, 64)
    dummy = torch.randn(8, 64, 224, 224)
    out = ARM(dummy)
    assert out.shape == dummy.shape, "Invalid output shape"
    print("Test Successful")

import torch
import torch.nn as nn
from torchvision import models

class ContextPath(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet18 = models.resnet18(pretrained=pretrained)
        self.conv_block = nn.Sequential( # 1/4
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool
        )

        self.layer1 = resnet18.layer1 # 1/4
        self.layer2 = resnet18.layer2 # 1/8
        self.layer3 = resnet18.layer3 # 1/16
        self.layer4 = resnet18.layer4 # 1/32

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv_block(x)
        down_4x = self.layer1(x)
        down_8x = self.layer2(down_4x)
        down_16x = self.layer3(down_8x)
        down_32x = self.layer4(down_16x)
        tail = self.avg_pool(down_32x)

        return down_16x, down_32x, tail


if __name__ == '__main__':
    dummy = torch.randn(8, 3, 224, 224)
    context_path = ContextPath()
    down_16x, down_32x, tail = context_path(dummy)
    assert down_16x.size(2) == dummy.size(2) / 16, "Invalid 16x downsample"
    assert down_32x.size(2) == dummy.size(2) / 32, "Invalid 32x downsample"
    assert tail.size(2) == 1, "Invalid global average pooling"
    print("Test Successful")

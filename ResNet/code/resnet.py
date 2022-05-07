import torch
import torch.nn as nn
from typing import List

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, block_channels, identity_downsample = None, stride = 1):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.identity_downsample = identity_downsample
        self.stride = stride

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, block_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_channels),
            nn.ReLU()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(block_channels, block_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(block_channels),
            nn.ReLU()
        )

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(block_channels, block_channels * self.expansion, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_channels * self.expansion)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.detach().clone()
        x = self.conv1x1_1(x)
        x = self.conv3x3(x)
        x = self.conv1x1_2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x) # Apply ReLU after shortcut
        return x


class ResNet(nn.Module):
    def __init__(self, block: Bottleneck, layers: List[int], in_channels = 3, num_classes = 1000):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # [1] conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2)
        )
        self.in_channels = 64

        # [2] conv2_x, 3_x, 4_x, 5_x
        self.conv2_x = self._make_layer(block, layers[0], block_channels = 64, stride = 1)
        self.conv3_x = self._make_layer(block, layers[1], block_channels = 128, stride = 2)
        self.conv4_x = self._make_layer(block, layers[2], block_channels = 256, stride = 2)
        self.conv5_x = self._make_layer(block, layers[3], block_channels = 512, stride = 2)

        # [3] classifiers
        self.classifiers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def _make_layer(self, block, num_blocks, block_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != 4 * block_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, 4 * block_channels, kernel_size=1, stride=stride, bias=False), # linear projection
                nn.BatchNorm2d(4 * block_channels)
            )

        # This connects between the final ouput of previous bottleneck to the input of the current bottleneck
        # The reason we do this is that the input of bottleneck might come from TWO paths
            # 1. Previous bottleneck (if current is conv3_x, then from conv2_x). In this case,
            # input size of conv1x1 of conv3_x(128)  is HALF of final output of conv2_x(256)
            # 2. bottleneck uses expansion of 4. Denoting current bottleneck's channels for 1x1, 3x3, 1x1 as p,p,4p,
            # the input channel would be 4p -> 1p.
            # Therefore, we must separate the very first(from ANOTHER bottleneck) bottleneck and
            # simply from the CURRENT repetitive bottleneck
        layers.append(
            Bottleneck(self.in_channels, block_channels, identity_downsample, stride=stride)
        )

        #4p -> p. Like a recursion, we're going back to the beginning of current bottleneck
        self.in_channels = block_channels * 4

        for _ in range(1, num_blocks):
            layers.append(
                Bottleneck(self.in_channels, block_channels, stride=1) # no identity downsample
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.classifiers(x)
        return x


resnet = {
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}

d = torch.randn(1,3,224,224)

model = ResNet(Bottleneck, resnet['resnet50'])

def print_layer_shape_hook(module, input, output):
    print("-"*80)
    print(module)
    print(module.__class__.__name__)
    print(f"[input] {input[0].shape}")
    print(f"[ouput] {output.shape}")

for name, submodule in model.named_modules():
    if isinstance(submodule, (nn.Conv2d, nn.MaxPool2d, nn.Linear, nn.AdaptiveAvgPool2d)):
        submodule.register_forward_hook(print_layer_shape_hook)

_ = model(d)

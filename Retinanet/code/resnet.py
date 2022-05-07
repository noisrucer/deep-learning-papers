import torch
import torch.nn as nn
from typing import List

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, block_channels, identity_downsample=None, stride=1):
        """
        Parameters:
            in_channels (int): channels for the previous "block"
            block_channels (int): channels for the start of the current "block"
            identity_downsample (nn.Module): identity downsample module for matching the dimension
            stride (int): stride

        """
        super().__init__()
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
            nn.Conv2d(block_channels, block_channels * self.expansion, 1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_channels * self.expansion)
        )

        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x.clone().detach()
        x = self.conv1x1_1(x)
        x = self.conv3x3(x)
        x = self.conv1x1_2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layers: List[int], block=Bottleneck, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # [1] Conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2)
        )
        self.in_channels = 64

        # [2] conv2_x, 3_x, 4_x, 5_x (same notation from the paper)
        self.conv2_x = self._make_layer(block, layers[0], block_channels=64, stride=1)
        self.conv3_x = self._make_layer(block, layers[1], block_channels=128, stride=2)
        self.conv4_x = self._make_layer(block, layers[2], block_channels=256, stride=2)
        self.conv5_x = self._make_layer(block, layers[3], block_channels=512, stride=2)


    def _make_layer(self, block, num_blocks, block_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != block_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, block_channels * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(4 * block_channels)
            )

        layers.append(
            block(self.in_channels, block_channels, identity_downsample, stride=stride)
        )

        self.in_channels = block_channels * 4

        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_channels, block_channels, stride=1)
            )

        return nn.Sequential(*layers)


    def forward(self, x):
        '''
        Instead of forwarding to the classification layer, return the features
        of conv3,4,5 for Feature Pyramid Network(FPN)
        '''
        print(x.shape)
        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)

        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        return x3, x4, x5

import torch
import torch.nn as nn
from darknet19 import DarkNet19

class Yolov2(nn.Module):
    def __init__(self, in_channel=3, n_classes=5):
        super().__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes

        self.darknet19 = DarkNet19(in_channel=in_channel)
        self.darknet19.load_weight()

        self.conv1 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv2 = nn.Conv2d(1024, 1024, 3, 1, 1) # skip
        self.conv3 = nn.Conv2d(3072, 1024, 3, 1, 1)
        self.final_conv = nn.Conv2d(1024, 5*(5+n_classes), 1, stride=1, padding=0)


    def forward(self, x):
        x, skip = self.darknet19(x) # x: (B, 1024, 13, 13), skip: (B, 2048, 13, 13)
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.cat([x, skip], dim=1) # (B, 3072, 13, 13)
        x = self.conv3(x)
        out = self.final_conv(x)

        return out


import torch
import torch.nn as nn

# 1) init function
    # 1-1) Conv layers
    # 1-2) FC layers
    # 1-3) Weight/Bias Initialization
# 2) Forward function

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 96, 11, stride=4), # Conv1 / Valid pad
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-5, beta=0.75, k=2), # LRN
            nn.MaxPool2d(3, stride=2), # overlapping pool
            nn.Conv2d(96, 256, 5, stride=1, padding=2), # Conv2 / Same pad
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-5, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, stride=1, padding=1), # Conv3 / Same pad
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, stride=1, padding=1), # Conv4 / Same pad
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, stride=1, padding=1), # Conv5 / Same pad
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Flatten()
        )

        self.classifiers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

        # Initialize Weights/Biases
        self.apply(AlexNet.init_params)

    @staticmethod
    def init_params(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 1)

            # Conv1 layer has in_channels = 3
            # Conv2 layer has in_channels = 256
            # This is unique. So let's take advantage of that.
            # isinstance() required since nn.Linear has name "in_features".
            if isinstance(m, nn.Conv2d) and (m.in_channels == 3 or m.in_channels == 256):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.classifiers(out)
        return out # Shape: (batch_size, 1000)

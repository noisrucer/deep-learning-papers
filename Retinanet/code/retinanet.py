import torch
import torch.nn as nn
import torch.nn.functional as F

from fpn import FPN
from resnet import ResNet
from anchors import Anchors
#  from focal_loss import FocalLoss

class ClassificationModel(nn.Module):
    def __init__(self, n_in_features=256, n_anchors=9, n_classes=10,
                 prior=0.01, feature_size=256):
        super().__init__()
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_in_features, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # KA channels
        self.out_layers = nn.Sequential(
            nn.Conv2d(feature_size, n_classes * n_anchors, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.out_layers(out) # (B, KA, W, H)

        out1 = out.permute(0, 2, 3, 1) # (B, W, H, KA)
        batch_size, w, h, channels = out1.shape

        # (B, W, H, A, K)
        out2 = out1.view(batch_size, w, h, self.n_anchors, self.n_classes)

        # (B, W*H*A, K)
        return out2.contiguous().view(x.shape[0], -1, self.n_classes)


class RegressionModel(nn.Module):
    def __init__(self, n_in_features=256, n_anchors=9, feature_size=256):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_in_features, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 4A channels
        self.out_layers = nn.Conv2d(feature_size, n_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.out_layers(out) # (B, 4K, W, H)

        out = out.permute(0, 2, 3, 1) # (B, W, H, 4K)

        return out.contiguous().view(out.shape[0], -1, 4) # (B, W*H*K, 4)


class Retinanet(nn.Module):
    def __init__(self, resnet_in_channels = [3, 4, 6, 3], n_classes=10):
        super().__init__()
        self.resnet = ResNet(layers=resnet_in_channels)

        self.regression_model = RegressionModel()
        self.classification_model = ClassificationModel(n_classes=n_classes)
        self.anchors = Anchors()
        #  self.focal_loss = FocalLoss()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        ######### ResNet ###########
        resnet_out = self.resnet(img_batch)

        ########## FPN ###########
        fpn_in_channels = [d.shape[1] for d in resnet_out]
        fpn = FPN(in_channels=fpn_in_channels, out_channel=256)
        features = fpn(resnet_out)

        ######## Box Regression ##########
        regression = torch.cat(
            [self.regression_model(feature) for feature in features],
            dim=1
        )

        ########### Classification ###########
        classification = torch.cat(
            [self.classification_model(feature) for feature in features],
            dim=1
        )

        ########## Anchors ###########
        anchors = self.anchors(img_batch)
        print("regression:",regression.shape)
        print("classification:", classification.shape)
        print("anchors:",anchors.shape)

        ########## Focal Loss ###########
        if self.training:
            return self.focal_loss(classification, regression, anchors, annotations)
        else:
            pass

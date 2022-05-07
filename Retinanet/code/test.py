import torch
from fpn import FPN
from resnet import ResNet
from retinanet import Retinanet
from anchors import Anchors
dummy = torch.randn(8, 3, 224, 224)
retinanet = Retinanet()
retinanet((dummy, 3))

#  anchors = Anchors()
#  x = anchors(dummy)
#  print(x.shape)



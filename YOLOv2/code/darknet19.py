import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import os.path as osp

#  from holocron import GlobalAvgPool2d

# (in_channel, List[(kernel_size, out_channel)], max_pooling)
model_arch = [
    [3, [(3, 32)], True],
    [32, [(3, 64)], True],
    [64, [(3, 128), (1, 64), (3, 128)], True],
    [128, [(3, 256), (1, 128), (3, 256)], True],
    [256, [(3, 512), (1, 256), (3, 512), (1, 256), (3, 512)], False],
    [512, [(3, 1024), (1, 512), (3, 1024), (1, 512), (3, 1024)], False]
]


class DarkNet19(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.in_channel = in_channel

        self.conv1 = self._make_layers(*model_arch[0])
        self.conv2 = self._make_layers(*model_arch[1])
        self.conv3 = self._make_layers(*model_arch[2])
        self.conv4 = self._make_layers(*model_arch[3])
        self.conv5 = self._make_layers(*model_arch[4]) # use (512, 26, 26) for skip-connection
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = self._make_layers(*model_arch[5])


    def forward(self, x):
        '''
        :param x (B, 3, 416, 416)
        '''

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        skip = x.clone().detach()
        skip = skip.view(-1, 2048, 13, 13).contiguous()

        x = self.max_pool(x)
        x = self.conv6(x)

        return x, skip


    def _make_layers(self, in_channel, conv_blocks: list, pool=True):
        '''
        Parameters:
            in_channels (int)
            conv_blocks (List[tuple]): [(kernel_size, out_channel),...]
        '''

        blocks = []

        for block in conv_blocks:
            kernel_size, out_channel = block
            padding = 1 if kernel_size == 3 else 0

            blocks.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            blocks.append(nn.BatchNorm2d(out_channel, track_running_stats=False))
            blocks.append(nn.LeakyReLU(0.1, inplace=True))
            in_channel = out_channel

        if pool:
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*blocks)

    def load_weight(self):
        #  url = 'https://github.com/frgfm/Holocron/releases/download/v0.1.3/darknet19_224-b1ce16a5.pt'
        #
        #  weight_path = 'pretrained/darknet19-pretrained.pth'
        #  if not osp.exists(weight_path):
        #      print("Downloading weights..")
        #      wget.download(url=url, out=weight_path)
        #
        #  checkpoint = torch.load(weight_path)
        #  del checkpoint['classifier.weight']
        #  del checkpoint['classifier.bias']

        checkpoint = torch.load('pretrained/darknet19_2.pth')
        dct = {}
        for k, v in zip(self.state_dict().keys(), checkpoint.values()):
            #  print("{}".format(k))
            dct[k] = v

        print("Loading pretrained weight for Darknet19...")
        self.load_state_dict(dct)




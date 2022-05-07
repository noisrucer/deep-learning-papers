import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, out_channel, upsample_cfg=dict(mode='nearest')):
        '''
        Parameters
            in_channels (List[int]): List of feature map dimensions
            out_channel (int): Output dimension(channel)
            upsample_cfg (dict): config for upsampling
        '''

        super().__init__()

        assert type(in_channels) == list
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.upsample_cfg = upsample_cfg
        self.n_in_features = len(in_channels)

        self.lateral_convs = nn.ModuleList()

        self.fpn_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                                  stride=1, padding=1)

        for i in range(self.n_in_features):
            lateral_conv = nn.Conv2d(in_channels[i], out_channel, kernel_size=1,
                                     stride=1, padding=0)
            self.lateral_convs.append(lateral_conv)

        self._normal_init(self.lateral_convs, 0, 0.01)
        self._normal_init(self.fpn_conv, 0, 0.01)

        self.P6_conv = nn.Conv2d(in_channels[2], out_channel, kernel_size=3, stride=2, padding=1)
        self.P7_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1)
        )


    def forward(self, feature_list: list):
        C3, C4, C5 = feature_list

        laterals = [
            self.lateral_convs[idx](feat_map) for (idx, feat_map) in
            enumerate(feature_list)
        ]

        outs = []

        merged = laterals[self.n_in_features - 1]
        outs.append(self._copy(self.fpn_conv(merged)))

        for lateral in laterals[-2::-1]:
            m = lateral + F.interpolate(merged, size=lateral.shape[-2:],
                                        **self.upsample_cfg)
            outs.append(self._copy(self.fpn_conv(m)))
            merged = m

        P5, P4, P3 = outs

        # Retinanet - P5, P6
        P6 = self.P6_conv(C5)
        P7 = self.P7_conv(P6)

        return [P3, P4, P5, P6, P7]


    def _copy(self, t):
        return t.detach().clone()

    def _normal_init(self, convs, mean, std):
        if isinstance(convs, nn.ModuleList):
            for conv in convs:
                conv.weight.data.normal_(mean, std)
                conv.bias.data.zero_()
        else:
            convs.weight.data.normal_(mean, std)
            convs.bias.data.zero_()

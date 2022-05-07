```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self,
                 in_channels: list,
                 out_channel,
                 upsample_cfg=dict(mode="nearest")
                ):
        """
        in_channels (List[int]): List of feature map dimensions used for FPN
        out_channel (int): Output dimension(channel) for FPN
        upsample_cfg (dict): config for upsampling (for F.interpolate)
        """
        super().__init__()

        assert type(in_channels) == list
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.upsample_cfg = upsample_cfg # for upsampling
        self.n_in_features = len(in_channels)

        # lateral_conv is 1x1 conv that's applied to bottom-up feature-maps to reduce the channel size
        self.lateral_convs = nn.ModuleList()

        # fpn_conv is 3x3 conv that's applied to P_x
        self.fpn_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        # Store 1x1 conv layer
        for i in range(self.n_in_features):
            lateral_conv = nn.Conv2d(in_channels[i], out_channel, kernel_size=1, stride=1, padding=0)
            self.lateral_convs.append(lateral_conv)

        self._normal_init(self.lateral_convs, 0, 0.01)
        self._normal_init(self.fpn_conv, 0, 0.01)


    def forward(self, feature_list: list):
        # Construct laterals (after 1x1 conv)
        laterals = [
            self.lateral_convs[idx](feat_map) for (idx, feat_map) in enumerate(feature_list)
        ]

        # Output feature maps of FPN
        outs = []

        # Start Top-down pathway: merge with laterals
        merged = laterals[self.n_in_features - 1] # Start the iteration with top-most
        outs.append(self._copy(self.fpn_conv(merged)))

        # Remaining
        for lateral in laterals[-2::-1]:
            # F.interpolate: Upsampling
            # Lateral connection with element-wise addition
            m = lateral + F.interpolate(merged, size=lateral.shape[-2:], **self.upsample_cfg)
            outs.append(self._copy(self.fpn_conv(m)))
            merged = m

        return outs[::-1]


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


# For Testing
def main():
    in_channels = [2, 3, 5, 7] # Channels
    scales = [340, 170, 84, 43] # Spatial Dimension

    # Creating dummy data
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]

    # Output channel
    out_channel = 256

    fpn = FPN(in_channels, out_channel).eval()
    outputs = fpn(inputs)

    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')

# Start
main()

'''
outputs[0].shape = torch.Size([1, 256, 340, 340])
outputs[1].shape = torch.Size([1, 256, 170, 170])
outputs[2].shape = torch.Size([1, 256, 84, 84])
outputs[3].shape = torch.Size([1, 256, 43, 43])
'''
```

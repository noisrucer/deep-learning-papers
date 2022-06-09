import torch
import torch.nn as nn

from .spatial_path import SpatialPath
from .context_path import ContextPath
from .attention_refinement_module import AttentionRefinementModule
from .feature_fusion_module import FeatureFusionModule

class BiSeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, training=True):
        super().__init__()
        self.training = training

        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        self.attention_refinement_module_16x = AttentionRefinementModule(256, 256)
        self.attention_refinement_module_32x = AttentionRefinementModule(512, 512)
        self.feature_fusion_module = FeatureFusionModule(1024, num_classes) # spatial(256) + context(256+512)
        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

        # Auxiliary Loss Functions
        self.aux_conv_16x = nn.Conv2d(256, num_classes, kernel_size=1)
        self.aux_conv_32x = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Spatial Path
        sp_x = self.spatial_path(x) # 1/8, channel 256

        # Context Path
        # down_16x: 1/16, channel 256
        # down_32x: 1/32, channel 512
        # tail: resolution 1, channel 512
        down_16x, down_32x, tail = self.context_path(x)

        # Attention Refinement Module
        down_16x = self.attention_refinement_module_16x(down_16x)
        down_32x = self.attention_refinement_module_32x(down_32x)
        down_32x = torch.mul(down_32x, tail)

        # Upsampling to match sp_x resolution - 1/8
        down_16x = nn.functional.interpolate(down_16x, size=sp_x.size()[-2:], mode='bilinear', align_corners=True)
        down_32x = nn.functional.interpolate(down_32x, size=sp_x.size()[-2:], mode='bilinear', align_corners=True)

        # For Auxiliary Loss Functions
        if self.training:
            aux_down_16x = self.aux_conv_16x(down_16x) # 1/8, channel num_classes
            aux_down_32x = self.aux_conv_32x(down_32x) # 1/8, channel num_classes
            aux_down_16x = nn.functional.interpolate(aux_down_16x, scale_factor=8, mode='bilinear', align_corners=True)
            aux_down_32x = nn.functional.interpolate(aux_down_32x, scale_factor=8, mode='bilinear', align_corners=True)

        # Concatenate spatial path outputs to feed into Feature Fusion Module
        cp_x = torch.cat([down_16x, down_32x], dim=1) # channel 256+512

        # Feature Fusion Module
        fused = self.feature_fusion_module(sp_x, cp_x) # 1/8, channel num_classes

        # Upsample to original resolution
        fused_upsampled = nn.functional.interpolate(fused, scale_factor=8, mode='bilinear', align_corners=True)

        # Final prediction
        out = self.final_conv(fused_upsampled) # (B, num_classes, H, W)

        if self.training:
            return out, aux_down_16x, aux_down_32x
        else:
            return out

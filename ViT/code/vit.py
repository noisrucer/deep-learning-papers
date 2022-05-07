import torch
import torch.nn as nn

from patch_embedding import PatchEmbedding
from encoder import Encoder
from classification_head import ClassificationHead

from torchsummary import summary

class ViT(nn.Module):
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 n_heads=8,
                 emb_dim=768,
                 attn_drop=0.,
                 drop_path=0.,
                 expansion=4,
                 img_size=224,
                 depth=12,
                 n_classes=10,
                 ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.encoder = Encoder(depth, emb_dim, n_heads, attn_drop, drop_path, expansion)
        self.classification_head = ClassificationHead(emb_dim, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.classification_head(x)
        return x

dummy = torch.randn(4, 3, 224, 224)
vit = ViT()
out = vit(dummy)

print("Output shape: {}".format(out.shape))

"""
Output shape: torch.Size([4, 10])
"""

summary(vit, (3, 224, 224), device='cpu')
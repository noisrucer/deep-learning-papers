'''
Reference: github.com/FrancescoSaverioZuppichini/ViT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=768, img_size=224):
        super().__init__()
        self.patch_size = 16
        self.n_patches = (img_size ** 2) // (patch_size ** 2)

        # [1] Project Method 1
        self.img2token = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size)
        self.linear_projection = nn.Linear(in_channels * (patch_size ** 2), emb_dim)

        # [1] Project Method 2 using Conv2d
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim, patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )

        # [2] Class Token - from BERT
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # [3] Position Embedding
        self.position_embedding = nn.Parameter(
            torch.randn(self.n_patches + 1, emb_dim)
        )


    def forward(self, x):
        B = x.shape[0]

        # [1] Tokenize & Linear Projection
        x = self.projection(x) # (B, n, e)

        # [2] Prepend class tokens
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        x = torch.cat([cls_tokens, x], dim=1) # (B, n+1, e)

        # [3] Position Embedding
        x += self.position_embedding # (B, n+1, e)

        return x

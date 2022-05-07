import torch.nn as nn
from einops.layers.torch import Reduce

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_dim=768, n_classes=10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, n_classes)
        )

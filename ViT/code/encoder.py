import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange, repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim=768, n_heads=8, attn_drop=0.):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (h d qkv) -> qkv b h n d", h=self.n_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2] # each (b, h, n, d_h)

        d_h = (self.emb_dim / self.n_heads) ** (0.5)
        attention = torch.einsum('bhqd, bhkd -> bhqk', q, k) # (b, h, n, n)
        attention = F.softmax(attention, dim=-1) / d_h
        attention = self.attn_drop(attention) # (b, h, n, n)

        out = torch.einsum('bhqk, bhvd -> bhqd', attention, v) # (b, h, n, d_h)
        out = rearrange(out, "b h n d_h -> b n (h d_h)") # (b, n, d)
        out = self.projection(out) # (b, n, emb_dim)

        return out

class MLP(nn.Sequential):
    def __init__(self, emb_dim, expansion=4, drop_path=0.):
        super().__init__(
            nn.Linear(emb_dim, emb_dim * expansion),
            nn.GELU(),
            nn.Dropout(drop_path),
            nn.Linear(emb_dim * expansion, emb_dim)
        )


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim=768, n_heads=8, attn_drop=0.,
                 drop_path=0., expansion=4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            MultiHeadAttention(emb_dim, n_heads, attn_drop),
            nn.Dropout(drop_path)
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            MLP(emb_dim, expansion, drop_path),
            nn.Dropout(drop_path)
        )

    def forward(self, x):
        skip1 = x
        x = self.block1(x)
        x += skip1

        skip2 = x
        x = self.block2(x)
        x += skip2

        return x


class Encoder(nn.Sequential):
    def __init__(self, depth=12, emb_dim=768, n_heads=8, attn_drop=0.,
                 drop_path=0., expansion=4):
        super().__init__(
            *[EncoderBlock(emb_dim, n_heads, attn_drop, drop_path, expansion)
              for _ in range(depth)]
        )


from __future__ import annotations
import torch
import torch.nn as nn
from timm.layers import Mlp
from .mlp_baseline import sinusoidal_time_emb

class TimmMLPNoisePredictor(nn.Module):
    """使用 timm 的 Mlp 作为特征级块，适配二维输入。"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, time_embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        in_features = input_dim + time_embed_dim
        layers = [Mlp(in_features=in_features, hidden_features=hidden_dim, out_features=hidden_dim, drop=dropout)]
        for _ in range(max(0, num_layers - 2)):
            layers.append(Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim, drop=dropout))
        self.blocks = nn.Sequential(*layers)
        self.proj_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_time_emb(t, self.time_embed_dim)
        h = torch.cat([x, te], dim=1)
        h = self.blocks(h)
        return self.proj_out(h)

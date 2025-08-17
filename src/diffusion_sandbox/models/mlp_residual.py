from __future__ import annotations
import torch
import torch.nn as nn
from .mlp_baseline import sinusoidal_time_emb

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.seq(x))

class ResidualMLPNoisePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, time_embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.in_proj = nn.Linear(input_dim + time_embed_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_time_emb(t, self.time_embed_dim)
        h = self.in_proj(torch.cat([x, te], dim=1))
        h = self.blocks(h)
        return self.out_proj(h)

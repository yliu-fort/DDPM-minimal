from __future__ import annotations
import math, torch
import torch.nn as nn

def sinusoidal_time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class MLPNoisePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, time_embed_dim: int) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim + time_embed_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, input_dim)]
        self.net = nn.Sequential(*layers)
        self.time_embed_dim = time_embed_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_time_emb(t, self.time_embed_dim)
        return self.net(torch.cat([x, te], dim=1))

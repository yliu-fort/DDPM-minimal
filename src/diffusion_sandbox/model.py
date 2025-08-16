from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn

def sinusoidal_time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal time embedding (positional encoding)。"""

    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)
    )
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
        x = torch.cat([x, te], dim=1)
        return self.net(x)

@dataclass
class DiffusionCoeffs:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    posterior_variance: torch.Tensor

class DDPM:
    def __init__(self, timesteps: int, beta_schedule: str, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.timesteps = timesteps
        self.device = device
        self.coeffs = self._build_coeffs(timesteps, beta_schedule, beta_start, beta_end, device)

    @staticmethod
    def _build_coeffs(T: int, schedule: str, beta_start: float, beta_end: float, device: torch.device) -> DiffusionCoeffs:
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, device=device)
        elif schedule == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, device=device)
            alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(0.0001, 0.9999)
        else:
            raise ValueError("Unknown beta schedule")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        return DiffusionCoeffs(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1 - alphas_cumprod),
            alphas_cumprod_prev=alphas_cumprod_prev,
            posterior_variance=betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        c = self.coeffs
        sqrt_acp = c.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_om = c.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_acp * x0 + sqrt_om * noise

    def p_sample(self, model: MLPNoisePredictor, x: torch.Tensor, t: int) -> torch.Tensor:
        c = self.coeffs
        t_batch = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        eps = model(x, t_batch)
        beta_t = c.betas[t]
        acp_t = c.alphas_cumprod[t]
        acp_prev = c.alphas_cumprod_prev[t]
        mean = (1 / torch.sqrt(c.alphas[t])) * (x - beta_t / torch.sqrt(1 - acp_t) * eps)
        if t == 0:
            return mean
        var = c.posterior_variance[t]
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: MLPNoisePredictor, n: int) -> torch.Tensor:
        x = torch.randn(n, 2, device=self.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t)
        return x

    def loss(self, model: MLPNoisePredictor, x0: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        eps_pred = model(x_noisy, t)
        return torch.mean((noise - eps_pred) ** 2)

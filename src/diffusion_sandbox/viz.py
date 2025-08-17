from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
import torch

def scatter_2d(real: Optional[torch.Tensor], fake: Optional[torch.Tensor], title: str = ""):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    if real is not None:
        r = real.detach().cpu().numpy()
        ax.scatter(r[:, 0], r[:, 1], s=3, alpha=0.6, label="real")
    if fake is not None:
        f = fake.detach().cpu().numpy()
        ax.scatter(f[:, 0], f[:, 1], s=3, alpha=0.6, label="fake")
    ax.set_title(title); ax.legend(); ax.set_aspect("equal")
    fig.tight_layout()
    return fig

def image_grid(x: torch.Tensor, nrow: int = 8, title: str = ""):
    x = (x.detach().cpu().clamp(-1,1) + 1) / 2.0
    B, C, H, W = x.shape
    nrow = min(nrow, B)
    ncol = (B + nrow - 1) // nrow
    fig = plt.figure(figsize=(nrow, ncol))
    for i in range(B):
        ax = fig.add_subplot(ncol, nrow, i+1)
        ax.imshow(x[i].permute(1,2,0).numpy())
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig

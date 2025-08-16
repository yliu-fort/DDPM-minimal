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
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig

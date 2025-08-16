#!/usr/bin/env python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path
import torch
from diffusion_sandbox.data import SyntheticGMM, SyntheticGMMCfg, SyntheticRing, SyntheticRingCfg
from diffusion_sandbox.viz import scatter_2d

out_dir = Path(__file__).resolve().parents[1] / "examples" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# GMM 可视化
gmm = SyntheticGMM(2000, SyntheticGMMCfg(num_modes=8, radius=4.0, std=0.15), seed=42)
fig = scatter_2d(torch.stack([gmm[i] for i in range(len(gmm))]), None, title="Synthetic GMM")
fig.savefig(out_dir / "gmm.png")

# Ring 可视化
ring = SyntheticRing(2000, SyntheticRingCfg(radius=4.0, noise_std=0.05), seed=42)
fig = scatter_2d(torch.stack([ring[i] for i in range(len(ring))]), None, title="Synthetic Ring")
fig.savefig(out_dir / "ring.png")

print(f"Saved figures to {out_dir}")
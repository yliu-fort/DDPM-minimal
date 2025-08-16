from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

@dataclass
class SyntheticGMMCfg:
    num_modes: int
    radius: float
    std: float

class SyntheticGMM(Dataset[torch.Tensor]):
    """Synthesize 2D GMM data: place multiple Gaussian modes equally spaced on a circular ring."""

    def __init__(self, n: int, cfg: SyntheticGMMCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        modes = []
        for k in range(cfg.num_modes):
            theta = 2 * math.pi * k / cfg.num_modes
            cx, cy = cfg.radius * math.cos(theta), cfg.radius * math.sin(theta)
            pts = rng.normal(loc=(cx, cy), scale=cfg.std, size=(n // cfg.num_modes, 2))
            modes.append(pts)
        x = np.concatenate(modes, axis=0).astype(np.float32)
        rng.shuffle(x)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]

@dataclass
class SyntheticRingCfg:
    radius: float
    noise_std: float

class SyntheticRing(Dataset[torch.Tensor]):
    """Unit ring + Gaussian noise."""

    def __init__(self, n: int, cfg: SyntheticRingCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * math.pi, size=(n,))
        x = np.stack([cfg.radius * np.cos(theta), cfg.radius * np.sin(theta)], axis=1)
        x += rng.normal(0, cfg.noise_std, size=x.shape)
        self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]

def build_dataloader(name: str, n: int, batch_size: int, num_workers: int, gmm: SyntheticGMMCfg, ring: SyntheticRingCfg, seed: int) -> DataLoader:
    if name == "gmm":
        ds = SyntheticGMM(n=n, cfg=gmm, seed=seed)
    elif name == "ring":
        ds = SyntheticRing(n=n, cfg=ring, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Type, Tuple, List, Any

# ---------------------------
# Utilities
# ---------------------------

def _split_counts(n: int, k: int) -> List[int]:
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def _add_noise(x: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape)
    return x.astype(np.float32)


# ---------------------------
# 1) GMM on a ring
# ---------------------------

@dataclass
class SyntheticGMMCfg:
    num_modes: int = 8                 # number of modes
    radius: float = 4.0        # ring radius
    mode_std: float = 0.15     # std inside each Gaussian mode
    noise_std: float = 0.0     # global isotropic noise added after sampling


class SyntheticGMM(Dataset[torch.Tensor]):
    """2D GMM with modes equally spaced on a circle (ring). Shape: [n, 2]."""

    def __init__(self, n: int, cfg: SyntheticGMMCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        angles = np.linspace(0, 2 * np.pi, cfg.num_modes, endpoint=False)
        means = np.stack([cfg.radius * np.cos(angles), cfg.radius * np.sin(angles)], axis=1)  # [k, 2]

        counts = _split_counts(n, cfg.num_modes)
        chunks: List[np.ndarray] = []
        for i, c in enumerate(counts):
            mu = means[i]
            pts = rng.normal(loc=mu, scale=cfg.mode_std, size=(c, 2))
            chunks.append(pts)
        x = np.concatenate(chunks, axis=0)

        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# 2) Single ring (uniform angle, small radial jitter)
# ---------------------------

@dataclass
class SyntheticRingCfg:
    radius: float = 4.0
    radial_std: float = 0.05    # radial jitter around the ring
    noise_std: float = 0.0

class SyntheticRing(Dataset[torch.Tensor]):
    """Uniform points on a circle with small radial jitter. Shape: [n, 2]."""

    def __init__(self, n: int, cfg: SyntheticRingCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        theta = rng.uniform(0.0, 2 * np.pi, size=(n,))
        r = cfg.radius + rng.normal(0.0, cfg.radial_std, size=(n,))
        x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# 3) Two Moons
# ---------------------------

@dataclass
class SyntheticTwoMoonsCfg:
    radius: float = 1.0
    width: float = 0.3           # vertical offset between two moons
    gap: float = 0.0             # horizontal gap between two moons
    noise_std: float = 0.05


class SyntheticTwoMoons(Dataset[torch.Tensor]):
    """Two interleaving half-circles. Shape: [n, 2]."""

    def __init__(self, n: int, cfg: SyntheticTwoMoonsCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        n1, n2 = _split_counts(n, 2)
        t1 = rng.uniform(0.0, np.pi, size=(n1,))
        x1 = np.stack([cfg.radius * np.cos(t1), cfg.radius * np.sin(t1)], axis=1)

        t2 = rng.uniform(0.0, np.pi, size=(n2,))
        x2 = np.stack([cfg.radius * np.cos(t2) + cfg.radius + cfg.gap,
                       -cfg.radius * np.sin(t2) + cfg.width], axis=1)

        x = np.concatenate([x1, x2], axis=0)
        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# 4) Concentric Circles
# ---------------------------

@dataclass
class SyntheticConcentricCirclesCfg:
    num_rings: int = 3
    r_min: float = 1.0
    r_max: float = 4.0
    per_ring_balance: bool = True  # balance samples across rings
    noise_std: float = 0.02


class SyntheticConcentricCircles(Dataset[torch.Tensor]):
    """Multiple rings with radii in [r_min, r_max]. Shape: [n, 2]."""

    def __init__(self, n: int, cfg: SyntheticConcentricCirclesCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        radii = np.linspace(cfg.r_min, cfg.r_max, cfg.num_rings)
        if cfg.per_ring_balance:
            counts = _split_counts(n, cfg.num_rings)
        else:
            # area-proportional sampling as an alternative
            weights = radii / radii.sum()
            counts = (weights * n).astype(int).tolist()
            while sum(counts) < n:
                counts[-1] += 1

        chunks: List[np.ndarray] = []
        for r, c in zip(radii, counts):
            theta = rng.uniform(0.0, 2 * np.pi, size=(c,))
            pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
            chunks.append(pts)
        x = np.concatenate(chunks, axis=0)

        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# 5) Spiral (multi-arm)
# ---------------------------

@dataclass
class SyntheticSpiralCfg:
    arms: int = 2
    turns: float = 2.0          # number of spiral turns
    a: float = 0.2              # radial scale r = a * theta
    noise_std: float = 0.05


class SyntheticSpiral(Dataset[torch.Tensor]):
    """Multiple interleaving Archimedean spirals. Shape: [n, 2]."""

    def __init__(self, n: int, cfg: SyntheticSpiralCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        counts = _split_counts(n, cfg.arms)
        chunks: List[np.ndarray] = []
        for i, c in enumerate(counts):
            theta = rng.uniform(0.0, cfg.turns * 2 * np.pi, size=(c,))
            r = cfg.a * theta
            # phase offset per arm
            offset = 2 * np.pi * i / cfg.arms
            x_arm = np.stack([r * np.cos(theta + offset), r * np.sin(theta + offset)], axis=1)
            chunks.append(x_arm)

        x = np.concatenate(chunks, axis=0)
        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# 6) Checkerboard
# ---------------------------

@dataclass
class SyntheticCheckerboardCfg:
    x_range: Tuple[float, float] = (-2.0, 2.0)
    y_range: Tuple[float, float] = (-2.0, 2.0)
    cells: int = 4               # number of cells per axis
    jitter: float = 0.05         # local noise inside a cell
    noise_std: float = 0.0       # global noise after composing


class SyntheticCheckerboard(Dataset[torch.Tensor]):
    """Alternating occupied cells on a grid, with jitter. Shape: [n, 2]."""

    def __init__(self, n: int, cfg: SyntheticCheckerboardCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        x_lin = np.linspace(cfg.x_range[0], cfg.x_range[1], cfg.cells + 1)
        y_lin = np.linspace(cfg.y_range[0], cfg.y_range[1], cfg.cells + 1)
        # centers of each cell
        xs = 0.5 * (x_lin[:-1] + x_lin[1:])
        ys = 0.5 * (y_lin[:-1] + y_lin[1:])
        centers = []
        for i, cx in enumerate(xs):
            for j, cy in enumerate(ys):
                if (i + j) % 2 == 0:
                    centers.append((cx, cy))
        centers = np.array(centers, dtype=np.float32)
        k = centers.shape[0]

        counts = _split_counts(n, k)
        chunks: List[np.ndarray] = []
        for idx, c in enumerate(counts):
            cx, cy = centers[idx]
            pts = np.stack([
                rng.normal(cx, cfg.jitter, size=(c,)),
                rng.normal(cy, cfg.jitter, size=(c,))
            ], axis=1)
            chunks.append(pts)
        x = np.concatenate(chunks, axis=0)

        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# 7) Pinwheel
# ---------------------------

@dataclass
class SyntheticPinwheelCfg:
    arms: int = 5
    radial_std: float = 0.3
    tangential_std: float = 0.05
    rate: float = 0.25          # rotation rate (controls how much twist increases with radius)
    noise_std: float = 0.0


class SyntheticPinwheel(Dataset[torch.Tensor]):
    """
    Pinwheel dataset used in flow-based models.
    Points sampled in anisotropic Gaussians whose orientation rotates with radius. Shape: [n, 2].
    """

    def __init__(self, n: int, cfg: SyntheticPinwheelCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        counts = _split_counts(n, cfg.arms)
        chunks: List[np.ndarray] = []

        for i, c in enumerate(counts):
            # base samples in arm's local frame
            r = rng.normal(0.0, cfg.radial_std, size=(c, 1)) + 1.0  # shift to positive radius
            t = rng.normal(0.0, cfg.tangential_std, size=(c, 1))
            pts = np.concatenate([r, t], axis=1)  # [c, 2]

            # rotation angle grows with radius
            angle = cfg.rate * r.squeeze(-1) + (2 * np.pi * i / cfg.arms)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.stack([np.stack([cos_a, -sin_a], axis=1),
                            np.stack([sin_a,  cos_a], axis=1)], axis=1)  # [c, 2, 2]
            pts = (rot @ pts[..., None]).squeeze(-1)  # [c, 2]
            chunks.append(pts)

        x = np.concatenate(chunks, axis=0)
        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# 8) Swiss Roll → 2D projection
# ---------------------------

@dataclass
class SyntheticSwissRoll2DCfg:
    turns: float = 3.0                 # controls t range
    height: float = 5.0                # z range [-height, height]
    t_min: float = 1.5 * np.pi
    t_max: float = 1.5 * np.pi + 2 * np.pi * 3.0
    noise_std: float = 0.05

    def __post_init__(self):
        # allow overriding via turns if user changes it
        self.t_max = self.t_min + 2 * np.pi * self.turns

class SyntheticSwissRoll2D(Dataset[torch.Tensor]):
    """
    Classic swiss roll in 3D, projected to 2D by (x, z). Shape: [n, 2].
    x = t * cos t, y = t * sin t, z ~ U[-height, height], then take (x, z).
    """

    def __init__(self, n: int, cfg: SyntheticSwissRoll2DCfg, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)

        t = rng.uniform(cfg.t_min, cfg.t_max, size=(n,))
        z = rng.uniform(-cfg.height, cfg.height, size=(n,))
        x = np.stack([t * np.cos(t), z], axis=1)

        x = _add_noise(x, rng, cfg.noise_std)
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# ---------------------------
# Simple factory
# ---------------------------

_DATASET_REGISTRY: Dict[str, Tuple[Type[Dataset], Type[Any]]] = {
    "gmm": (SyntheticGMM, SyntheticGMMCfg),
    "ring": (SyntheticRing, SyntheticRingCfg),
    "two_moons": (SyntheticTwoMoons, SyntheticTwoMoonsCfg),
    "concentric": (SyntheticConcentricCircles, SyntheticConcentricCirclesCfg),
    "spiral": (SyntheticSpiral, SyntheticSpiralCfg),
    "checkerboard": (SyntheticCheckerboard, SyntheticCheckerboardCfg),
    "pinwheel": (SyntheticPinwheel, SyntheticPinwheelCfg),
    "swiss_roll2d": (SyntheticSwissRoll2D, SyntheticSwissRoll2DCfg),
}


def build_dataloader(name: str, n: int, batch_size: int, num_workers: int, cfg_kwargs: Dict[str, Any] | None = None, seed: int = 0) -> DataLoader:
    """
    Factory to build a synthetic dataset loader by name.

    Example:
        dl = build_dataloader(
            name="gmm",
            n=2048,
            batch_size=512,
            num_workers=0,
            cfg_kwargs={"num_modes": 12, "radius": 5.0, "mode_std": 0.1, "noise_std": 0.0},
            seed=42,
        )
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown synthetic dataset '{name}'. Available: {list(_DATASET_REGISTRY.keys())}")
    cls, cfg_cls = _DATASET_REGISTRY[name]
    cfg_kwargs = cfg_kwargs or {}
    cfg = cfg_cls(**cfg_kwargs)  # type: ignore
    ds = cls(n=n, cfg=cfg, seed=seed)  # type: ignore
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
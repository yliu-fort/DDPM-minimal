from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import yaml

@dataclass
class CudnnCfg:
    benchmark: bool
    deterministic: bool

@dataclass
class RunCfg:
    experiment_name: str
    output_dir: str
    seed: int
    device: Literal["auto", "cpu", "cuda"]
    cudnn: CudnnCfg

@dataclass
class GMMCfg:
    num_modes: int
    radius: float
    mode_std: float
    noise_std: float

@dataclass
class RingCfg:
    radius: float
    radial_std: float
    noise_std: float

@dataclass
class TwoMoonsCfg:
    radius: float
    width: float
    gap: float
    noise_std: float

@dataclass
class ConcentricCirclesCfg:
    num_rings: int
    r_min: float
    r_max: float
    per_ring_balance: bool
    noise_std: float

@dataclass
class SpiralCfg:
    arms: int
    turns: float
    a: float
    noise_std: float

@dataclass
class CheckerboardCfg:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    cells: int
    jitter: float
    noise_std: float

@dataclass
class PinwheelCfg:
    arms: int
    radial_std: float
    tangential_std: float
    rate: float
    noise_std: float

@dataclass
class SwissRoll2DCfg:
    turns: float
    height: float
    noise_std: float

@dataclass
class DataCfg:
    name: Literal["gmm", "ring", "concentric", "two_moons", "swiss_roll2d", "spiral", "checkerboard", "pinwheel"]
    num_samples: int
    batch_size: int
    num_workers: int
    gmm: GMMCfg
    ring: RingCfg
    concentric: ConcentricCirclesCfg
    two_moons: TwoMoonsCfg
    swiss_roll2d: SwissRoll2DCfg
    spiral: SpiralCfg
    checkerboard: CheckerboardCfg
    pinwheel: PinwheelCfg

@dataclass
class DiffusionCfg:
    timesteps: int
    beta_schedule: Literal["linear", "cosine"]
    beta_start: float
    beta_end: float

@dataclass
class ModelCfg:
    input_dim: int
    hidden_dim: int
    num_layers: int
    time_embed_dim: int

@dataclass
class TrainCfg:
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    log_interval: int
    ckpt_interval: int
    sample_interval: int
    sample_size: int

@dataclass
class MlflowCfg:
    enabled: bool
    tracking_uri: str
    experiment_name: str

@dataclass
class TrackingCfg:
    tensorboard: bool
    mlflow: MlflowCfg

@dataclass
class Cfg:
    run: RunCfg
    data: DataCfg
    diffusion: DiffusionCfg
    model: ModelCfg
    train: TrainCfg
    tracking: TrackingCfg


def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Cfg(
        run=RunCfg(**{k: v for k, v in raw["run"].items() if k not in {"cudnn",}}, cudnn=CudnnCfg(**raw["run"]["cudnn"])),
        data=DataCfg(
            **{k: v for k, v in raw["data"].items() if k not in {"gmm", "ring", "concentric", "two_moons", "swiss_roll2d", "spiral", "checkerboard", "pinwheel"}},
            gmm=GMMCfg(**raw["data"]["gmm"]),
            ring=RingCfg(**raw["data"]["ring"]),
            concentric=ConcentricCirclesCfg(**raw["data"]["concentric"]),
            two_moons=TwoMoonsCfg(**raw["data"]["two_moons"]),
            swiss_roll2d=SwissRoll2DCfg(**raw["data"]["swiss_roll2d"]),
            spiral=SpiralCfg(**raw["data"]["spiral"]),
            checkerboard=CheckerboardCfg(**raw["data"]["checkerboard"]),
            pinwheel=PinwheelCfg(**raw["data"]["pinwheel"]),
        ),
        diffusion=DiffusionCfg(**raw["diffusion"]),
        model=ModelCfg(**raw["model"]),
        train=TrainCfg(**raw["train"]),
        tracking=TrackingCfg(tensorboard=raw["tracking"]["tensorboard"], mlflow=MlflowCfg(**raw["tracking"]["mlflow"])),
    )
    return cfg

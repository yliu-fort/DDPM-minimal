from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any
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
class DataCfg:
    name: Literal["gmm", "ring", "concentric", "two_moons", "swiss_roll2d", "spiral", "checkerboard", "pinwheel", "cifar10"]
    num_samples: int
    batch_size: int
    num_workers: int
    cfg: Dict[str, Any]

@dataclass
class DiffusionCfg:
    timesteps: int
    beta_schedule: Literal["linear", "cosine"]
    beta_start: float
    beta_end: float

@dataclass
class ModelCfg:
    input_dim: int = 2
    hidden_dim: int = 128
    num_layers: int = 3
    time_embed_dim: int = 32
    name: str = "mlp_baseline"
    common: Optional[Dict[str, Any]] = None
    mlp_baseline: Optional[Dict[str, Any]] = None
    mlp_residual: Optional[Dict[str, Any]] = None
    timm_mlp: Optional[Dict[str, Any]] = None
    diffusers_unet1d: Optional[Dict[str, Any]] = None
    diffusers_unet2d: Optional[Dict[str, Any]] = None

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
            **{k: v for k, v in raw["data"].items() if k not in {}},
        ),
        diffusion=DiffusionCfg(**raw["diffusion"]),
        model=ModelCfg(**raw.get("model", {})),
        train=TrainCfg(**raw["train"]),
        tracking=TrackingCfg(tensorboard=raw["tracking"]["tensorboard"], mlflow=MlflowCfg(**raw["tracking"]["mlflow"])),
    )
    return cfg

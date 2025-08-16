from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
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
    std: float

@dataclass
class RingCfg:
    radius: float
    noise_std: float

@dataclass
class DataCfg:
    name: Literal["gmm", "ring"]
    num_samples: int
    batch_size: int
    num_workers: int
    gmm: GMMCfg
    ring: RingCfg

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
            **{k: v for k, v in raw["data"].items() if k not in {"gmm", "ring"}},
            gmm=GMMCfg(**raw["data"]["gmm"]),
            ring=RingCfg(**raw["data"]["ring"]),
        ),
        diffusion=DiffusionCfg(**raw["diffusion"]),
        model=ModelCfg(**raw["model"]),
        train=TrainCfg(**raw["train"]),
        tracking=TrackingCfg(tensorboard=raw["tracking"]["tensorboard"], mlflow=MlflowCfg(**raw["tracking"]["mlflow"])),
    )
    return cfg

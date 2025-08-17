from __future__ import annotations
from typing import Dict, Type
import torch.nn as nn

from .mlp_baseline import MLPNoisePredictor
from .mlp_residual import ResidualMLPNoisePredictor
from .timm_mlp import TimmMLPNoisePredictor
from .diffusers_unet1d import DiffusersUNet1DNoisePredictor
from .diffusers_unet2d import DiffusersUNet2DNoisePredictor

REGISTRY: Dict[str, Type[nn.Module]] = {
    "mlp_baseline": MLPNoisePredictor,
    "mlp_residual": ResidualMLPNoisePredictor,
    "timm_mlp": TimmMLPNoisePredictor,
    "diffusers_unet1d": DiffusersUNet1DNoisePredictor,
    "diffusers_unet2d": DiffusersUNet2DNoisePredictor,
}

from __future__ import annotations
import torch
import torch.nn as nn
from diffusers.models import UNet2DModel

class DiffusersUNet2DNoisePredictor(nn.Module):
    """UNet2D，支持无条件或类别条件（通过 class_labels）。"""
    def __init__(
        self,
        input_dim: int,
        time_embed_dim: int,
        base_channels: int = 128,
        layers_per_block: int = 2,
        attn_on_16x16: bool = True,
        num_classes: int = 0,
        class_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        down_blocks = ["DownBlock2D", "DownBlock2D", "DownBlock2D"]
        up_blocks = ["UpBlock2D", "UpBlock2D", "UpBlock2D"]
        if attn_on_16x16:
            down_blocks[1] = "AttnDownBlock2D"
            up_blocks[1] = "AttnUpBlock2D"

        self.model = UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(base_channels, base_channels*2, base_channels*2),
            layers_per_block=layers_per_block,
            down_block_types=tuple(down_blocks),
            up_block_types=tuple(up_blocks),
            class_embed_type=("timestep" if num_classes > 0 else None),
            num_class_embeds=(num_classes if num_classes > 0 else None),
            dropout=class_dropout_prob if num_classes > 0 else 0.0,
            #norm_num_groups=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if self.num_classes > 0:
            return self.model(x, t, class_labels=y).sample
        else:
            return self.model(x, t).sample

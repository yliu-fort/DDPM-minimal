from __future__ import annotations
import torch
import torch.nn as nn
from diffusers.models import UNet1DModel

class DiffusersUNet1DNoisePredictor(nn.Module):
    """
    将二维点 [B,2] 线性映射为长度 L 的一维序列 [B,L]，经 UNet1D 建模后再投影回 [B,2]。
    这样可以保证任何 padding 都小于 L，避免长度过短导致的 padding 报错。
    """
    def __init__(
        self,
        input_dim: int,
        time_embed_dim: int,           # 与其他模型接口对齐，这里不直接用
        base_channels: int = 64,       # 建议 >= 64，避免注意力 num_heads=0 的问题
        layers_per_block: int = 2,
        down_block_types=None,
        up_block_types=None,
        seq_len: int = 32,             # L 必须明显大于所有卷积的 padding；16 或 32 都行
    ) -> None:
        super().__init__()
        if input_dim != 2:
            raise ValueError("当前适配器假设 input_dim=2（二维合成数据）。")
        if seq_len < 8:
            raise ValueError("seq_len 太短，需 >= 8 以适配 UNet1D 的 padding。")

        self.seq_len = seq_len
        self.in_proj = nn.Linear(input_dim, seq_len)
        self.out_proj = nn.Linear(seq_len, input_dim)

        down_block_types = down_block_types or ["DownBlock1D", "DownBlock1D"]
        up_block_types   = up_block_types   or ["UpBlock1D",   "UpBlock1D"]

        self.model = UNet1DModel(
            sample_size=seq_len,                  # ★ 关键：告诉 UNet 我们的序列长度是 L
            in_channels=1,
            out_channels=1,
            block_out_channels=(base_channels, base_channels * 2),
            layers_per_block=layers_per_block,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # [B,2] -> [B,L]
        h = self.in_proj(x)
        y = self.model(h.unsqueeze(1), t).sample  # [B,1,L]
        y = y.squeeze(1)                          # [B,L]
        return self.out_proj(y)                   # [B,2]
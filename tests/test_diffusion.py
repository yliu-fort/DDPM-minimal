import unittest
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from diffusion_sandbox.model import DDPM
from diffusion_sandbox.models import REGISTRY

def _forward_loss(Model):
    model = Model(input_dim=2, hidden_dim=32, num_layers=2, time_embed_dim=16) if "MLP" in Model.__name__ else Model(input_dim=2, time_embed_dim=16, base_channels=8)
    ddpm = DDPM(timesteps=10, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, device=torch.device("cpu"))
    x0 = torch.randn(64, 2)
    loss = ddpm.loss(model, x0)
    assert loss.dim() == 0

class TestUtils(unittest.TestCase):
    def test_registry_models_forward_and_sample(self):
        for name in ["mlp_baseline", "mlp_residual"]:
            Model = REGISTRY[name]
            _forward_loss(Model)

        # timm 与 diffusers 仅做一次前向 shape 冒烟（避免下载权重）
        
        for name in ["timm_mlp", "diffusers_unet1d"]:
            Model = REGISTRY[name]
            if name == "timm_mlp":
                m = Model(input_dim=2, hidden_dim=32, num_layers=2, time_embed_dim=16)
            else:
                m = Model(input_dim=2, time_embed_dim=16, base_channels=64, seq_len=32)
            x = torch.randn(8, 2)
            t = torch.randint(0, 10, (8,))
            y = m(x, t)
            assert y.shape == x.shape

    def test_unet2d_forward_shape(self):
        Model = REGISTRY["diffusers_unet2d"]
        m = Model(input_dim=3, time_embed_dim=128, base_channels=128, layers_per_block=2, attn_on_16x16=True, num_classes=0)
        x = torch.randn(4, 3, 32, 32)
        t = torch.randint(0, 10, (4,))
        y = m(x, t)
        assert y.shape == x.shape

if __name__ == "__main__":
    unittest.main()
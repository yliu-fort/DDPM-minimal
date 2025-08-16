import unittest
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from diffusion_sandbox.model import MLPNoisePredictor, DDPM

class TestUtils(unittest.TestCase):
    def test_forward_and_sample(self):
        device = torch.device("cpu")
        model = MLPNoisePredictor(input_dim=2, hidden_dim=32, num_layers=3, time_embed_dim=16).to(device)
        ddpm = DDPM(timesteps=10, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, device=device)

        x0 = torch.randn(64, 2)
        loss = ddpm.loss(model, x0)
        assert loss.dim() == 0

        with torch.no_grad():
            samples = ddpm.sample(model, 128)
            assert samples.shape == (128, 2)

if __name__ == "__main__":
    unittest.main()
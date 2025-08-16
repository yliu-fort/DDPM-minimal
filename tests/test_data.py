import unittest
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffusion_sandbox.data import SyntheticGMM, SyntheticGMMCfg, SyntheticRing, SyntheticRingCfg

class TestUtils(unittest.TestCase):
    def test_gmm_shape(self):
        ds = SyntheticGMM(800, SyntheticGMMCfg(), seed=0)
        assert len(ds) == 800
        x = ds[0]
        assert x.shape[0] == 2

    def test_ring_shape(self):
        ds = SyntheticRing(1000, SyntheticRingCfg(), seed=0)
        assert len(ds) == 1000
        x = ds[0]
        assert x.shape[0] == 2

if __name__ == "__main__":
    unittest.main()
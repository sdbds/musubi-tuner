import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musubi_tuner.soar_train_utils import get_sigmas_from_continuous_timesteps  # noqa: E402


class DummyConfig:
    num_train_timesteps = 1000


class DummyNoiseScheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.config = DummyConfig()
        self.config.num_train_timesteps = num_train_timesteps
        self.sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        self.timesteps = (self.sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)


class TestHvTrainNetworkSigmas(unittest.TestCase):
    def test_continuous_timestep_sampling_uses_analytic_sigma(self):
        scheduler = DummyNoiseScheduler()
        timesteps = torch.tensor([124.45, 501.0], dtype=torch.float32)

        sigmas = get_sigmas_from_continuous_timesteps(
            scheduler,
            timesteps,
            "cpu",
            n_dim=4,
            dtype=torch.float32,
        )

        self.assertEqual(sigmas.shape, torch.Size([2, 1, 1, 1]))
        self.assertTrue(torch.allclose(sigmas.flatten(), (timesteps - 1.0) / 1000.0))


if __name__ == "__main__":
    unittest.main()

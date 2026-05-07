import contextlib
import unittest
from argparse import Namespace

import torch

from tests.test_support import import_zimage_train_network_module


zimage_train_network = import_zimage_train_network_module()
ZImageNetworkTrainer = zimage_train_network.ZImageNetworkTrainer


class DummyTransformer:
    all_patch_size = (1,)

    def __call__(self, *, x, t, cap_feats, cap_mask):
        self.last_x = x
        self.last_t = t
        self.last_cap_feats = cap_feats
        self.last_cap_mask = cap_mask
        return x


class DummyAccelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return contextlib.nullcontext()


class TestZImagePredictVelocity(unittest.TestCase):
    def setUp(self):
        self.trainer = ZImageNetworkTrainer()
        self.args = Namespace(split_attn=False, gradient_checkpointing=False)
        self.accelerator = DummyAccelerator()
        self.transformer = DummyTransformer()
        self.noisy_model_input = torch.randn(2, 4, 2, 2)
        self.timesteps = torch.tensor([250.0, 750.0], dtype=torch.float32)
        self.llm_embed = [torch.randn(2, 8), torch.randn(3, 8)]

    def test_predict_velocity_returns_image_shaped_output(self):
        model_pred = self.trainer.predict_velocity(
            self.args,
            self.accelerator,
            self.transformer,
            {"llm_embed": self.llm_embed},
            self.noisy_model_input,
            self.timesteps,
            torch.float32,
        )

        self.assertEqual(model_pred.shape, self.noisy_model_input.shape)
        self.assertEqual(self.transformer.last_cap_feats.shape[0], self.noisy_model_input.shape[0])

    def test_call_dit_preserves_opposite_flow_matching_target(self):
        latents = torch.full_like(self.noisy_model_input, 2.0)
        noise = torch.full_like(self.noisy_model_input, -1.0)
        batch = {"llm_embed": self.llm_embed}

        model_pred, target = self.trainer.call_dit(
            self.args,
            self.accelerator,
            self.transformer,
            latents,
            batch,
            noise,
            self.noisy_model_input,
            self.timesteps,
            torch.float32,
        )

        self.assertEqual(model_pred.shape, self.noisy_model_input.shape)
        self.assertTrue(torch.equal(target, latents - noise))

    def test_soar_adapter_uses_zimage_target_convention(self):
        model_pred = torch.full_like(self.noisy_model_input, 3.0)
        clean = torch.full_like(self.noisy_model_input, 2.0)
        aux = torch.full_like(self.noisy_model_input, 5.0)
        sigma = torch.full((2, 1, 1, 1), 0.5)

        self.assertTrue(self.trainer.supports_soar(self.args))
        self.assertTrue(torch.equal(self.trainer.soar_velocity_to_standard(model_pred), -model_pred))
        self.assertTrue(torch.equal(self.trainer.soar_standard_to_local_target(clean, aux, sigma), torch.full_like(clean, -6.0)))


if __name__ == "__main__":
    unittest.main()

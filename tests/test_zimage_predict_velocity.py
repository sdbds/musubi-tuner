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

    def test_soar_cfg_rollout_combines_zimage_predictions_in_standard_space(self):
        # The CFG rollout now uses a single batched forward over [uncond; cond] (HY-SOAR style),
        # so the fake forward returns distinct values for the two halves and the test verifies
        # the combination works in standard-velocity space.
        args = Namespace(split_attn=False, gradient_checkpointing=False, soar_cfg_scale_sampling=4.5)
        self.trainer._soar_empty_llm_embed = torch.full((1, 8), 9.0)
        captured = {}
        bsize = self.noisy_model_input.shape[0]

        def fake_predict_velocity_for_soar(args, accelerator, transformer, batch, noisy_model_input, timesteps, network_dtype):
            captured["batch"] = batch
            captured["noisy_shape"] = tuple(noisy_model_input.shape)
            captured["ts_shape"] = tuple(timesteps.shape)
            # First half = uncond branch returns -1.0, second half = cond branch returns -3.0
            out = torch.empty_like(noisy_model_input)
            out[:bsize] = -1.0
            out[bsize:] = -3.0
            return out

        self.trainer.predict_velocity_for_soar = fake_predict_velocity_for_soar
        # cond_model_pred is the main-pass prediction; not used by the CFG-rollout path now
        # (the rollout re-runs cond inside the batched forward to match HY-SOAR).
        cond_model_pred = torch.full_like(self.noisy_model_input, -7.0)

        rollout = self.trainer.get_soar_rollout_velocity_standard(
            args,
            self.accelerator,
            self.transformer,
            {"llm_embed": self.llm_embed},
            self.noisy_model_input,
            self.timesteps,
            torch.float32,
            cond_model_pred,
        )

        # cond_std = -(-3.0) = 3.0 ; uncond_std = -(-1.0) = 1.0
        # combined = 1.0 + 4.5*(3.0 - 1.0) = 10.0
        self.assertTrue(torch.equal(rollout, torch.full_like(self.noisy_model_input, 10.0)))
        # Batched forward: 2*B
        self.assertEqual(captured["noisy_shape"][0], 2 * bsize)
        self.assertEqual(captured["ts_shape"][0], 2 * bsize)
        # llm_embed list has 2*B entries; first B are uncond (empty embed), last B are the cond originals
        self.assertEqual(len(captured["batch"]["llm_embed"]), 2 * bsize)
        for i in range(bsize):
            self.assertTrue(torch.equal(captured["batch"]["llm_embed"][i], self.trainer._soar_empty_llm_embed))
        for i, original in enumerate(self.llm_embed):
            self.assertTrue(torch.equal(captured["batch"]["llm_embed"][bsize + i], original))

    def test_soar_cfg_scale_one_keeps_cond_only_rollout(self):
        args = Namespace(soar_cfg_scale_sampling=1.0)
        cond_model_pred = torch.full_like(self.noisy_model_input, -3.0)

        rollout = self.trainer.get_soar_rollout_velocity_standard(
            args,
            self.accelerator,
            self.transformer,
            {"llm_embed": self.llm_embed},
            self.noisy_model_input,
            self.timesteps,
            torch.float32,
            cond_model_pred,
        )

        self.assertTrue(torch.equal(rollout, torch.full_like(self.noisy_model_input, 3.0)))


if __name__ == "__main__":
    unittest.main()

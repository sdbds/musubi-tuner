import unittest
from argparse import Namespace

import torch

from tests.test_support import import_qwen_image_train_network_module


qwen_image_train_network = import_qwen_image_train_network_module()
QwenImageNetworkTrainer = qwen_image_train_network.QwenImageNetworkTrainer


class TestQwenImageSoarLora(unittest.TestCase):
    def test_supports_soar_only_for_standard_text_to_image(self):
        trainer = QwenImageNetworkTrainer()

        self.assertTrue(
            trainer.supports_soar(Namespace(is_edit=False, is_layered=False, remove_first_image_from_target=False))
        )
        self.assertFalse(
            trainer.supports_soar(Namespace(is_edit=True, is_layered=False, remove_first_image_from_target=False))
        )
        self.assertFalse(
            trainer.supports_soar(Namespace(is_edit=False, is_layered=True, remove_first_image_from_target=False))
        )
        self.assertFalse(
            trainer.supports_soar(Namespace(is_edit=False, is_layered=False, remove_first_image_from_target=True))
        )

    def test_predict_velocity_for_soar_preserves_5d_auxiliary_shape(self):
        trainer = QwenImageNetworkTrainer()
        aux_latents = torch.randn(2, 4, 1, 2, 2)
        timesteps = torch.tensor([100.0, 200.0])
        calls = {}

        def fake_call_dit(args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps_arg, network_dtype):
            calls["latents"] = latents
            calls["noise"] = noise
            calls["noisy_model_input"] = noisy_model_input
            calls["timesteps"] = timesteps_arg
            calls["network_dtype"] = network_dtype
            return noisy_model_input + 1, torch.zeros_like(noisy_model_input)

        trainer.call_dit = fake_call_dit

        model_pred = trainer.predict_velocity_for_soar(
            Namespace(),
            accelerator=None,
            transformer=object(),
            batch={"vl_embed": [torch.zeros(1, 1)], "latents": aux_latents},
            noisy_model_input=aux_latents,
            timesteps=timesteps,
            network_dtype=torch.bfloat16,
        )

        self.assertTrue(torch.equal(model_pred, aux_latents + 1))
        self.assertEqual(calls["latents"].shape, torch.Size([2, 4, 1, 2, 2]))
        self.assertTrue(torch.equal(calls["noise"], torch.zeros_like(aux_latents)))
        self.assertTrue(torch.equal(calls["noisy_model_input"], aux_latents))
        self.assertTrue(torch.equal(calls["timesteps"], timesteps))
        self.assertEqual(calls["network_dtype"], torch.bfloat16)

    def test_cfg_rollout_replaces_vl_embed_only(self):
        trainer = QwenImageNetworkTrainer()
        trainer._soar_empty_vl_embed = torch.full((1, 8), 9.0)
        aux_latents = torch.randn(2, 4, 1, 2, 2)
        timesteps = torch.tensor([100.0, 200.0])
        captured = {}

        def fake_predict_velocity_for_soar(args, accelerator, transformer, batch, noisy_model_input, timesteps_arg, network_dtype):
            captured["batch"] = batch
            captured["noisy_model_input"] = noisy_model_input
            captured["timesteps"] = timesteps_arg
            return torch.full_like(noisy_model_input, 1.0)

        trainer.predict_velocity_for_soar = fake_predict_velocity_for_soar
        rollout = trainer.get_soar_rollout_velocity_standard(
            Namespace(
                soar_cfg_scale_sampling=4.5,
                is_edit=False,
                is_layered=False,
                remove_first_image_from_target=False,
            ),
            accelerator=None,
            transformer=object(),
            batch={"vl_embed": [torch.zeros(1, 8), torch.zeros(2, 8)], "latents": aux_latents},
            noisy_model_input=aux_latents,
            timesteps=timesteps,
            network_dtype=torch.bfloat16,
            cond_model_pred=torch.full_like(aux_latents, 3.0),
        )

        self.assertTrue(torch.equal(rollout, torch.full_like(aux_latents, 10.0)))
        self.assertEqual(len(captured["batch"]["vl_embed"]), 2)
        self.assertTrue(torch.equal(captured["batch"]["vl_embed"][0], trainer._soar_empty_vl_embed))
        self.assertTrue(torch.equal(captured["batch"]["latents"], aux_latents))
        self.assertTrue(torch.equal(captured["noisy_model_input"], aux_latents))
        self.assertTrue(torch.equal(captured["timesteps"], timesteps))


if __name__ == "__main__":
    unittest.main()

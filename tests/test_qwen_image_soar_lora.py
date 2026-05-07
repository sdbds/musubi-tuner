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


if __name__ == "__main__":
    unittest.main()

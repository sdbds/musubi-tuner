import unittest
from argparse import Namespace

import torch

from tests.test_support import import_flux_2_train_network_module


flux_2_train_network = import_flux_2_train_network_module()
Flux2NetworkTrainer = flux_2_train_network.Flux2NetworkTrainer


class TestFlux2SoarLora(unittest.TestCase):
    def test_supports_soar(self):
        self.assertTrue(Flux2NetworkTrainer().supports_soar(Namespace()))

    def test_predict_velocity_for_soar_wraps_auxiliary_call(self):
        trainer = Flux2NetworkTrainer()
        aux_latents = torch.randn(2, 4, 2, 2)
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
            batch={"ctx_vec": torch.zeros(2, 1, 1)},
            noisy_model_input=aux_latents,
            timesteps=timesteps,
            network_dtype=torch.float32,
        )

        self.assertTrue(torch.equal(model_pred, aux_latents + 1))
        self.assertTrue(torch.equal(calls["latents"], aux_latents))
        self.assertTrue(torch.equal(calls["noise"], torch.zeros_like(aux_latents)))
        self.assertTrue(torch.equal(calls["noisy_model_input"], aux_latents))
        self.assertTrue(torch.equal(calls["timesteps"], timesteps))
        self.assertEqual(calls["network_dtype"], torch.float32)


if __name__ == "__main__":
    unittest.main()

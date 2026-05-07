import unittest
from argparse import Namespace

import torch

from tests.test_support import import_zimage_train_module


zimage_train = import_zimage_train_module()
_compute_zimage_loss_weighting_from_sigma = zimage_train._compute_zimage_loss_weighting_from_sigma
_run_soar_auxiliary_pass = zimage_train._run_soar_auxiliary_pass


class DummyAccelerator:
    def __init__(self):
        self.backward_calls = []

    def backward(self, loss):
        self.backward_calls.append(float(loss.detach().item()))


class TestZImageSoarAux(unittest.TestCase):
    def test_auxiliary_pass_normalizes_lambda_by_total_count(self):
        accelerator = DummyAccelerator()
        args = Namespace(soar_lambda_aux=1.0, weighting_scheme="none")
        clean_latents = torch.ones(2, 1, 2, 2)

        call_count = []

        def fake_predict(aux_latents, aux_timesteps):
            call_count.append((aux_latents.shape, aux_timesteps.shape))
            return torch.zeros_like(aux_latents)

        aux_points = [
            {
                "latents": torch.zeros_like(clean_latents),
                "sigmas": torch.ones(2, dtype=torch.float32),
                "timesteps": torch.tensor([10.0, 20.0]),
            }
            for _ in range(3)
        ]

        loss_aux_sum, aux_count = _run_soar_auxiliary_pass(
            args=args,
            accelerator=accelerator,
            predict_velocity_fn=fake_predict,
            clean_latents=clean_latents,
            aux_points=aux_points,
            total_count=8.0,
        )

        self.assertEqual(len(call_count), 3)
        self.assertEqual(aux_count, 6.0)
        self.assertAlmostEqual(float(loss_aux_sum.item()), 6.0)
        self.assertEqual(len(accelerator.backward_calls), 3)
        for value in accelerator.backward_calls:
            self.assertAlmostEqual(value, 0.25)


class TestZImageLossWeighting(unittest.TestCase):
    def test_weighting_keeps_image_loss_rank(self):
        loss = torch.zeros(2, 4, 8, 8)
        sigmas = torch.tensor([0.25, 0.5], dtype=torch.float32).view(2, 1, 1, 1)

        for scheme in ("sigma_sqrt", "cosmap"):
            with self.subTest(scheme=scheme):
                weighting = _compute_zimage_loss_weighting_from_sigma(scheme, sigmas, loss.ndim)

                self.assertEqual(weighting.shape, torch.Size([2, 1, 1, 1]))
                self.assertEqual((loss * weighting).shape, loss.shape)


if __name__ == "__main__":
    unittest.main()

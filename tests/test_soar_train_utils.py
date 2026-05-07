import argparse
import unittest
from argparse import Namespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musubi_tuner.soar_train_utils import (  # noqa: E402
    add_soar_arguments,
    compute_loss_weighting_from_sigma,
    compute_per_sample_loss,
    default_flow_matching_target,
    is_soar_enabled,
    validate_soar_args,
    zimage_flow_matching_target,
)


class TestSoarTrainUtils(unittest.TestCase):
    def test_add_soar_arguments_is_idempotent(self):
        parser = argparse.ArgumentParser()
        add_soar_arguments(parser)
        add_soar_arguments(parser)

        args = parser.parse_args([])
        self.assertFalse(args.soar)
        self.assertEqual(args.soar_lambda_aux, 1.0)
        self.assertEqual(args.soar_trajectory_length, 6)
        self.assertEqual(args.soar_num_sampling_steps, 40)
        self.assertEqual(args.soar_sigma_upper_ratio, 1.5)

    def test_validate_soar_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=-1.0,
                    soar_trajectory_length=6,
                    soar_num_sampling_steps=40,
                    soar_sigma_upper_ratio=1.5,
                )
            )
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=1.0,
                    soar_trajectory_length=0,
                    soar_num_sampling_steps=40,
                    soar_sigma_upper_ratio=1.5,
                )
            )
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=1.0,
                    soar_trajectory_length=6,
                    soar_num_sampling_steps=1,
                    soar_sigma_upper_ratio=1.5,
                )
            )
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=1.0,
                    soar_trajectory_length=6,
                    soar_num_sampling_steps=40,
                    soar_sigma_upper_ratio=0.9,
                )
            )

    def test_is_soar_enabled_respects_zero_lambda(self):
        self.assertFalse(is_soar_enabled(Namespace(soar=True, soar_lambda_aux=0.0, soar_trajectory_length=6)))
        self.assertTrue(is_soar_enabled(Namespace(soar=True, soar_lambda_aux=1.0, soar_trajectory_length=6)))

    def test_loss_weighting_keeps_target_rank(self):
        loss = torch.zeros(2, 4, 8, 8)
        sigmas = torch.tensor([0.25, 0.5], dtype=torch.float32).view(2, 1, 1, 1, 1)

        weighting = compute_loss_weighting_from_sigma("sigma_sqrt", sigmas, loss.ndim)

        self.assertEqual(weighting.shape, torch.Size([2, 1, 1, 1]))
        self.assertEqual((loss * weighting).shape, loss.shape)

    def test_per_sample_loss_rejects_batch_broadcast(self):
        model_pred = torch.zeros(2, 1, 2, 2)
        target = torch.ones_like(model_pred)
        bad_weighting = torch.ones(2, 2, 1, 1)

        with self.assertRaises(ValueError):
            compute_per_sample_loss(model_pred, target, bad_weighting)

    def test_target_conventions_are_opposite(self):
        clean = torch.full((1, 1, 1, 1), 2.0)
        aux = torch.full_like(clean, 5.0)
        sigma = torch.full_like(clean, 0.5)

        self.assertTrue(torch.equal(default_flow_matching_target(clean, aux, sigma), torch.full_like(clean, 6.0)))
        self.assertTrue(torch.equal(zimage_flow_matching_target(clean, aux, sigma), torch.full_like(clean, -6.0)))


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musubi_tuner.soar_utils import (  # noqa: E402
    build_single_step_ode_aux_points,
    per_point_aux_scale,
    sigma_to_t,
    sigma_to_training_timestep,
    single_step_aux_points,
    t_to_sigma_timestep,
)


class DummyNoiseScheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        self.timesteps = (self.sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)


class TestSoarUtils(unittest.TestCase):
    def test_per_point_aux_scale_normalizes_lambda(self):
        self.assertAlmostEqual(per_point_aux_scale(1.0, 6), 1.0 / 6)
        self.assertAlmostEqual(per_point_aux_scale(0.5, 1), 0.5)

    def test_per_point_aux_scale_rejects_non_positive_trajectory_length(self):
        with self.assertRaises(ValueError):
            per_point_aux_scale(1.0, 0)

    def test_build_single_step_ode_aux_points_returns_requested_count_and_shapes(self):
        start_state = torch.zeros(2, 4, 2, 2)
        end_state = torch.ones(2, 4, 2, 2)
        sigma_start = torch.tensor([0.2, 0.3], dtype=torch.float32)
        sigma_end = torch.tensor([0.8, 0.9], dtype=torch.float32)

        points = build_single_step_ode_aux_points(
            start_state=start_state,
            end_state=end_state,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            trajectory_length=3,
            generator=torch.Generator().manual_seed(1234),
        )

        self.assertEqual(len(points), 3)
        for aux_state, aux_sigma in points:
            self.assertEqual(aux_state.shape, start_state.shape)
            self.assertEqual(aux_sigma.shape, sigma_start.shape)
            self.assertTrue(torch.all(aux_sigma >= sigma_start))
            self.assertTrue(torch.all(aux_sigma <= sigma_end))
            self.assertTrue(torch.all(aux_state >= start_state))
            self.assertTrue(torch.all(aux_state <= end_state))

    def test_build_single_step_ode_aux_points_supports_reversed_sigma_ranges(self):
        start_state = torch.full((1, 2, 2, 2), 2.0)
        end_state = torch.zeros(1, 2, 2, 2)
        sigma_start = torch.tensor([0.9], dtype=torch.float32)
        sigma_end = torch.tensor([0.4], dtype=torch.float32)

        points = build_single_step_ode_aux_points(
            start_state=start_state,
            end_state=end_state,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            trajectory_length=2,
            generator=torch.Generator().manual_seed(4321),
        )

        self.assertEqual(len(points), 2)
        sigma_lower = torch.minimum(sigma_start, sigma_end)
        sigma_upper = torch.maximum(sigma_start, sigma_end)
        state_lower = torch.minimum(start_state, end_state)
        state_upper = torch.maximum(start_state, end_state)

        for aux_state, aux_sigma in points:
            self.assertTrue(torch.all(aux_sigma >= sigma_lower))
            self.assertTrue(torch.all(aux_sigma <= sigma_upper))
            self.assertTrue(torch.all(aux_state >= state_lower))
            self.assertTrue(torch.all(aux_state <= state_upper))

    def test_t_to_sigma_timestep_and_sigma_to_t_round_trip(self):
        scheduler = DummyNoiseScheduler()
        t = torch.tensor([1.0, 0.5, 0.1], dtype=torch.float32)
        sigma, timesteps = t_to_sigma_timestep(t, scheduler)

        self.assertEqual(sigma.shape, t.shape)
        self.assertEqual(timesteps.shape, t.shape)

        reconstructed_t = sigma_to_t(sigma, scheduler)
        self.assertEqual(reconstructed_t.shape, t.shape)
        self.assertTrue(torch.all(reconstructed_t <= 1.0))
        self.assertTrue(torch.all(reconstructed_t >= 0.0))

    def test_single_step_aux_points_returns_expected_shape(self):
        scheduler = DummyNoiseScheduler()
        z_t0 = torch.zeros(2, 4, 2, 2)
        v_standard = torch.ones_like(z_t0) * 0.5
        z_noise = torch.ones_like(z_t0)
        t0 = torch.tensor([0.8, 0.6], dtype=torch.float32)

        points = single_step_aux_points(
            z_t0=z_t0,
            sigma_t0=t0,
            v_standard=v_standard,
            z_noise=z_noise,
            points_per_path=2,
            noise_scheduler=scheduler,
            num_sampling_steps=40,
        )

        self.assertEqual(len(points), 2)
        for point in points:
            self.assertEqual(point["latents"].shape, z_t0.shape)
            self.assertEqual(point["sigmas"].shape, torch.Size([2]))
            self.assertEqual(point["timesteps"].shape, torch.Size([2]))
            self.assertTrue(torch.allclose(point["timesteps"], sigma_to_training_timestep(point["sigmas"], scheduler)))

    def test_single_step_aux_points_keeps_scheduler_math_in_float32(self):
        scheduler = DummyNoiseScheduler()
        z_t0 = torch.zeros(2, 4, 2, 2, dtype=torch.bfloat16)
        v_standard = torch.ones_like(z_t0) * 0.5
        z_noise = torch.ones_like(z_t0)
        t0 = torch.tensor([0.8, 0.6], dtype=torch.bfloat16)

        points = single_step_aux_points(
            z_t0=z_t0,
            sigma_t0=t0,
            v_standard=v_standard,
            z_noise=z_noise,
            points_per_path=1,
            noise_scheduler=scheduler,
            num_sampling_steps=40,
        )

        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertEqual(point["latents"].dtype, z_t0.dtype)
        self.assertEqual(point["sigmas"].dtype, torch.float32)
        self.assertEqual(point["timesteps"].dtype, torch.float32)

    def test_single_step_aux_points_can_emit_discrete_scheduler_timesteps(self):
        scheduler = DummyNoiseScheduler()
        z_t0 = torch.zeros(1, 4, 2, 2)
        v_standard = torch.ones_like(z_t0) * 0.5
        z_noise = torch.ones_like(z_t0)

        points = single_step_aux_points(
            z_t0=z_t0,
            sigma_t0=torch.tensor([0.5], dtype=torch.float32),
            v_standard=v_standard,
            z_noise=z_noise,
            points_per_path=4,
            noise_scheduler=scheduler,
            num_sampling_steps=40,
            continuous_timesteps=False,
        )

        schedule_timesteps = scheduler.timesteps.to(dtype=torch.float32)
        for point in points:
            self.assertTrue(torch.all(torch.isin(point["timesteps"], schedule_timesteps)))

    def test_single_step_aux_points_continuous_timesteps_match_training_offset(self):
        scheduler = DummyNoiseScheduler()
        z_t0 = torch.zeros(1, 4, 2, 2)
        v_standard = torch.ones_like(z_t0) * 0.5
        z_noise = torch.ones_like(z_t0)

        points = single_step_aux_points(
            z_t0=z_t0,
            sigma_t0=torch.tensor([0.5], dtype=torch.float32),
            v_standard=v_standard,
            z_noise=z_noise,
            points_per_path=4,
            noise_scheduler=scheduler,
            num_sampling_steps=40,
            continuous_timesteps=True,
        )

        for point in points:
            self.assertTrue(torch.allclose(point["timesteps"], sigma_to_training_timestep(point["sigmas"], scheduler)))

    def test_single_step_aux_points_bounds_default_sigma_upper_by_ratio(self):
        scheduler = DummyNoiseScheduler()
        z_t0 = torch.zeros(1, 4, 2, 2)
        v_standard = torch.ones_like(z_t0) * 0.5
        z_noise = torch.ones_like(z_t0)
        sigma_t0 = torch.tensor([0.2], dtype=torch.float32)

        torch.manual_seed(1234)
        points = single_step_aux_points(
            z_t0=z_t0,
            sigma_t0=sigma_t0,
            v_standard=v_standard,
            z_noise=z_noise,
            points_per_path=8,
            noise_scheduler=scheduler,
            num_sampling_steps=40,
        )

        for point in points:
            self.assertTrue(torch.all(point["sigmas"] <= sigma_t0 * 1.5))

    def test_single_step_aux_points_interpolates_toward_noise_at_sigma_one(self):
        scheduler = DummyNoiseScheduler()
        z_t0 = torch.zeros(1, 1, 1, 1)
        v_standard = torch.zeros_like(z_t0)
        z_noise = torch.ones_like(z_t0)
        sigma_t0 = torch.tensor([0.2], dtype=torch.float32)

        points = single_step_aux_points(
            z_t0=z_t0,
            sigma_t0=sigma_t0,
            v_standard=v_standard,
            z_noise=z_noise,
            points_per_path=1,
            noise_scheduler=scheduler,
            num_sampling_steps=40,
            sigma_upper=torch.tensor([0.3], dtype=torch.float32),
        )

        aux_sigma = points[0]["sigmas"].view(1, 1, 1, 1)
        sigma_t1 = (sigma_t0 - 1.0 / 40.0).view(1, 1, 1, 1)
        expected_alpha = (aux_sigma - sigma_t1) / (1.0 - sigma_t1)
        self.assertTrue(torch.allclose(points[0]["latents"], expected_alpha))
        self.assertTrue(torch.all(points[0]["latents"] < 0.2))


if __name__ == "__main__":
    unittest.main()

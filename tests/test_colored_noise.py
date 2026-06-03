from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musubi_tuner.modules.colored_noise import (
    ColoredNoiseShaper,
    add_colored_noise_args,
    build_colored_noise_shaper_from_args,
    parse_alpha_tilting,
    shape_packed_2x2_noise,
    validate_colored_noise_args,
)
from musubi_tuner.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler


def test_colored_noise_shapes_4d_noise_to_unit_std() -> None:
    shaper = ColoredNoiseShaper(gamma_matrix=torch.zeros(4, 8))
    noise = torch.randn(2, 3, 16, 16)

    shaped = shaper.shape(noise, step_index=1, total_steps=4)

    assert shaped.shape == noise.shape
    assert shaped.dtype == noise.dtype
    assert torch.isfinite(shaped).all()
    assert shaped.std().item() == pytest.approx(1.0, rel=0.05)


def test_colored_noise_shapes_5d_video_spatial_axes() -> None:
    gamma = torch.linspace(0.0, 0.8, steps=12).repeat(5, 1)
    shaper = ColoredNoiseShaper(gamma_matrix=gamma, gamma_matrix_divider=2.0, sqrt_gamma=True)
    noise = torch.randn(1, 4, 3, 16, 16, dtype=torch.float32)

    shaped = shaper.shape(noise, step_index=2, total_steps=5)

    assert shaped.shape == noise.shape
    assert shaped.dtype == noise.dtype
    assert torch.isfinite(shaped).all()
    assert shaped.std().item() == pytest.approx(1.0, rel=0.05)


def test_colored_noise_shapes_non_trailing_spatial_axes() -> None:
    shaper = ColoredNoiseShaper(gamma_matrix=torch.zeros(3, 8))
    noise = torch.randn(2, 16, 16, 4, dtype=torch.float32)

    shaped = shaper.shape(noise, step_index=0, total_steps=3, spatial_dims=(1, 2))

    assert shaped.shape == noise.shape
    assert shaped.dtype == noise.dtype
    assert torch.isfinite(shaped).all()
    assert shaped.std().item() == pytest.approx(1.0, rel=0.05)


def test_packed_2x2_noise_uses_same_layout_as_manual_unpack_repack() -> None:
    shaper = ColoredNoiseShaper(gamma_matrix=torch.linspace(0.0, 0.5, steps=10).repeat(2, 1))
    packed_height, packed_width, layers, channels = 4, 5, 2, 3
    packed = torch.randn(1, layers * packed_height * packed_width, channels * 4)

    shaped = shape_packed_2x2_noise(
        packed,
        shaper,
        packed_height=packed_height,
        packed_width=packed_width,
        layers=layers,
        step_index=0,
        total_steps=2,
    )

    manual = packed.reshape(1, layers, packed_height, packed_width, channels, 2, 2)
    manual = manual.permute(0, 1, 4, 2, 5, 3, 6).reshape(1, layers, channels, packed_height * 2, packed_width * 2)
    manual = shaper.shape(manual, step_index=0, total_steps=2)
    manual = manual.reshape(1, layers, channels, packed_height, 2, packed_width, 2)
    manual = manual.permute(0, 1, 3, 5, 2, 4, 6).reshape_as(packed)

    assert shaped.shape == packed.shape
    assert torch.allclose(shaped, manual)


def test_colored_noise_arg_helpers_validate_and_cache(tmp_path: Path) -> None:
    gamma_path = tmp_path / "gamma.pt"
    torch.save(torch.zeros(2, 8), gamma_path)
    parser = argparse.ArgumentParser()
    add_colored_noise_args(parser)
    args = parser.parse_args(["--cns", "--cns_gamma_matrix_path", str(gamma_path)])

    validate_colored_noise_args(args, parser)
    shaper = build_colored_noise_shaper_from_args(args)

    assert shaper is build_colored_noise_shaper_from_args(args)


def test_alpha_tilting_parser() -> None:
    assert parse_alpha_tilting(None) == 0.0
    assert parse_alpha_tilting([]) == 0.0
    assert parse_alpha_tilting([0.25]) == 0.25
    assert parse_alpha_tilting([0.15, -0.5]) == (0.15, -0.5)
    with pytest.raises(ValueError):
        parse_alpha_tilting([0.0, 0.1, 0.2])


def test_dpm_sde_scheduler_accepts_colored_noise_for_5d_latents() -> None:
    scheduler = FlowDPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", solver_order=1)
    scheduler.set_timesteps(3)
    scheduler.set_colored_noise_shaper(ColoredNoiseShaper(gamma_matrix=torch.zeros(3, 8)))

    sample = torch.randn(1, 4, 2, 8, 8)
    model_output = torch.randn_like(sample)
    prev_sample = scheduler.step(
        model_output,
        scheduler.timesteps[0],
        sample,
        generator=torch.Generator().manual_seed(0),
        return_dict=False,
    )[0]

    assert prev_sample.shape == sample.shape
    assert torch.isfinite(prev_sample).all()

"""Reproducible effective controls and base-relative enhancement supervision."""

import math

import torch

from musubi_tuner.dlssnr.controls import fixed_control_tensor, resolve_fixed_controls
from musubi_tuner.dlssnr.temporal import augmentation_seed


def encode_control_point(reference: dict, ratios: tuple[float, float]) -> dict[str, torch.Tensor]:
    if len(ratios) != 2 or any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in ratios
    ):
        raise ValueError("control ratios must be two finite numbers in [0,1]")
    reference = resolve_fixed_controls(reference)
    reference_lanes = fixed_control_tensor(reference, 1, 1)[:, 0, 0].clone()
    indices = [1, 4 if reference["nr_auto_mask"] else 2]
    reference_values = reference_lanes[indices]
    if not (reference_values > 0).all():
        raise ValueError("reference tone and structure must be positive after FP16 encoding")
    sampled = {**reference, "nr_tone": reference["nr_tone"] * ratios[0], "nr_structure": reference["nr_structure"] * ratios[1]}
    sampled_lanes = fixed_control_tensor(sampled, 1, 1)[:, 0, 0].clone()
    values = sampled_lanes[indices]
    return {
        "reference_lanes": reference_lanes,
        "sampled_lanes": sampled_lanes,
        "ratios": values / reference_values,
        "values": values,
    }


def validate_control_settings(settings: dict) -> None:
    for name in ("anchor_probability", "corner_probability"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"control_{name} must be finite and in [0,1]")
    if settings["anchor_probability"] + settings["corner_probability"] > 1:
        raise ValueError("control anchor and corner probabilities must sum to at most one")
    sigma = settings["residual_sigma"]
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or not 0 < sigma <= 32:
        raise ValueError("control_residual_sigma must be finite and satisfy 0 < sigma <= 32")


def sample_control_point(
    reference: dict, settings: dict, *, seed: int, epoch: int, sample_id: str, crop_id: int
) -> dict[str, torch.Tensor]:
    validate_control_settings(settings)
    generator = torch.Generator(device="cpu").manual_seed(augmentation_seed(seed, epoch, sample_id, crop_id, domain="controls"))
    branch = float(torch.rand((), generator=generator))
    if branch < settings["anchor_probability"]:
        ratios = (1, 1)
    elif branch < settings["anchor_probability"] + settings["corner_probability"]:
        corner = int(torch.randint(4, (), generator=generator))
        ratios = (corner // 2, corner % 2)
    else:
        ratios = tuple(torch.rand(2, generator=generator).tolist())
    return encode_control_point(reference, ratios)

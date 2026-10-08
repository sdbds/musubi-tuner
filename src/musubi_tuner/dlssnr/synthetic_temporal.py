"""Lazy subpixel translations of paired stills with explicit geometric support."""

import math

import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.temporal import augmentation_seed


def validate_max_shift(max_shift_px: float) -> None:
    if type(max_shift_px) not in (int, float) or not math.isfinite(max_shift_px) or not 0 <= max_shift_px <= 1:
        raise ValueError("synthetic_max_shift_px must be finite and in [0,1]")


def sample_offsets(length: int, max_shift_px: float, *, seed: int, epoch: int, sample_id: str, crop_id: int) -> torch.Tensor:
    if type(length) is not int or length < 1:
        raise ValueError("synthetic clip length must be a positive integer")
    validate_max_shift(max_shift_px)
    generator = torch.Generator(device="cpu").manual_seed(
        augmentation_seed(seed, epoch, sample_id, crop_id, domain="synthetic_jitter")
    )
    offsets = torch.zeros(length, 2, dtype=torch.float32)
    offsets[1:] = (torch.rand(length - 1, 2, generator=generator) * 2 - 1) * max_shift_px
    return offsets


def validate_synthetic_support(sample: dict, max_shift_px: float) -> None:
    validate_max_shift(max_shift_px)
    mask = sample["loss_mask"]
    margin = math.ceil(max_shift_px)
    interior = mask[..., margin:-margin, margin:-margin] if margin else mask
    if not (interior > 0).any():
        raise ValueError(f"synthetic sample {sample.get('sample_id', '')} has no safe label support at shift {max_shift_px}")


def _grid(offset, height, width):
    x = torch.arange(width, dtype=torch.float32, device=offset.device) + offset[0]
    y = torch.arange(height, dtype=torch.float32, device=offset.device) + offset[1]
    xx, yy = x[None, :].expand(height, width), y[:, None].expand(height, width)
    grid = torch.stack((2 * (xx + 0.5) / width - 1, 2 * (yy + 0.5) / height - 1), dim=-1)[None]
    support = ((xx >= 0) & (xx <= width - 1) & (yy >= 0) & (yy <= height - 1))[None]
    return grid, support


def _sample(value, grid, *, padding="border"):
    return F.grid_sample(value[None], grid, mode="bilinear", padding_mode=padding, align_corners=False)[0]


def _previous_footprint(previous_support, motion):
    height, width = previous_support.shape[-2:]
    x = torch.arange(width, dtype=torch.float32, device=motion.device) + motion[0]
    y = torch.arange(height, dtype=torch.float32, device=motion.device) + motion[1]
    inside = ((x >= 0) & (x <= width - 1))[None, :] & ((y >= 0) & (y <= height - 1))[:, None]
    # Floor/ceil collapse at integer centers, so zero-weight neighbors do not
    # invalidate an exact sample. Every contributing previous pixel must exist.
    for xx in (x.floor().long(), x.ceil().long()):
        for yy in (y.floor().long(), y.ceil().long()):
            inside = inside & previous_support[0, yy.clamp(0, height - 1)[:, None], xx.clamp(0, width - 1)[None, :]]
    return inside[None]


@torch.no_grad()
def synthesize_clip(sample: dict, offsets: torch.Tensor) -> dict:
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 2
        or offsets.shape[0] < 1
        or not torch.isfinite(offsets).all()
        or offsets.abs().max() > 1
    ):
        raise ValueError("synthetic offsets must be finite [T,2] values in [-1,1]")
    if offsets[0].ne(0).any():
        raise ValueError("the first synthetic offset must be zero")
    source, target, controls, mask = (sample[name] for name in ("source", "target", "controls", "loss_mask"))
    height, width = source.shape[-2:]
    offsets = offsets.to(device=source.device, dtype=torch.float32)
    frames, previous_support = [], None
    with torch.autocast(source.device.type, enabled=False):
        for index, offset in enumerate(offsets):
            grid, support = _grid(offset, height, width)
            stationary = bool(offset.eq(0).all())
            translated_source = source if stationary else _sample(source, grid).clamp(0, 1)
            translated_controls = controls if stationary or "fixed_controls" in sample else _sample(controls, grid)
            coverage = mask if stationary else _sample(mask, grid, padding="zeros").clamp(0, 1)
            translated_mask = coverage * support
            if index == 0:
                translated_target = target
            else:
                weighted = torch.where(mask > 0, target, 0) * mask
                numerator = weighted if stationary else _sample(weighted, grid, padding="zeros")
                label = target if stationary else numerator / torch.where(coverage > 0, coverage, 1)
                translated_target = torch.where(translated_mask > 0, label, 0).clamp(0, 1)
            delta = torch.zeros_like(offset) if index == 0 else offset - offsets[index - 1]
            history = torch.zeros_like(support) if index == 0 else support & _previous_footprint(previous_support, delta)
            frames.append(
                {
                    "source": translated_source,
                    "target": translated_target,
                    "controls": translated_controls,
                    "loss_mask": translated_mask,
                    "motion": delta[:, None, None].expand(2, height, width),
                    "history_valid": history,
                    "temporal_valid": history.clone(),
                    "reset": index == 0,
                    "frame_index": sample["frame_index"] + index,
                }
            )
            previous_support = support
    metadata = {name: value for name, value in sample.items() if name not in frames[0]}
    return {**metadata, "frames": frames, "temporal_support": "joint_loss_mask"}

"""Colored Noise Sampling utilities."""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import torch


logger = logging.getLogger(__name__)

AlphaTilting = Union[float, Tuple[float, float]]


def add_colored_noise_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Colored Noise Sampling")
    group.add_argument(
        "--cns",
        action="store_true",
        help="Enable Colored Noise Sampling for initial sampling noise. SDE samplers may also color per-step noise.",
    )
    group.add_argument("--cns_gamma_matrix_path", type=str, default=None, help="Path to a CNS gamma matrix .pt file.")
    group.add_argument(
        "--cns_gamma_matrix_divider", type=float, default=1.0, help="Divider applied to gamma rows before CNS residuals."
    )
    group.add_argument("--cns_sqrt_gamma", action="store_true", help="Use sqrt residual energy for CNS frequency scaling.")
    group.add_argument("--cns_power_gamma", type=float, default=1.0, help="Power applied to CNS residual energy.")
    group.add_argument(
        "--cns_alpha_tilting",
        type=float,
        nargs="*",
        default=None,
        help="CNS alpha tilt. Pass one value or two values for start/end interpolation.",
    )
    group.add_argument("--cns_alpha_tilting_inside_exp", action="store_true", help="Apply CNS alpha tilt inside the exponent.")
    group.add_argument(
        "--cns_alpha_tilting_use_fnorm", action="store_true", help="Use normalized radial frequency for CNS alpha tilt."
    )
    group.add_argument(
        "--cns_alpha_exponential_interpolation",
        action="store_true",
        help="Use exponential interpolation for two-value CNS alpha tilt.",
    )
    group.add_argument(
        "--cns_alpha_exponential_interpolation_sharpness",
        type=float,
        default=4.0,
        help="Sharpness for CNS alpha exponential interpolation.",
    )
    group.add_argument("--cns_energy_scale", type=float, default=1.0, help="Scale shaped CNS noise after unit-std normalization.")


def parse_alpha_tilting(values: Optional[Sequence[float]]) -> AlphaTilting:
    if values is None or len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return float(values[0]), float(values[1])
    raise ValueError("--cns_alpha_tilting accepts zero, one, or two float values")


def colored_noise_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "cns", False))


def validate_colored_noise_args(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None) -> None:
    if not colored_noise_enabled(args):
        return

    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    gamma_matrix_path = getattr(args, "cns_gamma_matrix_path", None)
    if not gamma_matrix_path:
        fail("--cns requires --cns_gamma_matrix_path")
    if not os.path.isfile(gamma_matrix_path):
        fail(f"--cns_gamma_matrix_path does not exist: {gamma_matrix_path}")

    try:
        parse_alpha_tilting(getattr(args, "cns_alpha_tilting", None))
    except ValueError as exc:
        fail(str(exc))

    if getattr(args, "cns_gamma_matrix_divider", 1.0) <= 0:
        fail("--cns_gamma_matrix_divider must be positive")
    if getattr(args, "cns_power_gamma", 1.0) <= 0:
        fail("--cns_power_gamma must be positive")
    if getattr(args, "cns_alpha_exponential_interpolation_sharpness", 4.0) <= 0:
        fail("--cns_alpha_exponential_interpolation_sharpness must be positive")
    if getattr(args, "cns_energy_scale", 1.0) <= 0:
        fail("--cns_energy_scale must be positive")


def build_colored_noise_shaper_from_args(args: argparse.Namespace) -> Optional["ColoredNoiseShaper"]:
    if not colored_noise_enabled(args):
        return None

    cached = getattr(args, "_cns_colored_noise_shaper", None)
    if cached is not None:
        return cached

    shaper = ColoredNoiseShaper.from_path(
        getattr(args, "cns_gamma_matrix_path"),
        gamma_matrix_divider=getattr(args, "cns_gamma_matrix_divider", 1.0),
        sqrt_gamma=getattr(args, "cns_sqrt_gamma", False),
        power_gamma=getattr(args, "cns_power_gamma", 1.0),
        alpha_tilting=parse_alpha_tilting(getattr(args, "cns_alpha_tilting", None)),
        alpha_tilting_inside_exp=getattr(args, "cns_alpha_tilting_inside_exp", False),
        alpha_tilting_use_fnorm=getattr(args, "cns_alpha_tilting_use_fnorm", False),
        alpha_exponential_interpolation=getattr(args, "cns_alpha_exponential_interpolation", False),
        alpha_exponential_interpolation_sharpness=getattr(args, "cns_alpha_exponential_interpolation_sharpness", 4.0),
        energy_scale=getattr(args, "cns_energy_scale", 1.0),
    )
    setattr(args, "_cns_colored_noise_shaper", shaper)
    return shaper


def apply_colored_noise_from_args(
    args: argparse.Namespace,
    noise: torch.Tensor,
    *,
    step_index: int = 0,
    total_steps: Optional[int] = None,
    spatial_dims: Tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    shaper = build_colored_noise_shaper_from_args(args)
    if shaper is None:
        return noise
    return shaper.shape(noise, step_index, total_steps, spatial_dims=spatial_dims)


def apply_colored_noise_to_packed_2x2_from_args(
    args: argparse.Namespace,
    noise: torch.Tensor,
    *,
    packed_height: int,
    packed_width: int,
    layers: Optional[int] = None,
    step_index: int = 0,
    total_steps: Optional[int] = None,
) -> torch.Tensor:
    shaper = build_colored_noise_shaper_from_args(args)
    if shaper is None:
        return noise
    return shape_packed_2x2_noise(
        noise,
        shaper,
        packed_height=packed_height,
        packed_width=packed_width,
        layers=layers,
        step_index=step_index,
        total_steps=total_steps,
    )


def shape_packed_2x2_noise(
    noise: torch.Tensor,
    shaper: "ColoredNoiseShaper",
    *,
    packed_height: int,
    packed_width: int,
    layers: Optional[int] = None,
    step_index: int = 0,
    total_steps: Optional[int] = None,
) -> torch.Tensor:
    if noise.ndim != 3:
        raise ValueError(f"Packed 2x2 CNS noise must be 3D [B, tokens, C*4], got shape {tuple(noise.shape)}")
    if noise.shape[2] % 4 != 0:
        raise ValueError(f"Packed 2x2 CNS channel dimension must be divisible by 4, got {noise.shape[2]}")

    tokens_per_layer = packed_height * packed_width
    if tokens_per_layer <= 0:
        raise ValueError("packed_height and packed_width must be positive")
    if layers is None:
        if noise.shape[1] % tokens_per_layer != 0:
            raise ValueError(
                f"Packed 2x2 CNS token count {noise.shape[1]} is not divisible by packed grid {packed_height}x{packed_width}"
            )
        layers = noise.shape[1] // tokens_per_layer
    elif layers <= 0:
        raise ValueError("layers must be positive")

    if noise.shape[1] != layers * tokens_per_layer:
        raise ValueError(
            f"Packed 2x2 CNS expected {layers * tokens_per_layer} tokens, got {noise.shape[1]}"
        )

    original_dtype = noise.dtype
    batch_size, _, packed_channels = noise.shape
    channels = packed_channels // 4
    unpacked = noise.to(torch.float32).reshape(batch_size, layers, packed_height, packed_width, channels, 2, 2)
    unpacked = unpacked.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch_size, layers, channels, packed_height * 2, packed_width * 2)
    shaped = shaper.shape(unpacked, step_index, total_steps, spatial_dims=(-2, -1))
    repacked = shaped.reshape(batch_size, layers, channels, packed_height, 2, packed_width, 2)
    repacked = repacked.permute(0, 1, 3, 5, 2, 4, 6).reshape(batch_size, layers * tokens_per_layer, packed_channels)
    return repacked.to(original_dtype)


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_gamma_matrix(path: str) -> torch.Tensor:
    if not path:
        raise ValueError("A gamma matrix path is required for colored noise sampling")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Gamma matrix not found: {path}")

    payload = _torch_load_cpu(path)
    if isinstance(payload, torch.Tensor):
        gamma_matrix = payload
    elif isinstance(payload, dict):
        gamma_matrix = None
        for key in ("gamma_matrix", "gamma", "matrix"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                gamma_matrix = value
                break
        if gamma_matrix is None:
            gamma_matrix = next((value for value in payload.values() if isinstance(value, torch.Tensor)), None)
        if gamma_matrix is None:
            raise ValueError(f"No tensor gamma matrix found in {path}")
    else:
        raise ValueError(f"Unsupported gamma matrix payload type: {type(payload)!r}")

    gamma_matrix = gamma_matrix.detach().to(device="cpu", dtype=torch.float32)
    if gamma_matrix.ndim != 2:
        raise ValueError(f"Gamma matrix must be 2D [steps, frequency_bins], got shape {tuple(gamma_matrix.shape)}")
    if gamma_matrix.shape[0] < 1 or gamma_matrix.shape[1] < 1:
        raise ValueError(f"Gamma matrix must not be empty, got shape {tuple(gamma_matrix.shape)}")
    if not torch.isfinite(gamma_matrix).all():
        raise ValueError("Gamma matrix contains non-finite values")
    return gamma_matrix


@dataclass
class ColoredNoiseShaper:
    gamma_matrix: torch.Tensor
    gamma_matrix_divider: float = 1.0
    sqrt_gamma: bool = False
    power_gamma: float = 1.0
    alpha_tilting: AlphaTilting = 0.0
    alpha_tilting_inside_exp: bool = False
    alpha_tilting_use_fnorm: bool = False
    alpha_exponential_interpolation: bool = False
    alpha_exponential_interpolation_sharpness: float = 4.0
    energy_scale: float = 1.0
    _freq_index_cache: Dict[Tuple[str, int, int, int], torch.Tensor] = field(default_factory=dict, init=False)

    @classmethod
    def from_path(
        cls,
        gamma_matrix_path: str,
        *,
        gamma_matrix_divider: float = 1.0,
        sqrt_gamma: bool = False,
        power_gamma: float = 1.0,
        alpha_tilting: AlphaTilting = 0.0,
        alpha_tilting_inside_exp: bool = False,
        alpha_tilting_use_fnorm: bool = False,
        alpha_exponential_interpolation: bool = False,
        alpha_exponential_interpolation_sharpness: float = 4.0,
        energy_scale: float = 1.0,
    ) -> "ColoredNoiseShaper":
        return cls(
            gamma_matrix=load_gamma_matrix(gamma_matrix_path),
            gamma_matrix_divider=gamma_matrix_divider,
            sqrt_gamma=sqrt_gamma,
            power_gamma=power_gamma,
            alpha_tilting=alpha_tilting,
            alpha_tilting_inside_exp=alpha_tilting_inside_exp,
            alpha_tilting_use_fnorm=alpha_tilting_use_fnorm,
            alpha_exponential_interpolation=alpha_exponential_interpolation,
            alpha_exponential_interpolation_sharpness=alpha_exponential_interpolation_sharpness,
            energy_scale=energy_scale,
        )

    def __post_init__(self) -> None:
        self.gamma_matrix = self.gamma_matrix.detach().to(device="cpu", dtype=torch.float32)
        if self.gamma_matrix.ndim != 2:
            raise ValueError(f"Gamma matrix must be 2D [steps, frequency_bins], got shape {tuple(self.gamma_matrix.shape)}")
        if self.gamma_matrix_divider <= 0:
            raise ValueError("gamma_matrix_divider must be positive")
        if self.power_gamma <= 0:
            raise ValueError("power_gamma must be positive")
        if self.alpha_exponential_interpolation_sharpness <= 0:
            raise ValueError("alpha_exponential_interpolation_sharpness must be positive")
        if self.energy_scale <= 0:
            raise ValueError("energy_scale must be positive")
        if self._alpha_is_nonzero() and not self.alpha_tilting_use_fnorm and not self.alpha_tilting_inside_exp:
            raise ValueError(
                "Non-zero alpha_tilting requires alpha_tilting_use_fnorm or alpha_tilting_inside_exp so the tilt is defined"
            )

    def shape(
        self,
        noise: torch.Tensor,
        step_index: int,
        total_steps: Optional[int] = None,
        *,
        spatial_dims: Tuple[int, int] = (-2, -1),
    ) -> torch.Tensor:
        if noise.ndim < 4:
            return noise

        spatial_dims = _normalize_spatial_dims(noise.ndim, spatial_dims)
        last_spatial_dims = (noise.ndim - 2, noise.ndim - 1)
        if spatial_dims != last_spatial_dims:
            permute_order = [dim for dim in range(noise.ndim) if dim not in spatial_dims] + list(spatial_dims)
            inverse_order = [0] * noise.ndim
            for new_dim, old_dim in enumerate(permute_order):
                inverse_order[old_dim] = new_dim

            moved = noise.permute(permute_order).contiguous()
            shaped = self.shape(moved, step_index, total_steps, spatial_dims=(-2, -1))
            return shaped.permute(inverse_order).contiguous()

        original_dtype = noise.dtype
        work = noise.to(torch.float32)
        gamma_row = self._gamma_row(step_index, total_steps, work.device)
        num_freq_bins = gamma_row.shape[0]

        noise_scaling = self._noise_scaling(gamma_row, step_index, total_steps)
        freq_indices = self._freq_indices(work.shape[-2], work.shape[-1], num_freq_bins, work.device)
        scaling_grid = noise_scaling[freq_indices]
        for _ in range(work.ndim - 2):
            scaling_grid = scaling_grid.unsqueeze(0)

        filtered = torch.fft.ifft2(torch.fft.fft2(work, dim=(-2, -1)) * scaling_grid, dim=(-2, -1)).real
        filtered_std = filtered.std()
        if torch.isfinite(filtered_std) and filtered_std > 1e-9:
            filtered = filtered / filtered_std
        else:
            logger.warning("Colored noise std was too small or non-finite; returning unshaped white noise")
            return noise

        if self.energy_scale != 1.0:
            filtered = filtered * self.energy_scale
        return filtered.to(original_dtype)

    def _alpha_is_nonzero(self) -> bool:
        if isinstance(self.alpha_tilting, tuple):
            return self.alpha_tilting[0] != 0.0 or self.alpha_tilting[1] != 0.0
        return float(self.alpha_tilting) != 0.0

    def _gamma_row(self, step_index: int, total_steps: Optional[int], device: torch.device) -> torch.Tensor:
        rows = self.gamma_matrix.shape[0]
        if rows == 1:
            matrix_index = 0
        elif total_steps is None or total_steps <= 1:
            matrix_index = min(max(step_index, 0), rows - 1)
        elif rows == total_steps:
            matrix_index = min(max(step_index, 0), rows - 1)
        else:
            progress = min(max(step_index / max(1, total_steps - 1), 0.0), 1.0)
            matrix_index = int(round(progress * (rows - 1)))
        return self.gamma_matrix[matrix_index].to(device=device)

    def _current_alpha(self, step_index: int, total_steps: Optional[int]) -> float:
        if not isinstance(self.alpha_tilting, tuple):
            return float(self.alpha_tilting)

        alpha_start, alpha_end = self.alpha_tilting
        progress = 0.0
        if total_steps is not None and total_steps > 1:
            progress = min(max(step_index / max(1, total_steps - 1), 0.0), 1.0)

        if self.alpha_exponential_interpolation:
            sharpness = self.alpha_exponential_interpolation_sharpness
            progress = (math.exp(sharpness * progress) - 1.0) / (math.exp(sharpness) - 1.0)
        return alpha_start + progress * (alpha_end - alpha_start)

    def _noise_scaling(self, gamma_row: torch.Tensor, step_index: int, total_steps: Optional[int]) -> torch.Tensor:
        current_alpha = self._current_alpha(step_index, total_steps)
        f_norm = torch.linspace(0.0, 1.0, steps=gamma_row.shape[0], device=gamma_row.device, dtype=gamma_row.dtype)
        base_residual = 1.0 - gamma_row / self.gamma_matrix_divider

        if current_alpha != 0.0:
            tilt_base: Union[float, torch.Tensor]
            if self.alpha_tilting_use_fnorm:
                tilt_base = f_norm
            else:
                tilt_base = 1.0

            if self.alpha_tilting_inside_exp:
                residual_energy = torch.exp(current_alpha * tilt_base * base_residual)
            else:
                residual_energy = torch.exp(current_alpha * tilt_base) * base_residual
        else:
            residual_energy = base_residual.clamp(min=0.0)

        if self.sqrt_gamma:
            noise_scaling = torch.sqrt(residual_energy.clamp(min=0.0))
        elif self.power_gamma != 1.0:
            noise_scaling = residual_energy.clamp(min=0.0) ** self.power_gamma
        else:
            noise_scaling = residual_energy.clamp(min=0.0)

        return torch.nan_to_num(noise_scaling, nan=0.0, posinf=1.0e6, neginf=0.0)

    def _freq_indices(self, height: int, width: int, num_freq_bins: int, device: torch.device) -> torch.Tensor:
        cache_key = (str(device), height, width, num_freq_bins)
        cached = self._freq_index_cache.get(cache_key)
        if cached is not None:
            return cached

        freq_y = torch.fft.fftfreq(height, device=device).view(height, 1)
        freq_x = torch.fft.fftfreq(width, device=device).view(1, width)
        radius = torch.sqrt(freq_x**2 + freq_y**2)
        radius_max = radius.max().clamp_min(1e-12)
        radius_norm = radius / radius_max
        freq_indices = (radius_norm * (num_freq_bins - 1)).long().clamp(0, num_freq_bins - 1)
        self._freq_index_cache[cache_key] = freq_indices
        return freq_indices


def _normalize_spatial_dims(ndim: int, spatial_dims: Tuple[int, int]) -> Tuple[int, int]:
    if len(spatial_dims) != 2:
        raise ValueError("spatial_dims must contain exactly two dimensions")

    normalized = []
    for dim in spatial_dims:
        if dim < 0:
            dim = ndim + dim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"spatial dimension {dim} is out of bounds for tensor with {ndim} dimensions")
        normalized.append(dim)
    if normalized[0] == normalized[1]:
        raise ValueError("spatial_dims must refer to two distinct dimensions")
    return normalized[0], normalized[1]

from __future__ import annotations

import argparse
import math
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file, save_file


SoarTargetFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
SoarPredictFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

SOAR_CFG_SCALE_SAMPLING_DEFAULT = 4.5
SOAR_EMPTY_PROMPT_CACHE_PREFIX = "__soar_empty_prompt"
SOAR_EMPTY_PROMPT_CACHE_DIR = "__soar_empty_prompt"

CONTINUOUS_TIMESTEP_SAMPLINGS = {
    "uniform",
    "sigmoid",
    "shift",
    "flux_shift",
    "qwen_shift",
    "logsnr",
    "qinglong_flux",
    "qinglong_qwen",
    "flux2_shift",
}


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_soar_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if not _parser_has_option(parser, "--soar"):
        parser.add_argument("--soar", action="store_true", help="Enable experimental SOAR auxiliary correction")
    if not _parser_has_option(parser, "--soar_lambda_aux"):
        parser.add_argument(
            "--soar_lambda_aux",
            type=float,
            default=1.0,
            help=(
                "Weight for the SOAR auxiliary loss. Official HY-SOAR full fine-tuning uses 1.0. "
                "For LoRA training, recommend 0.1-0.3 to keep main objective dominant and avoid "
                "progressive blurring of generated samples."
            ),
        )
    if not _parser_has_option(parser, "--soar_trajectory_length"):
        parser.add_argument(
            "--soar_trajectory_length",
            type=int,
            default=6,
            help="Number of auxiliary points sampled from the single ODE rollout path",
        )
    if not _parser_has_option(parser, "--soar_num_sampling_steps"):
        parser.add_argument(
            "--soar_num_sampling_steps",
            type=int,
            default=40,
            help="Sampler step count used to define the single-step rollout distance",
        )
    if not _parser_has_option(parser, "--soar_sigma_upper_ratio"):
        parser.add_argument(
            "--soar_sigma_upper_ratio",
            type=float,
            default=1.5,
            help=(
                "Upper sigma ratio for SOAR auxiliary interpolation, clamped to 1.0. "
                "1.5 matches HY-SOAR full fine-tuning. For LoRA, recommend 1.0 to keep auxiliary "
                "points within the same noise band as the main step (prevents over-noisy supervision)."
            ),
        )
    if not _parser_has_option(parser, "--soar_cfg_scale_sampling"):
        parser.add_argument(
            "--soar_cfg_scale_sampling",
            type=float,
            default=SOAR_CFG_SCALE_SAMPLING_DEFAULT,
            help=(
                "CFG scale used only for SOAR rollout point generation. Default 4.5 matches "
                "HY-SOAR. Set to 1.0 to disable CFG rollout (cond-only SOAR-lite). When != 1.0, "
                "an empty-prompt cache must exist for each dataset cache_directory; generate it "
                "via the matching `*_cache_text_encoder_outputs.py --soar` script."
            ),
        )
    return parser


def validate_soar_args(args: argparse.Namespace, *, allow_fused_backward: bool = False) -> None:
    if not hasattr(args, "soar"):
        return
    if args.soar and not allow_fused_backward and getattr(args, "fused_backward_pass", False):
        raise ValueError("--soar is not compatible with --fused_backward_pass")
    if args.soar_lambda_aux < 0:
        raise ValueError("--soar_lambda_aux must be non-negative")
    if args.soar_trajectory_length < 1:
        raise ValueError("--soar_trajectory_length must be at least 1")
    if args.soar_num_sampling_steps < 2:
        raise ValueError("--soar_num_sampling_steps must be at least 2")
    if getattr(args, "soar_sigma_upper_ratio", 1.5) < 1.0:
        raise ValueError("--soar_sigma_upper_ratio must be at least 1.0")
    if getattr(args, "soar_cfg_scale_sampling", SOAR_CFG_SCALE_SAMPLING_DEFAULT) <= 0:
        raise ValueError("--soar_cfg_scale_sampling must be positive")


def is_soar_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "soar", False)
        and getattr(args, "soar_lambda_aux", 0.0) > 0
        and getattr(args, "soar_trajectory_length", 0) > 0
    )


def is_soar_cfg_rollout_enabled(args: argparse.Namespace) -> bool:
    return abs(float(getattr(args, "soar_cfg_scale_sampling", SOAR_CFG_SCALE_SAMPLING_DEFAULT)) - 1.0) > 1e-6


def combine_cfg_standard_velocity(
    uncond_velocity: torch.Tensor, cond_velocity: torch.Tensor, cfg_scale_sampling: float
) -> torch.Tensor:
    return uncond_velocity + float(cfg_scale_sampling) * (cond_velocity - uncond_velocity)


def get_soar_empty_prompt_cache_path(cache_directory: str | os.PathLike, architecture: str) -> str:
    architecture = str(architecture).replace(os.sep, "_").replace("/", "_")
    return os.path.join(str(cache_directory), SOAR_EMPTY_PROMPT_CACHE_DIR, f"{architecture}.safetensors")


def get_legacy_soar_empty_prompt_cache_path(cache_directory: str | os.PathLike, architecture: str) -> str:
    architecture = str(architecture).replace(os.sep, "_").replace("/", "_")
    return os.path.join(str(cache_directory), f"{SOAR_EMPTY_PROMPT_CACHE_PREFIX}_{architecture}.safetensors")


def get_dataset_cache_directories(train_dataset_group) -> list[str]:
    directories = []
    seen = set()
    for dataset in getattr(train_dataset_group, "datasets", []):
        cache_directory = getattr(dataset, "cache_directory", None)
        if cache_directory is None:
            continue
        cache_directory = os.path.normpath(str(cache_directory))
        if cache_directory not in seen:
            directories.append(cache_directory)
            seen.add(cache_directory)
    return directories


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def make_soar_empty_prompt_tensor_key(tensor_base_key: str, tensor: torch.Tensor) -> str:
    return f"{tensor_base_key}_{_dtype_to_str(tensor.dtype)}"


def save_soar_empty_prompt_cache(
    *,
    cache_directories: list[str] | tuple[str, ...],
    architecture: str,
    tensor_base_key: str,
    tensor: torch.Tensor,
    metadata: Optional[dict[str, str]] = None,
    skip_existing: bool = False,
) -> list[str]:
    saved_paths = []
    tensor = tensor.detach().cpu().contiguous()
    tensor_key = make_soar_empty_prompt_tensor_key(tensor_base_key, tensor)
    metadata = dict(metadata or {})
    metadata.update(
        {
            "cache_kind": "soar_empty_prompt",
            "prompt": "",
            "architecture": str(architecture),
            "format_version": "1",
        }
    )
    metadata = {str(key): str(value) for key, value in metadata.items()}

    for cache_directory in cache_directories:
        cache_path = get_soar_empty_prompt_cache_path(cache_directory, architecture)
        legacy_cache_path = get_legacy_soar_empty_prompt_cache_path(cache_directory, architecture)
        if os.path.exists(legacy_cache_path):
            os.remove(legacy_cache_path)
        if skip_existing and os.path.exists(cache_path):
            continue
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        save_file({tensor_key: tensor}, cache_path, metadata=metadata)
        saved_paths.append(cache_path)
    return saved_paths


def load_soar_empty_prompt_tensor(
    *,
    cache_directories: list[str] | tuple[str, ...],
    architecture: str,
    tensor_base_key: str,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    checked_paths = []
    for cache_directory in cache_directories:
        cache_paths = [
            get_soar_empty_prompt_cache_path(cache_directory, architecture),
            get_legacy_soar_empty_prompt_cache_path(cache_directory, architecture),
        ]
        existing_cache_path = next((path for path in cache_paths if os.path.exists(path)), None)
        checked_paths.extend(cache_paths)
        if existing_cache_path is None:
            continue

        tensors = load_file(existing_cache_path, device=str(device))
        exact_key = tensor_base_key
        if exact_key in tensors:
            return tensors[exact_key]
        matching_keys = sorted(key for key in tensors if key.startswith(f"{tensor_base_key}_"))
        if not matching_keys:
            raise ValueError(
                f"SOAR empty prompt cache {existing_cache_path} does not contain tensor key {tensor_base_key}_<dtype>"
            )
        return tensors[matching_keys[0]]

    raise FileNotFoundError(
        "SOAR CFG rollout requires an empty prompt cache. Checked: "
        + ", ".join(checked_paths)
        + ". Re-run the matching text encoder cache script first."
    )


def compute_loss_weighting_from_sigma(
    weighting_scheme: str,
    sigmas: torch.Tensor,
    target_ndim: int,
) -> Optional[torch.Tensor]:
    sigmas = sigmas.float()
    if weighting_scheme == "sigma_sqrt":
        weighting = sigmas**-2.0
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        return None

    while weighting.ndim > target_ndim:
        if weighting.shape[-1] != 1:
            raise ValueError(f"Cannot squeeze weighting shape {tuple(weighting.shape)} to rank {target_ndim}")
        weighting = weighting.squeeze(-1)
    while weighting.ndim < target_ndim:
        weighting = weighting.unsqueeze(-1)
    return weighting


def is_continuous_timestep_sampling(timestep_sampling: str) -> bool:
    return timestep_sampling in CONTINUOUS_TIMESTEP_SAMPLINGS


def get_sigmas_from_continuous_timesteps(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    scheduler_config = getattr(noise_scheduler, "config", None)
    num_train_timesteps = getattr(scheduler_config, "num_train_timesteps", len(noise_scheduler.timesteps))
    sigma = ((timesteps.to(device=device, dtype=torch.float32) - 1.0) / float(num_train_timesteps)).clamp(0.0, 1.0)
    sigma = sigma.to(dtype=dtype).flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_per_sample_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    weighting: Optional[torch.Tensor],
) -> torch.Tensor:
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    if weighting is not None:
        if weighting.ndim != loss.ndim:
            raise ValueError(f"Weighting rank {weighting.ndim} does not match loss rank {loss.ndim}")
        if weighting.shape[0] != loss.shape[0]:
            raise ValueError(f"Weighting batch {weighting.shape[0]} does not match loss batch {loss.shape[0]}")
        for weight_dim, loss_dim in zip(weighting.shape[1:], loss.shape[1:]):
            if weight_dim not in (1, loss_dim):
                raise ValueError(f"Weighting shape {tuple(weighting.shape)} is not broadcast-safe for loss {tuple(loss.shape)}")
        loss = loss * weighting.float()
    return loss.reshape(loss.shape[0], -1).mean(dim=1)


def default_flow_matching_target(clean_latents: torch.Tensor, aux_latents: torch.Tensor, aux_sigmas: torch.Tensor) -> torch.Tensor:
    return (aux_latents - clean_latents) / aux_sigmas


def zimage_flow_matching_target(clean_latents: torch.Tensor, aux_latents: torch.Tensor, aux_sigmas: torch.Tensor) -> torch.Tensor:
    return (clean_latents - aux_latents) / aux_sigmas


def run_soar_auxiliary_pass(
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    predict_velocity_fn: SoarPredictFn,
    target_fn: SoarTargetFn,
    clean_latents: torch.Tensor,
    aux_points: list[dict[str, torch.Tensor]],
    total_count: float,
) -> tuple[torch.Tensor, float]:
    loss_aux_sum = torch.tensor(0.0, device=clean_latents.device, dtype=torch.float32)
    aux_count = 0.0
    batch_size = float(clean_latents.shape[0])

    for point in aux_points:
        aux_latents = point["latents"].to(device=clean_latents.device, dtype=clean_latents.dtype)
        aux_sigmas = point["sigmas"].to(device=clean_latents.device, dtype=torch.float32)
        aux_sigmas_expanded = aux_sigmas.view(-1, *([1] * (clean_latents.ndim - 1)))
        aux_timesteps = point["timesteps"].to(device=clean_latents.device, dtype=torch.float32)

        target = target_fn(clean_latents, aux_latents, aux_sigmas_expanded)
        model_pred = predict_velocity_fn(aux_latents, aux_timesteps)
        weighting = compute_loss_weighting_from_sigma(args.weighting_scheme, aux_sigmas_expanded, model_pred.ndim)
        per_sample = compute_per_sample_loss(model_pred, target, weighting)
        point_loss_sum = per_sample.sum()

        accelerator.backward(args.soar_lambda_aux * point_loss_sum / total_count)

        loss_aux_sum += point_loss_sum.detach()
        aux_count += batch_size

    return loss_aux_sum, aux_count

"""Versioned MOSS caches in Musubi's native safetensors format."""

import os
import re

import torch
from safetensors import safe_open

from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.dataset.cache_io import CACHE_FORMAT_VERSION, save_latent_cache_common, save_text_encoder_output_cache_common
from musubi_tuner.moss_image.utils import MAX_TOKEN_LENGTH, TEXT_ENCODER_HIDDEN_SIZE, pad_text_embeds

CACHE_VERSION = "1"
LATENT_FORMAT = "normalized_patchified_128"
TEXT_KEYS = ("varlen_moss_embed_bfloat16", "varlen_moss_uncond_bfloat16")


def _finite(tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError("MOSS cache tensors must contain only finite values; regenerate the cache")


def _text_shape(shape, max_length):
    if len(shape) != 2 or not 1 <= shape[0] <= max_length or shape[1] != TEXT_ENCODER_HIDDEN_SIZE:
        raise ValueError(f"MOSS text features must be [1..{max_length}, {TEXT_ENCODER_HIDDEN_SIZE}], got {shape}")


def save_latents(item, latent: torch.Tensor):
    if latent.ndim != 3 or latent.shape[0] != 128 or min(latent.shape[1:]) < 1 or latent.dtype != torch.float32:
        raise ValueError("MOSS latents must be normalized, patchified FP32 [128, H/16, W/16]")
    _finite(latent)
    h, w = latent.shape[-2:]
    if item.bucket_size is not None and tuple(item.bucket_size[:2]) != (w * 16, h * 16):
        raise ValueError("MOSS latent spatial dimensions do not match the image bucket")
    save_latent_cache_common(
        item,
        {f"latents_{h}x{w}_float32": latent.detach().cpu().contiguous()},
        ARCHITECTURE_MOSS_IMAGE_FULL,
        additional_metadata={"moss_cache_version": CACHE_VERSION, "moss_latent_format": LATENT_FORMAT},
    )


def save_text(item, embed: torch.Tensor, uncond: torch.Tensor, max_length: int = MAX_TOKEN_LENGTH):
    if not 1 <= max_length <= MAX_TOKEN_LENGTH:
        raise ValueError(f"max_length must be in [1, {MAX_TOKEN_LENGTH}]")
    for row in (embed, uncond):
        _text_shape(row.shape, max_length)
        if row.dtype != torch.bfloat16:
            raise ValueError("MOSS text caches must retain BF16 features")
        _finite(row)
    save_text_encoder_output_cache_common(
        item,
        {key: row.detach().cpu().contiguous() for key, row in zip(TEXT_KEYS, (embed, uncond))},
        ARCHITECTURE_MOSS_IMAGE_FULL,
        merge_existing=False,
        additional_metadata={"moss_cache_version": CACHE_VERSION, "moss_max_token_length": str(max_length)},
    )


def load_cache(path: str, kind: str, *, header_only: bool = False, expected_size=None) -> dict[str, torch.Tensor]:
    if kind not in ("latent", "text"):
        raise ValueError("cache kind must be latent or text")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("architecture") != ARCHITECTURE_MOSS_IMAGE_FULL:
            raise ValueError(f"MOSS cache architecture mismatch: {path}")
        if metadata.get("moss_cache_version") != CACHE_VERSION or metadata.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(f"MOSS cache version mismatch; regenerate: {path}")
        keys = set(handle.keys())
        if kind == "latent":
            if metadata.get("moss_latent_format") != LATENT_FORMAT or len(keys) != 1:
                raise ValueError(f"Invalid MOSS latent format: {path}")
            key = next(iter(keys))
            match = re.fullmatch(r"latents_(\d+)x(\d+)_float32", key)
            view = handle.get_slice(key)
            shape = view.get_shape()
            if (
                not match
                or len(shape) != 3
                or shape[0] != 128
                or min(shape[1:]) < 1
                or shape[1:] != [int(match[1]), int(match[2])]
                or view.get_dtype() != "F32"
            ):
                raise ValueError(f"Invalid normalized MOSS latent tensor: {path}")
            if expected_size is not None and (shape[2] * 16, shape[1] * 16) != tuple(expected_size[:2]):
                raise ValueError(f"MOSS latent bucket mismatch; regenerate for the current dataset resolution: {path}")
        else:
            if keys != set(TEXT_KEYS):
                raise ValueError(f"MOSS text cache must contain prompt and unconditional features: {path}")
            try:
                max_length = int(metadata["moss_max_token_length"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Missing/invalid MOSS text length metadata: {path}") from exc
            if not 1 <= max_length <= MAX_TOKEN_LENGTH:
                raise ValueError(f"Invalid MOSS maximum text length: {path}")
            for key in keys:
                view = handle.get_slice(key)
                _text_shape(view.get_shape(), max_length)
                if view.get_dtype() != "BF16":
                    raise ValueError(f"MOSS text features must be BF16: {path}")
        if header_only:
            return {}
        tensors = {key: handle.get_tensor(key) for key in keys}
    for value in tensors.values():
        _finite(value)
    return tensors


def cache_is_current(item, kind: str, max_length: int = MAX_TOKEN_LENGTH) -> bool:
    path = item.latent_cache_path if kind == "latent" else item.text_encoder_output_cache_path
    if not os.path.isfile(path):
        return False
    data = load_cache(path, kind)
    if kind == "text":
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata()
        return metadata.get("caption1") == item.caption and metadata.get("moss_max_token_length") == str(max_length)
    if item.bucket_size is not None:
        latent = next(iter(data.values()))
        return tuple(item.bucket_size[:2]) == (latent.shape[-1] * 16, latent.shape[-2] * 16)
    return True


def prepare_conditioning(prompts, unconditional, dropout_rate: float, device: torch.device):
    if not 0 <= dropout_rate <= 1:
        raise ValueError("caption_dropout_rate must be in [0, 1]")
    if len(prompts) != len(unconditional):
        raise ValueError("MOSS prompt and unconditional batch lengths differ")
    selected = []
    for prompt, uncond in zip(prompts, unconditional):
        dropped = dropout_rate == 1 or (dropout_rate > 0 and torch.rand(()).item() < dropout_rate)
        selected.append((uncond if dropped else prompt).to(device=device))
    return pad_text_embeds(selected)


def load_datasets(args):
    from musubi_tuner.dataset import config_utils
    from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
    from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

    config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(config, args, architecture=ARCHITECTURE_MOSS_IMAGE)
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    validate_datasets(datasets)
    return datasets


def validate_datasets(datasets):
    from musubi_tuner.dataset.image_video_dataset import ImageDataset

    if any(not isinstance(dataset, ImageDataset) for dataset in datasets):
        raise ValueError("MOSS-Image supports image datasets only")
    for dataset in datasets:
        if getattr(dataset, "control_directory", None) or getattr(dataset, "num_control_images", 0):
            raise ValueError("MOSS-Image does not support control/reference images")

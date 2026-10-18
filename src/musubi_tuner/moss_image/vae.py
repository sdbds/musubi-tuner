"""MOSS FLUX.2 VAE compatibility, adapted from sd-scripts (Apache-2.0).

The reused Musubi AutoEncoder encodes and decodes normalized, patchified latents.
"""

import logging
from typing import Optional, Union

import torch
from torch import Tensor
from accelerate import init_empty_weights

from musubi_tuner.flux_2.flux2_models import AutoEncoder, AutoEncoderParams
from musubi_tuner.utils.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)
PATCH_SIZE = 2
BN_EPS = 1e-4


def patchify_latents(z: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, C * 4, H / 2, W / 2] (2x2 patchify, channel order (c, pi, pj))"""
    b, c, h, w = z.shape
    return (
        z.reshape(b, c, h // PATCH_SIZE, PATCH_SIZE, w // PATCH_SIZE, PATCH_SIZE)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * PATCH_SIZE * PATCH_SIZE, h // PATCH_SIZE, w // PATCH_SIZE)
    )


def unpatchify_latents(z: Tensor) -> Tensor:
    """[B, C * 4, H / 2, W / 2] -> [B, C, H, W]"""
    b, c4, h, w = z.shape
    c = c4 // (PATCH_SIZE * PATCH_SIZE)
    return z.reshape(b, c, PATCH_SIZE, PATCH_SIZE, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * PATCH_SIZE, w * PATCH_SIZE)


def normalize_latents(z: Tensor, running_mean: Tensor, running_var: Tensor, eps: float = BN_EPS) -> Tensor:
    """BatchNorm (eval) normalization of patchified latents with the given statistics."""
    s = torch.sqrt(running_var.to(z) + eps).view(1, -1, 1, 1)
    m = running_mean.to(z).view(1, -1, 1, 1)
    return (z - m) / s


def denormalize_latents(z: Tensor, running_mean: Tensor, running_var: Tensor, eps: float = BN_EPS) -> Tensor:
    """Inverse of `normalize_latents`."""
    s = torch.sqrt(running_var.to(z) + eps).view(1, -1, 1, 1)
    m = running_mean.to(z).view(1, -1, 1, 1)
    return z * s + m


def _convert_diffusers_vae_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a diffusers AutoencoderKLFlux2 state dict to the original (BFL) key names."""
    new_sd = {}
    num_levels = len({k.split(".")[2] for k in sd if k.startswith("decoder.up_blocks.")})

    def convert_resnet(key: str) -> str:
        return key.replace("conv_shortcut.", "nin_shortcut.")

    for key, value in sd.items():
        new_key = key
        if key.startswith("quant_conv."):
            new_key = "encoder." + key
        elif key.startswith("post_quant_conv."):
            new_key = "decoder." + key
        elif ".mid_block." in key:
            prefix, rest = key.split(".mid_block.", 1)
            if rest.startswith("resnets."):
                _, idx, rest2 = rest.split(".", 2)
                new_key = f"{prefix}.mid.block_{int(idx) + 1}.{convert_resnet(rest2)}"
            elif rest.startswith("attentions."):
                _, idx, rest2 = rest.split(".", 2)
                mapping = {
                    "group_norm": "norm",
                    "to_q": "q",
                    "to_k": "k",
                    "to_v": "v",
                    "to_out.0": "proj_out",
                }
                for src, dst in mapping.items():
                    if rest2.startswith(src + "."):
                        rest2 = dst + rest2[len(src) :]
                        break
                new_key = f"{prefix}.mid.attn_1.{rest2}"
                if rest2.endswith(".weight") and value.ndim == 2:
                    value = value[:, :, None, None]  # Linear -> 1x1 Conv2d
        elif key.startswith("encoder.down_blocks."):
            _, _, idx, rest = key.split(".", 3)
            if rest.startswith("resnets."):
                _, bidx, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{idx}.block.{bidx}.{convert_resnet(rest2)}"
            elif rest.startswith("downsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{idx}.downsample.{rest2}"
        elif key.startswith("decoder.up_blocks."):
            _, _, idx, rest = key.split(".", 3)
            level = num_levels - 1 - int(idx)  # diffusers up_blocks.0 is the deepest level (BFL up.{n-1})
            if rest.startswith("resnets."):
                _, bidx, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.block.{bidx}.{convert_resnet(rest2)}"
            elif rest.startswith("upsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.upsample.{rest2}"
        elif key.startswith("encoder.conv_norm_out.") or key.startswith("decoder.conv_norm_out."):
            new_key = key.replace(".conv_norm_out.", ".norm_out.")
        new_sd[new_key] = value
    return new_sd


def is_diffusers_vae_state_dict(sd: dict[str, Tensor]) -> bool:
    return any(k.startswith("encoder.down_blocks.") for k in sd)


def load_vae(
    ckpt_path: str, dtype: Optional[torch.dtype], device: Union[str, torch.device], disable_mmap: bool = False
) -> AutoEncoder:
    """Load the FLUX.2 VAE from a safetensors file (BFL / ComfyUI format or diffusers format)."""
    logger.info(f"Loading FLUX.2 VAE from {ckpt_path}")
    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap)
    if dtype is not None:
        sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}  # keep bn.num_batches_tracked as is

    # remove prefixes used by some single-file checkpoints
    for prefix in ["vae.", "first_stage_model."]:
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}

    if is_diffusers_vae_state_dict(sd):
        logger.info("Converting diffusers format VAE state dict to the original format")
        sd = _convert_diffusers_vae_state_dict(sd)

    encoder_ch = sd["encoder.conv_in.weight"].shape[0]
    levels = sorted({int(key.split(".")[2]) for key in sd if key.startswith("encoder.down.")})
    blocks = {int(key.split(".")[4]) for key in sd if key.startswith("encoder.down.0.block.")}
    params = AutoEncoderParams(
        ch=encoder_ch,
        ch_mult=[sd[f"encoder.down.{level}.block.0.conv2.weight"].shape[0] // encoder_ch for level in levels],
        num_res_blocks=len(blocks),
        z_channels=sd["decoder.post_quant_conv.weight"].shape[0],
    )
    if len(levels) != 4 or params.z_channels != 32:
        raise ValueError("MOSS requires a FLUX.2 VAE with spatial downscale 8 and 32 raw latent channels")
    decoder_ch = sd["decoder.conv_out.weight"].shape[1]  # ch * ch_mult[0] = ch
    if decoder_ch != params.ch:
        logger.info(f"Decoder width {decoder_ch} differs from the encoder width {params.ch} (FLUX.2 small decoder)")
        params.decoder_ch = decoder_ch

    with init_empty_weights():
        ae = AutoEncoder(params)
    if dtype is not None:
        ae.to(dtype)

    info = ae.load_state_dict(sd, strict=True, assign=True)
    ae.to(device)  # buffers created by init_empty_weights (bn.num_batches_tracked) are on the CPU
    ae.bn.eval()
    logger.info(f"Loaded FLUX.2 VAE: {info}")
    return ae

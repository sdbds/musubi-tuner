# LoRA module for Lance

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora


LANCE_TARGET_REPLACE_MODULES = ["Lance"]

LANCE_SAFE_INCLUDE_PATTERNS = [
    r"language_model\.model\.layers\.\d+\.self_attn\.(q_proj_moe_gen|k_proj_moe_gen|v_proj_moe_gen|o_proj_moe_gen)",
    r"language_model\.model\.layers\.\d+\.mlp_moe_gen\..*",
    r"vae2llm",
    r"llm2vae",
]

# The LoRA base implementation treats include patterns as overrides for
# excluded modules. This negative-lookahead excludes everything except the
# conservative generation expert surface above; user include_patterns can opt
# additional modules back in explicitly.
LANCE_DEFAULT_EXCLUDE_PATTERNS = [r"(?!(" + "|".join(LANCE_SAFE_INCLUDE_PATTERNS) + r")$).*"]


def _coerce_patterns(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise TypeError(f"patterns must be a list, tuple, string, or None, got {type(value)!r}")


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    exclude_patterns = _coerce_patterns(kwargs.get("exclude_patterns", None))
    exclude_patterns.extend(LANCE_DEFAULT_EXCLUDE_PATTERNS)
    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        LANCE_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        LANCE_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )

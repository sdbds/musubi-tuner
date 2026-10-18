"""MOSS adapters with sd-scripts-compatible lora_unet parameter names."""

import torch

from musubi_tuner.networks import lora

MOSS_TARGET_REPLACE_MODULES = ["MossImageBlock"]
MOSS_ADAPTER_REPLACE_MODULES = ["MossImageTextEmbedder"]


def _enabled(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() not in ("true", "false"):
        raise ValueError("train_text_embedder must be true or false")
    return str(value).lower() == "true"


def create_arch_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    targets = list(MOSS_TARGET_REPLACE_MODULES)
    if _enabled(kwargs.pop("train_text_embedder", False)) or _enabled(kwargs.pop("train_llm_adapter", False)):
        targets += MOSS_ADAPTER_REPLACE_MODULES
    network = lora.create_network(
        targets,
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
    if not network.unet_loras:
        raise ValueError("MOSS LoRA found no target modules")
    return network


def create_arch_network_from_weights(multiplier, weights_sd, text_encoders=None, unet=None, for_inference=False, **kwargs):
    validate_weights(weights_sd, unet)
    if any(key.startswith("lora_te") for key in weights_sd):
        raise ValueError("MOSS text encoder LoRA is not supported")
    requested = {key.split(".", 1)[0] for key in weights_sd if key.endswith(".lora_down.weight")}
    if not requested:
        raise ValueError("No MOSS LoRA weights found")
    network = lora.create_network_from_weights(
        MOSS_TARGET_REPLACE_MODULES + MOSS_ADAPTER_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
    created = {module.lora_name for module in network.unet_loras}
    if requested != created:
        raise ValueError(f"MOSS LoRA weights do not map to supported modules: {sorted(requested - created)}")
    return network


def validate_weights(weights, model):
    modules = {
        "lora_unet_" + name.replace(".", "_"): module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and (name.startswith("layers.") or name.startswith("text_embedder."))
    }
    names = {key.split(".", 1)[0] for key in weights}
    if not names or names - modules.keys():
        raise ValueError(f"MOSS LoRA contains unsupported modules: {sorted(names - modules.keys())}")
    allowed = {name + suffix for name in names for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")}
    if set(weights) - allowed:
        raise ValueError("MOSS supports standard DiT LoRA down/up/alpha weights only")
    for name in names:
        down, up = weights.get(name + ".lora_down.weight"), weights.get(name + ".lora_up.weight")
        module = modules[name]
        if (
            down is None
            or up is None
            or down.ndim != 2
            or up.ndim != 2
            or down.shape[0] < 1
            or down.shape[1] != module.in_features
            or up.shape != (module.out_features, down.shape[0])
        ):
            raise ValueError(f"Invalid MOSS LoRA dimensions for {name}")
        alpha = weights.get(name + ".alpha")
        if alpha is not None and alpha.numel() != 1:
            raise ValueError(f"Invalid MOSS LoRA alpha for {name}")
        for value in (down, up, alpha):
            if value is not None and not torch.isfinite(value).all():
                raise ValueError("MOSS LoRA weights must be finite")

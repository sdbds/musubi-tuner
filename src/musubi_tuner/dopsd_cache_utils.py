from __future__ import annotations

import logging
import os

from PIL import Image
import torch

logger = logging.getLogger(__name__)

QWEN3_VL_PROCESSOR_IDS = {
    "4B": "Qwen/Qwen3-VL-4B-Instruct",
    "8B": "Qwen/Qwen3-VL-8B-Instruct",
}
MIN_QWEN3_VL_TRANSFORMERS_VERSION = "4.57.6"


def content_to_pil_image(content) -> Image.Image:
    if isinstance(content, list):
        if len(content) == 0:
            raise ValueError("D-OPSD teacher cache requires a target image, got an empty content list")
        content = content[0]
    if content is None:
        raise ValueError("D-OPSD teacher cache requires dataset content. Use image datasets with cached latents.")
    image = Image.fromarray(content[..., :3]) if not isinstance(content, Image.Image) else content
    return image.convert("RGB")


def _resolve_module(root: torch.nn.Module, paths: tuple[str, ...]) -> tuple[str, torch.nn.Module]:
    for path in paths:
        module = root
        found = True
        for attr in path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                found = False
                break
        if found and isinstance(module, torch.nn.Module):
            return path, module
    raise ValueError(f"Could not find any module path from: {', '.join(paths)}")


def qwen3_vl_processor_id_for_variant(qwen_variant: str) -> str:
    normalized_variant = qwen_variant.upper()
    if normalized_variant not in QWEN3_VL_PROCESSOR_IDS:
        supported = ", ".join(sorted(QWEN3_VL_PROCESSOR_IDS))
        raise ValueError(f"Unsupported Qwen3-VL processor variant '{qwen_variant}'. Supported variants: {supported}")
    return QWEN3_VL_PROCESSOR_IDS[normalized_variant]


def load_qwen3_vl_processor(qwen_variant: str):
    import transformers

    require_qwen3_vl_transformers(transformers)
    processor_id = qwen3_vl_processor_id_for_variant(qwen_variant)
    logger.info(f"Loading official Qwen3-VL processor from {processor_id}")
    return transformers.AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)


def require_qwen3_vl_transformers(transformers_module) -> None:
    from packaging.version import Version

    installed = Version(transformers_module.__version__)
    required = Version(MIN_QWEN3_VL_TRANSFORMERS_VERSION)
    if installed < required:
        raise RuntimeError(
            "Qwen3-VL D-OPSD teacher cache requires "
            f"transformers>={MIN_QWEN3_VL_TRANSFORMERS_VERSION}; installed version is {transformers_module.__version__}. "
            "Install this project's pinned dependency set or upgrade transformers."
        )


def _is_safetensors_path(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".safetensors")


def _strip_state_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def normalize_qwen_vl_single_file_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized_state = {}
    converted = 0
    for key, value in state_dict.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("model."):
            new_key = key.replace("model.", "model.language_model.", 1)
            converted += 1
        elif key.startswith("visual."):
            new_key = key.replace("visual.", "model.visual.", 1)
            converted += 1
        else:
            new_key = key

        if new_key in normalized_state:
            raise ValueError(f"Duplicate D-OPSD teacher encoder key after Qwen-VL prefix normalization: {new_key}")
        normalized_state[new_key] = value

    if converted:
        logger.info(
            "Normalized %d single-file Qwen-VL safetensors keys to transformers format "
            "(model.* -> model.language_model.*, visual.* -> model.visual.*)",
            converted,
        )
    return normalized_state


def prepare_qwen_vl_state_dict_for_load(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    if (
        "lm_head.weight" in model_state
        and "lm_head.weight" not in state_dict
        and "model.language_model.embed_tokens.weight" in state_dict
        and model_state["lm_head.weight"].shape == state_dict["model.language_model.embed_tokens.weight"].shape
    ):
        state_dict = dict(state_dict)
        state_dict["lm_head.weight"] = state_dict["model.language_model.embed_tokens.weight"]
        logger.info("Added tied Qwen-VL lm_head.weight from model.language_model.embed_tokens.weight for strict load")
    return state_dict


def _load_qwen3_language_state(
    llm_weight_source: str,
    dtype: torch.dtype,
) -> tuple[str, dict[str, torch.Tensor]]:
    if not _is_safetensors_path(llm_weight_source):
        import transformers

        logger.info(f"Loading D-OPSD teacher LLM reweight source from {llm_weight_source}")
        source_model = transformers.AutoModelForCausalLM.from_pretrained(
            llm_weight_source,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        source_model.eval()
        source_path, source_lm = _resolve_module(
            source_model,
            (
                "model",
                "language_model",
                "model.language_model",
                "transformer",
            ),
        )
        source_state = {key: value.detach().cpu() for key, value in source_lm.state_dict().items()}
        del source_model
        return source_path, source_state

    from musubi_tuner.utils.safetensors_utils import load_split_weights

    logger.info(f"Loading D-OPSD teacher LLM reweight source safetensors from {llm_weight_source}")
    raw_state = load_split_weights(llm_weight_source, device="cpu", dtype=dtype)
    model_state = _strip_state_prefix(raw_state, "model.")
    if not model_state:
        model_state = raw_state
    return "safetensors:model", model_state


def replace_vlm_language_model_weights(
    teacher_encoder: torch.nn.Module,
    llm_weight_source: str,
    dtype: torch.dtype,
) -> None:
    target_path, target_lm = _resolve_module(
        teacher_encoder,
        (
            "language_model",
            "model.language_model",
            "model.llm",
            "llm",
            "text_model",
            "model.text_model",
        ),
    )
    source_path, source_state = _load_qwen3_language_state(llm_weight_source, dtype)

    target_state = target_lm.state_dict()
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    matched_params = sum(value.numel() for value in compatible_state.values())
    target_params = sum(value.numel() for value in target_state.values())
    matched_ratio = matched_params / max(target_params, 1)
    if matched_ratio < 0.8:
        raise ValueError(
            "D-OPSD teacher LLM reweight matched too few parameters "
            f"({matched_ratio:.1%}) between source '{source_path}' and target '{target_path}'. "
            "Use a Qwen3 text model that matches the Qwen3-VL language hidden architecture."
        )

    target_lm.load_state_dict(compatible_state, strict=False)
    logger.info(
        "Applied D-OPSD teacher LLM reweight: "
        f"{matched_ratio:.1%} of target language parameters copied from {source_path} to {target_path}"
    )


def _load_auto_vlm_from_safetensors(
    model_class,
    model_path: str,
    model_config_id: str,
    dtype: torch.dtype,
):
    import transformers
    from accelerate import init_empty_weights

    from musubi_tuner.utils.safetensors_utils import load_split_weights

    require_qwen3_vl_transformers(transformers)
    logger.info(f"Loading D-OPSD teacher encoder config from {model_config_id}")
    config = transformers.AutoConfig.from_pretrained(model_config_id, trust_remote_code=True)
    with init_empty_weights():
        model = model_class.from_config(config, trust_remote_code=True)

    logger.info(f"Loading D-OPSD teacher encoder safetensors from {model_path}")
    state_dict = load_split_weights(model_path, device="cpu", dtype=dtype)
    state_dict = normalize_qwen_vl_single_file_state_dict(state_dict)
    state_dict = prepare_qwen_vl_state_dict_for_load(model, state_dict)
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded D-OPSD teacher encoder from safetensors: {info}")
    return model


def load_auto_vlm(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    llm_reweight_source: str | None,
    already_reweighted: bool,
    allow_raw_vlm: bool,
    recipe_name: str,
    model_config_id: str | None = None,
):
    import transformers

    require_qwen3_vl_transformers(transformers)
    model_classes = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            model_classes.append(model_class)

    if llm_reweight_source is None and not already_reweighted and not allow_raw_vlm:
        raise ValueError(
            f"Paper-consistent {recipe_name} D-OPSD teacher cache requires a VLM with its LLM component reweighted "
            "from the matching Qwen3 text model. The cache script normally uses --text_encoder for this. Pass "
            "--dopsd_teacher_already_reweighted if the supplied VLM checkpoint was prepared externally. "
            "For ablations only, pass --dopsd_teacher_allow_raw_vlm."
        )

    errors = []
    for model_class in model_classes:
        try:
            if _is_safetensors_path(model_path):
                if model_config_id is None:
                    raise ValueError("A model config id is required when loading a D-OPSD teacher from safetensors")
                model = _load_auto_vlm_from_safetensors(model_class, model_path, model_config_id, dtype)
            else:
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
            if llm_reweight_source is not None:
                replace_vlm_language_model_weights(model, llm_reweight_source, dtype)
            elif allow_raw_vlm:
                logger.warning(
                    f"Using a raw VLM for {recipe_name} D-OPSD teacher cache. "
                    "This is not the paper-consistent recipe and may produce feature-space mismatch artifacts."
                )
            else:
                logger.info("Using externally reweighted D-OPSD teacher encoder")
            model.to(device)
            model.eval()
            return model
        except Exception as exc:
            errors.append(f"{model_class.__name__}: {exc}")

    raise RuntimeError("Could not load D-OPSD teacher encoder with transformers Auto classes. " + " | ".join(errors))


def validate_multimodal_inputs(inputs) -> None:
    image_tensor_keys = ("pixel_values", "pixel_values_videos", "image_embeds")
    if any(key in inputs and torch.is_tensor(inputs[key]) and inputs[key].numel() > 0 for key in image_tensor_keys):
        return
    raise ValueError(
        "D-OPSD teacher processor did not produce image tensors. "
        "The teacher condition must be multimodal f_mm(y, x0); check that the processor/model path is a VLM."
    )

"""
Cache text encoder outputs for Z-Image architecture.

This script encodes text prompts using Z-Image's Qwen3 text encoder and caches
the embeddings for faster training. Z-Image uses only a single text encoder (Qwen3),
making this simpler than other architectures that use multiple encoders.
"""

import argparse
import logging

from PIL import Image
import torch
from transformers import AutoProcessor

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_Z_IMAGE, ItemInfo, save_text_encoder_output_cache_z_image
from musubi_tuner.dopsd_train_utils import DOPSD_TEACHER_EMBED_KEY
from musubi_tuner.hv_train_network import clean_memory_on_device
from musubi_tuner.utils import model_utils
from musubi_tuner.zimage import zimage_config, zimage_utils
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(tokenizer, text_encoder, batch: list[ItemInfo], device: torch.device):
    """
    Encode a batch of prompts and save their text encoder outputs.

    Args:
        tokenizer: Qwen3 tokenizer
        text_encoder: Qwen3 text encoder model
        batch: List of ItemInfo containing captions to encode
        device: Device to use for encoding
    """
    prompts = [item.caption for item in batch]

    # Encode prompts using Qwen3
    # get_text_embeds returns (prompt_embeds, prompt_masks)
    # prompt_embeds: (B, seq_len, hidden_size)
    # prompt_masks: (B, seq_len) boolean mask
    prompt_embeds, prompt_masks = zimage_utils.get_text_embeds(tokenizer, text_encoder, prompts)

    # Move to CPU for saving
    prompt_embeds = prompt_embeds.cpu()

    # Save each item's embedding
    # We save variable-length embeddings (trimmed to actual text length) to save space
    for item, embed, mask in zip(batch, prompt_embeds, prompt_masks):
        # Trim to actual text length based on attention mask
        actual_length = int(mask.sum().item())
        embed_trimmed = embed[:actual_length]  # (actual_length, hidden_size)

        save_text_encoder_output_cache_z_image(item, embed_trimmed)


def _content_to_pil_image(content) -> Image.Image:
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


def _replace_vlm_language_model_weights(
    teacher_encoder: torch.nn.Module,
    llm_weight_source: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> None:
    import transformers

    logger.info(f"Loading D-OPSD teacher LLM reweight source from {llm_weight_source}")
    source_model = transformers.AutoModelForCausalLM.from_pretrained(
        llm_weight_source,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    source_model.eval()

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
    source_path, source_lm = _resolve_module(
        source_model,
        (
            "model",
            "language_model",
            "model.language_model",
            "transformer",
        ),
    )

    target_state = target_lm.state_dict()
    source_state = source_lm.state_dict()
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
    del source_model


def _load_auto_vlm(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
    llm_reweight_source: str | None,
    already_reweighted: bool,
    allow_raw_vlm: bool,
):
    import transformers

    model_classes = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            model_classes.append(model_class)

    if llm_reweight_source is None and not already_reweighted and not allow_raw_vlm:
        raise ValueError(
            "Paper-consistent Z-Image D-OPSD teacher cache requires Qwen3-VL with its LLM component reweighted "
            "from Qwen3-4B. Pass --dopsd_teacher_llm_reweight_source path/to/qwen3-4b, or pass "
            "--dopsd_teacher_already_reweighted if the supplied VLM checkpoint was prepared externally. "
            "For ablations only, pass --dopsd_teacher_allow_raw_vlm."
        )

    errors = []
    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            if llm_reweight_source is not None:
                _replace_vlm_language_model_weights(model, llm_reweight_source, dtype, trust_remote_code)
            elif allow_raw_vlm:
                logger.warning(
                    "Using a raw VLM for D-OPSD teacher cache. This is not the paper's Z-Image recipe and may "
                    "produce feature-space mismatch artifacts."
                )
            else:
                logger.info("Using externally reweighted D-OPSD teacher encoder")
            model.to(device)
            model.eval()
            return model
        except Exception as exc:
            errors.append(f"{model_class.__name__}: {exc}")

    raise RuntimeError("Could not load D-OPSD teacher encoder with transformers Auto classes. " + " | ".join(errors))


def _validate_multimodal_inputs(inputs) -> None:
    image_tensor_keys = ("pixel_values", "pixel_values_videos", "image_embeds")
    if any(key in inputs and torch.is_tensor(inputs[key]) and inputs[key].numel() > 0 for key in image_tensor_keys):
        return
    raise ValueError(
        "D-OPSD teacher processor did not produce image tensors. "
        "The teacher condition must be multimodal f_mm(y, x0); check that the processor/model path is a VLM."
    )


def _apply_chat_template(processor, image: Image.Image, caption: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": caption},
            ],
        }
    ]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_and_save_dopsd_teacher_batch(
    processor,
    teacher_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    hidden_state_index: int,
    expected_dim: int,
    teacher_embed_key: str,
):
    images = [_content_to_pil_image(item.content) for item in batch]
    texts = [_apply_chat_template(processor, image, item.caption) for image, item in zip(images, batch)]

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    _validate_multimodal_inputs(inputs)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = teacher_encoder(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states[hidden_state_index]
    attention_mask = inputs.attention_mask.to(dtype=torch.bool)

    for item, hidden, mask in zip(batch, hidden_states, attention_mask):
        embed = hidden[mask].detach().cpu()
        if embed.shape[-1] != expected_dim:
            raise ValueError(
                f"D-OPSD teacher embedding dim {embed.shape[-1]} does not match Z-Image cap_feat_dim {expected_dim}. "
                "Use a Qwen3-VL teacher whose LLM hidden size matches Qwen3-4B, or cache externally with the same key."
            )
        save_text_encoder_output_cache_z_image(item, dopsd_teacher_embed=embed, dopsd_teacher_key=teacher_embed_key)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = zimage_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_Z_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # Prepare cache files and paths
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Determine dtype for text encoder
    if args.fp8_llm:
        te_dtype = torch.float8_e4m3fn
        logger.info("Using fp8 for Qwen3 text encoder")
    else:
        te_dtype = torch.bfloat16
        logger.info("Using bfloat16 for Qwen3 text encoder")

    # Load Qwen3 tokenizer and text encoder
    logger.info(f"Loading Qwen3 text encoder from {args.text_encoder}")
    tokenizer, text_encoder = zimage_utils.load_qwen3(args.text_encoder, dtype=te_dtype, device=device, disable_mmap=True)
    text_encoder.eval()

    # Encode with Qwen3 text encoder
    logger.info("Encoding prompts with Qwen3 text encoder")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal tokenizer, text_encoder
        encode_and_save_batch(tokenizer, text_encoder, batch, device)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )

    # Clean up
    del tokenizer, text_encoder
    clean_memory_on_device(device)

    if args.dopsd_cache_teacher_outputs:
        if args.dopsd_teacher_text_encoder is None:
            raise ValueError("--dopsd_teacher_text_encoder is required when --dopsd_cache_teacher_outputs is set")

        teacher_dtype = model_utils.str_to_dtype(args.dopsd_teacher_dtype)
        processor_path = args.dopsd_teacher_processor if args.dopsd_teacher_processor is not None else args.dopsd_teacher_text_encoder
        logger.info(f"Loading D-OPSD teacher processor from {processor_path}")
        teacher_processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=args.dopsd_teacher_trust_remote_code)
        logger.info(f"Loading D-OPSD teacher encoder from {args.dopsd_teacher_text_encoder}")
        teacher_encoder = _load_auto_vlm(
            args.dopsd_teacher_text_encoder,
            teacher_dtype,
            device,
            args.dopsd_teacher_trust_remote_code,
            args.dopsd_teacher_llm_reweight_source,
            args.dopsd_teacher_already_reweighted,
            args.dopsd_teacher_allow_raw_vlm,
        )

        logger.info("Encoding D-OPSD multimodal teacher outputs")

        def encode_for_dopsd_teacher(batch: list[ItemInfo]):
            nonlocal teacher_processor, teacher_encoder
            encode_and_save_dopsd_teacher_batch(
                teacher_processor,
                teacher_encoder,
                batch,
                device,
                args.dopsd_teacher_hidden_state_index,
                zimage_config.DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
                args.dopsd_teacher_embed_key,
            )

        cache_text_encoder_outputs.process_text_encoder_batches(
            args.num_workers,
            False,
            args.batch_size,
            datasets,
            all_cache_files_for_dataset,
            all_cache_paths_for_dataset,
            encode_for_dopsd_teacher,
            requires_content=True,
        )

        del teacher_processor, teacher_encoder
        clean_memory_on_device(device)

    # Remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )

    logger.info("Done!")


def zimage_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Z-Image specific arguments to the parser."""
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3 text encoder checkpoint path or directory",
    )
    parser.add_argument(
        "--fp8_llm",
        action="store_true",
        help="Use fp8 precision for Qwen3 text encoder (reduces memory usage)",
    )
    parser.add_argument(
        "--dopsd_cache_teacher_outputs",
        action="store_true",
        help="Also cache D-OPSD multimodal teacher embeddings into the Z-Image text encoder cache",
    )
    parser.add_argument(
        "--dopsd_teacher_text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-compatible teacher encoder path for D-OPSD cache generation",
    )
    parser.add_argument(
        "--dopsd_teacher_processor",
        type=str,
        default=None,
        help="Optional processor path for the D-OPSD teacher encoder; defaults to --dopsd_teacher_text_encoder",
    )
    parser.add_argument(
        "--dopsd_teacher_llm_reweight_source",
        type=str,
        default=None,
        help="Qwen3-4B text model path used to replace the D-OPSD VLM language-model weights",
    )
    parser.add_argument(
        "--dopsd_teacher_already_reweighted",
        action="store_true",
        help="Assert that --dopsd_teacher_text_encoder already contains Qwen3-4B-reweighted LLM weights",
    )
    parser.add_argument(
        "--dopsd_teacher_allow_raw_vlm",
        action="store_true",
        help="Allow raw VLM teacher cache generation for ablations; this is not paper-consistent for Z-Image",
    )
    parser.add_argument(
        "--dopsd_teacher_dtype",
        type=str,
        default="bfloat16",
        help="Dtype for the D-OPSD teacher encoder, e.g. bfloat16 or float16",
    )
    parser.add_argument(
        "--dopsd_teacher_trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the D-OPSD teacher encoder/processor",
    )
    parser.add_argument(
        "--dopsd_teacher_hidden_state_index",
        type=int,
        default=-2,
        help="Hidden-state layer index to cache from the D-OPSD teacher encoder",
    )
    parser.add_argument(
        "--dopsd_teacher_embed_key",
        type=str,
        default=DOPSD_TEACHER_EMBED_KEY,
        help="Cache key for D-OPSD teacher embeddings",
    )
    return parser


if __name__ == "__main__":
    main()

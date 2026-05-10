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

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_Z_IMAGE, ItemInfo, save_text_encoder_output_cache_z_image
from musubi_tuner.dopsd_cache_utils import (
    content_to_pil_image,
    load_auto_vlm,
    load_qwen3_vl_processor,
    qwen3_vl_processor_id_for_variant,
    validate_multimodal_inputs,
)
from musubi_tuner.dopsd_train_utils import DOPSD_TEACHER_EMBED_KEY
from musubi_tuner.hv_train_network import clean_memory_on_device
from musubi_tuner.utils import model_utils
from musubi_tuner.zimage import zimage_config, zimage_utils
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DOPSD_ZIMAGE_TEACHER_HIDDEN_STATE_INDEX = -2
DOPSD_ZIMAGE_TEACHER_QWEN_VARIANT = "4B"


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
    images = [content_to_pil_image(item.content) for item in batch]
    texts = [_apply_chat_template(processor, image, item.caption) for image, item in zip(images, batch)]

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    validate_multimodal_inputs(inputs)
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
        teacher_config_id = qwen3_vl_processor_id_for_variant(DOPSD_ZIMAGE_TEACHER_QWEN_VARIANT)
        teacher_processor = load_qwen3_vl_processor(DOPSD_ZIMAGE_TEACHER_QWEN_VARIANT)
        teacher_llm_reweight_source = (
            None if args.dopsd_teacher_already_reweighted or args.dopsd_teacher_allow_raw_vlm else args.text_encoder
        )
        logger.info(f"Loading D-OPSD teacher encoder from {args.dopsd_teacher_text_encoder}")
        teacher_encoder = load_auto_vlm(
            args.dopsd_teacher_text_encoder,
            teacher_dtype,
            device,
            teacher_llm_reweight_source,
            args.dopsd_teacher_already_reweighted,
            args.dopsd_teacher_allow_raw_vlm,
            "Z-Image",
            teacher_config_id,
        )

        logger.info("Encoding D-OPSD multimodal teacher outputs")

        def encode_for_dopsd_teacher(batch: list[ItemInfo]):
            nonlocal teacher_processor, teacher_encoder
            encode_and_save_dopsd_teacher_batch(
                teacher_processor,
                teacher_encoder,
                batch,
                device,
                DOPSD_ZIMAGE_TEACHER_HIDDEN_STATE_INDEX,
                zimage_config.DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
                DOPSD_TEACHER_EMBED_KEY,
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
        help=(
            "Qwen3-VL teacher encoder weights path or directory for D-OPSD cache generation; "
            "processor/tokenizer are loaded from the official Qwen3-VL-4B repo"
        ),
    )
    parser.add_argument(
        "--dopsd_teacher_already_reweighted",
        action="store_true",
        help="Assert that --dopsd_teacher_text_encoder already contains --text_encoder-reweighted LLM weights",
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
    return parser


if __name__ == "__main__":
    main()

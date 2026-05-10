import argparse

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_flux_2

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.dopsd_train_utils import DOPSD_FLUX2_TEACHER_EMBED_KEY
from musubi_tuner.utils import model_utils
from musubi_tuner.dopsd_cache_utils import (
    content_to_pil_image,
    load_auto_vlm,
    load_qwen3_vl_processor,
    qwen3_vl_processor_id_for_variant,
    validate_multimodal_inputs,
)
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_embedder: torch.nn.Module, batch: list[ItemInfo], device: torch.device, arch_full: str):
    prompts = [item.caption for item in batch]
    autocast_dtype = torch.bfloat16 if text_embedder.dtype.itemsize == 1 else text_embedder.dtype  # use bfloat16 for fp8 models
    with torch.autocast(device_type=device.type, dtype=autocast_dtype), torch.no_grad():
        ctx_vec = text_embedder(prompts)
        ctx_vec = ctx_vec.cpu()  # [1, 512, 15360]

    # save prompt cache
    for item, _ctx_vec in zip(batch, ctx_vec):
        save_text_encoder_output_cache_flux_2(item, _ctx_vec, arch_full=arch_full)


def _apply_flux2_teacher_chat_template(processor, image, caption: str) -> str:
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
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_and_save_dopsd_teacher_batch(
    processor,
    teacher_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    expected_dim: int,
    teacher_embed_key: str,
    arch_full: str,
):
    images = [content_to_pil_image(item.content) for item in batch]
    texts = [_apply_flux2_teacher_chat_template(processor, image, item.caption) for image, item in zip(images, batch)]

    inputs = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True,
        max_length=flux2_utils.MAX_LENGTH,
        return_tensors="pt",
    )
    validate_multimodal_inputs(inputs)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = teacher_encoder(**inputs, output_hidden_states=True, use_cache=False)

    ctx_vec = torch.cat([outputs.hidden_states[layer] for layer in flux2_utils.OUTPUT_LAYERS_QWEN3], dim=-1)
    if ctx_vec.shape[-1] != expected_dim:
        raise ValueError(
            f"D-OPSD FLUX.2 teacher ctx dim {ctx_vec.shape[-1]} does not match model context dim {expected_dim}. "
            "Use a Qwen3-VL teacher whose language hidden size matches this Klein variant."
        )

    for item, ctx_i in zip(batch, ctx_vec):
        save_text_encoder_output_cache_flux_2(
            item,
            arch_full=arch_full,
            dopsd_teacher_ctx_vec=ctx_i.detach().cpu(),
            dopsd_teacher_key=teacher_embed_key,
        )


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = flux_2_setup_parser(parser)

    args = parser.parse_args()
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=model_version_info.architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load Mistral 3 or Qwen-3 text encoder
    m3_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
    text_embedder = flux2_utils.load_text_embedder(
        model_version_info, args.text_encoder, dtype=m3_dtype, device=device, disable_mmap=True
    )

    # Encode with Mistral 3 or Qwen-3 text encoder
    logger.info("Encoding with text encoder")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal text_embedder
        encode_and_save_batch(text_embedder, batch, device, model_version_info.architecture_full)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_embedder

    if args.dopsd_cache_teacher_outputs:
        if args.model_version not in {"klein-4b", "klein-9b"}:
            raise ValueError("--dopsd_cache_teacher_outputs for FLUX.2 only supports klein-4b and klein-9b")
        if args.dopsd_teacher_text_encoder is None:
            raise ValueError("--dopsd_teacher_text_encoder is required when --dopsd_cache_teacher_outputs is set")

        teacher_dtype = model_utils.str_to_dtype(args.dopsd_teacher_dtype)
        teacher_config_id = qwen3_vl_processor_id_for_variant(model_version_info.qwen_variant)
        teacher_processor = load_qwen3_vl_processor(model_version_info.qwen_variant)
        teacher_llm_reweight_source = (
            None if args.dopsd_teacher_already_reweighted or args.dopsd_teacher_allow_raw_vlm else args.text_encoder
        )
        logger.info(f"Loading D-OPSD FLUX.2 teacher encoder from {args.dopsd_teacher_text_encoder}")
        teacher_encoder = load_auto_vlm(
            args.dopsd_teacher_text_encoder,
            teacher_dtype,
            device,
            teacher_llm_reweight_source,
            args.dopsd_teacher_already_reweighted,
            args.dopsd_teacher_allow_raw_vlm,
            "FLUX.2 Klein",
            teacher_config_id,
        )

        logger.info("Encoding D-OPSD FLUX.2 multimodal teacher outputs")

        def encode_for_dopsd_teacher(batch: list[ItemInfo]):
            nonlocal teacher_processor, teacher_encoder
            encode_and_save_dopsd_teacher_batch(
                teacher_processor,
                teacher_encoder,
                batch,
                device,
                model_version_info.params.context_in_dim,
                DOPSD_FLUX2_TEACHER_EMBED_KEY,
                model_version_info.architecture_full,
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

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def flux_2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="text encoder (mistral 3) checkpoint path")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--dopsd_cache_teacher_outputs",
        action="store_true",
        help="Also cache D-OPSD multimodal teacher ctx vectors into the FLUX.2 text encoder cache",
    )
    parser.add_argument(
        "--dopsd_teacher_text_encoder",
        type=str,
        default=None,
        help=(
            "Qwen3-VL teacher encoder weights path or directory for D-OPSD cache generation; "
            "processor/tokenizer are loaded from the matching official Qwen3-VL repo"
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
        help="Allow raw VLM teacher cache generation for ablations; this is not paper-consistent for FLUX.2 Klein",
    )
    parser.add_argument(
        "--dopsd_teacher_dtype",
        type=str,
        default="bfloat16",
        help="Dtype for the D-OPSD teacher encoder, e.g. bfloat16 or float16",
    )
    flux2_utils.add_model_version_args(parser)
    return parser


if __name__ == "__main__":
    main()

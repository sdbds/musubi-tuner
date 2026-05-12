import argparse

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_flux_2

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.dopsd_train_utils import DOPSD_FLUX2_IDENTITY_EDIT_PROMPT, DOPSD_FLUX2_TEACHER_EMBED_KEY
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


def encode_and_save_dopsd_teacher_batch(
    text_embedder: torch.nn.Module,
    batch: list[ItemInfo],
    device: torch.device,
    expected_dim: int,
    teacher_embed_key: str,
    arch_full: str,
):
    prompts = [DOPSD_FLUX2_IDENTITY_EDIT_PROMPT for _ in batch]
    autocast_dtype = torch.bfloat16 if text_embedder.dtype.itemsize == 1 else text_embedder.dtype
    with torch.autocast(device_type=device.type, dtype=autocast_dtype), torch.no_grad():
        ctx_vec = text_embedder(prompts).cpu()
    if ctx_vec.shape[-1] != expected_dim:
        raise ValueError(
            f"D-OPSD FLUX.2 teacher ctx dim {ctx_vec.shape[-1]} does not match model context dim {expected_dim}. "
            "Use the matching Qwen3 text encoder for this Klein variant."
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
    if args.dopsd_cache_teacher_outputs and args.model_version not in {"klein-4b", "klein-9b"}:
        raise ValueError("--dopsd_cache_teacher_outputs for FLUX.2 only supports klein-4b and klein-9b")

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

    if args.dopsd_cache_teacher_outputs:
        logger.info("Encoding D-OPSD FLUX.2 identity-edit teacher context")

        def encode_for_dopsd_teacher(batch: list[ItemInfo]):
            nonlocal text_embedder
            encode_and_save_dopsd_teacher_batch(
                text_embedder,
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
        )

    del text_embedder

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
        help="Also cache the FLUX.2 D-OPSD identity-edit teacher ctx vector into the text encoder cache",
    )
    flux2_utils.add_model_version_args(parser)
    return parser


if __name__ == "__main__":
    main()

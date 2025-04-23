import argparse
import logging
import math
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SiglipImageProcessor, SiglipVisionModel

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache_framepack, ARCHITECTURE_FRAMEPACK
from frame_pack import hunyuan
from frame_pack.framepack_utils import load_image_encoders, load_vae
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from frame_pack.clip_vision import hf_clip_vision_encode
import cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    vae: AutoencoderKLCausal3D,
    feature_extractor: SiglipImageProcessor,
    image_encoder: SiglipVisionModel,
    batch: List[ItemInfo],
    latent_window_size: int,
):
    """Encode a batch of original RGB videos and save FramePack section caches."""

    # Stack batch into tensor (B,C,F,H,W) in RGB order
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    height, width = contents.shape[3], contents.shape[4]
    if height < 8 or width < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    latent_f = (batch[0].frame_count - 1) // 4 + 1
    total_latent_sections = math.floor((latent_f - 1) / latent_window_size)
    if total_latent_sections < 1:
        raise ValueError(f"Not enough frames for FramePack: {batch[0].frame_count}, minimum: {latent_window_size*4+1}")

    latent_f = total_latent_sections * latent_window_size + 1
    frame_count = (latent_f - 1) * 4 + 1
    if frame_count != batch[0].frame_count:
        logger.info(f"Frame count mismatch: {frame_count} != {batch[0].frame_count}, trimming to {frame_count}")
        contents = contents[:, :, :frame_count, :, :]

    latent_paddings = list(reversed(range(total_latent_sections)))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

    # VAE encode (list of tensor -> stack)
    latents = hunyuan.vae_encode(contents, vae)  # include scaling factor
    latents = latents.to("cpu")

    # Vision encoding per‑item (once)
    images = np.stack([item.content[0] for item in batch], axis=0)  # B, H, W, C

    # encode image with image encoder
    image_embeddings = []
    with torch.no_grad():
        for image in images:
            image_encoder_output = hf_clip_vision_encode(image, feature_extractor, image_encoder)
            image_embeddings.append(image_encoder_output.last_hidden_state)
    image_embeddings = torch.cat(image_embeddings, dim=0)  # B, LEN, 1152

    for b, item in enumerate(batch):
        original_latent_cache_path = item.latent_cache_path
        video_lat = latents[b : b + 1]  # keep batch dim, B, C, F, H, W

        # emulate inferece step
        history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=video_lat.dtype)
        latent_f_index = latent_f - latent_window_size
        section_index = total_latent_sections - 1
        for latent_padding in latent_paddings:
            is_last_section = section_index == 0
            latent_padding_size = latent_padding * latent_window_size

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = video_lat[:, :, 0:1, :, :].to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, : 1 + 2 + 16, :, :].split(
                [1, 2, 16], dim=2
            )
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            target_latents = video_lat[:, :, latent_f_index : latent_f_index + latent_window_size, :, :]

            # save cache (file path is inside item.latent_cache_path pattern), remove batch dim
            item.latent_cache_path = append_section_idx_to_latent_cache_path(original_latent_cache_path, section_index)
            save_latent_cache_framepack(
                item_info=item,
                latent=target_latents.squeeze(0),
                latent_indices=latent_indices.squeeze(0),
                clean_latents=clean_latents.squeeze(0),
                clean_latent_indices=clean_latent_indices.squeeze(0),
                clean_latents_2x=clean_latents_2x.squeeze(0),
                clean_latent_2x_indices=clean_latent_2x_indices.squeeze(0),
                clean_latents_4x=clean_latents_4x.squeeze(0),
                clean_latent_4x_indices=clean_latent_4x_indices.squeeze(0),
                image_embeddings=image_embeddings[b],
            )

            section_index -= 1
            latent_f_index -= latent_window_size

        if is_last_section:
            generated_latents = video_lat[:, :, : latent_window_size + 1, :, :]
        else:
            generated_latents = target_latents

        history_latents = torch.cat([generated_latents, history_latents], dim=2)


def framepack_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--image_encoder", type=str, required=True, help="Image encoder (CLIP) checkpoint path or directory")
    parser.add_argument("--latent_window_size", type=int, default=9, help="FramePack latent window size (default 9)")
    return parser


def main(args: argparse.Namespace):
    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FRAMEPACK)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "vae checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device=device)
    vae.to(device)

    logger.info(f"Loading image encoder from {args.image_encoder}")
    feature_extractor, image_encoder = load_image_encoders(args)
    image_encoder.eval()
    image_encoder.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, feature_extractor, image_encoder, batch, args.latent_window_size)

    # reuse core loop from cache_latents with no change
    encode_datasets_framepack(datasets, encode, args)


def append_section_idx_to_latent_cache_path(latent_cache_path: str, section_idx: int) -> str:
    tokens = latent_cache_path.split("_")
    tokens[-3] = f"{tokens[-3]}-{section_idx:04d}"  # append section index to "frame_pos-count"
    return "_".join(tokens)


def encode_datasets_framepack(datasets: list[BaseDataset], encode: callable, args: argparse.Namespace):
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_latent_cache_paths = []
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            batch: list[ItemInfo] = batch  # type: ignore

            # latent_cache_path is "{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors"
            # we expand it to "{basename}_{section_idx:04d}_{w:04d}x{h:04d}_{self.architecture}.safetensors"
            filtered_batch = []
            for item in batch:
                latent_f = (item.frame_count - 1) // 4 + 1
                num_sections = math.floor((latent_f - 1) / args.latent_window_size)
                all_existing = True
                for sec in range(num_sections):
                    p = append_section_idx_to_latent_cache_path(item.latent_cache_path, sec)
                    all_latent_cache_paths.append(p)
                    all_existing = all_existing and os.path.exists(p)

                if all_existing:
                    filtered_batch.append(item)

            if args.skip_existing:
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode(batch[i : i + bs])

        # normalize paths
        all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
        all_latent_cache_paths = set(all_latent_cache_paths)

        # remove old cache files not in the dataset
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_latent_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


if __name__ == "__main__":
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    parser = framepack_setup_parser(parser)

    args = parser.parse_args()

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in FramePack")
    # if args.batch_size != 1:
    #     args.batch_size = 1
    #     logger.info("Batch size is set to 1 for FramePack.")

    main(args)

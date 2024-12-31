import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from PIL import Image

import logging

from dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache
from hunyuan_model.vae import load_vae
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_image(image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]]) -> int:
    import cv2

    imgs = [image] if not isinstance(image, list) else [image[0], image[-1]]
    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"{'First' if i == 0 else 'Last'} image: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        cv2_img = np.array(img) if isinstance(img, Image.Image) else img
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv2_img)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
    return k


def show_console(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],
    width: int,
    back: str,
    interactive: bool = False,
) -> int:
    from ascii_magic import from_pillow_image, Back

    back = None
    if back is not None:
        back = getattr(Back, back.upper())

    k = None
    imgs = [image] if not isinstance(image, list) else image
    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"{'First' if i == 0 else 'Last'} image: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        img = image if not isinstance(image, list) else image[0]
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        ascii_img = from_pillow_image(pil_img)
        ascii_img.to_terminal(columns=width, back=back)

        if interactive:
            k = input("Press q to quit, d to next dataset, other key to next: ")
            if k == "q" or k == "d":
                return ord(k)

    if not interactive:
        return ord(" ")
    return ord(k) if k else ord(" ")


def show_datasets(
    datasets: list[BaseDataset], debug_mode: str, console_width: int, console_back: str, console_num_images: Optional[int]
):
    print(f"d: next dataset, q: quit")

    num_workers = max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        print(f"Dataset [{i}]")
        batch_index = 0
        num_images_to_show = console_num_images
        k = None
        for key, batch in dataset.retrieve_latent_cache_batches(num_workers):
            print(f"bucket resolution: {key}, count: {len(batch)}")
            for j, item_info in enumerate(batch):
                item_info: ItemInfo
                print(f"{batch_index}-{j}: {item_info}")
                if debug_mode == "image":
                    k = show_image(item_info.content)
                elif debug_mode == "console":
                    k = show_console(item_info.content, console_width, console_back, console_num_images is None)
                    if num_images_to_show is not None:
                        num_images_to_show -= 1
                        if num_images_to_show == 0:
                            k = ord("d")  # next dataset

                if k == ord("q"):
                    return
                elif k == ord("d"):
                    break
            if k == ord("d"):
                break
            batch_index += 1


def encode_and_save_batch(vae: AutoencoderKLCausal3D, batch: list[ItemInfo]):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    # print(f"encode batch: {contents.shape}")
    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()
        latent = latent * vae.config.scaling_factor

    # # debug: decode and save
    # with torch.no_grad():
    #     latent_to_decode = latent / vae.config.scaling_factor
    #     images = vae.decode(latent_to_decode, return_dict=False)[0]
    #     images = (images / 2 + 0.5).clamp(0, 1)
    #     images = images.cpu().float().numpy()
    #     images = (images * 255).astype(np.uint8)
    #     images = images.transpose(0, 2, 3, 4, 1)  # B, C, F, H, W -> B, F, H, W, C
    #     for b in range(images.shape[0]):
    #         for f in range(images.shape[1]):
    #             fln = os.path.splitext(os.path.basename(batch[b].item_key))[0]
    #             img = Image.fromarray(images[b, f])
    #             img.save(f"./logs/decode_{fln}_{b}_{f:03d}.jpg")

    for item, l in zip(batch, latent):
        # print(f"save latent cache: {item.latent_cache_path}, latent shape: {l.shape}")
        save_latent_cache(item, l)


def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "vae checkpoint is required"

    # Load VAE model: HunyuanVideo VAE model is float16
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    print(f"Loaded VAE: {vae.config}, dtype: {vae.dtype}")

    if args.vae_chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
        logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    elif args.vae_tiling:
        vae.enable_spatial_tiling(True)

    # Encode images
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        print(f"Encoding dataset [{i}]")
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            if args.skip_existing:
                filtered_batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode_and_save_batch(vae, batch[i : i + bs])


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--vae", type=str, required=False, default=None, help="path to vae checkpoint")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is float16")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--debug_mode", type=str, default=None, choices=["image", "console"], help="debug mode")
    parser.add_argument("--console_width", type=int, default=80, help="debug mode: console width")
    parser.add_argument(
        "--console_back", type=str, default=None, help="debug mode: console background color, one of ascii_magic.Back"
    )
    parser.add_argument(
        "--console_num_images",
        type=int,
        default=None,
        help="debug mode: not interactive, number of images to show for each dataset",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
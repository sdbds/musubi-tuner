"""Cache normalized MOSS-Image FLUX.2 VAE latents."""

import logging

import numpy as np
import torch

from musubi_tuner import cache_latents
from musubi_tuner.moss_image import cache
from musubi_tuner.moss_image.vae import load_vae

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_and_save_batch(vae, batch):
    contents = []
    for item in batch:
        if item.control_content:
            raise ValueError("MOSS-Image does not support control/reference images")
        content = item.content
        if isinstance(content, list):
            if len(content) != 1:
                raise ValueError("MOSS-Image requires one image per item")
            content = content[0]
        contents.append(torch.from_numpy(np.asarray(content)[..., :3].copy()))
    pixels = (torch.stack(contents).permute(0, 3, 1, 2).to(device=vae.device, dtype=torch.float32) / 127.5 - 1.0).contiguous()
    if pixels.shape[-1] % 16 or pixels.shape[-2] % 16:
        raise ValueError("MOSS image dimensions must be divisible by 16")
    with torch.autocast(device_type=vae.device.type, enabled=False):
        latents = vae.encode(pixels)
    for item, latent in zip(batch, latents):
        cache.save_latents(item, latent.float())


def setup_parser():
    return cache_latents.setup_parser_common()


def main():
    logging.basicConfig(level=logging.INFO)
    args = setup_parser().parse_args()
    datasets = cache.load_datasets(args)
    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return
    if not args.vae:
        raise ValueError("--vae is required")
    if args.vae_dtype not in (None, "float32", "fp32"):
        raise ValueError("MOSS-Image VAE caching requires FP32")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False
    vae = load_vae(args.vae, dtype=torch.float32, device=device).eval().requires_grad_(False)
    cache_latents.encode_datasets(
        datasets,
        lambda batch: encode_and_save_batch(vae, batch),
        args,
        cache_is_current=lambda item: cache.cache_is_current(item, "latent"),
    )


if __name__ == "__main__":
    main()

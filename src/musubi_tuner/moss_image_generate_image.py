"""Standalone MOSS-Image generation from base/full checkpoints and optional DiT LoRA."""

import argparse
import gc
import logging
import math
from pathlib import Path
import time

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
from safetensors.torch import load_file

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.moss_image import cache, sampling, utils
from musubi_tuner.moss_image.vae import load_vae
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device


@torch.no_grad()
def generate_image(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = model_utils.str_to_dtype(args.dtype)
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("MOSS dimensions must be positive multiples of 16")
    if args.infer_steps < 1 or not math.isfinite(args.guidance_scale) or args.flow_shift <= 0:
        raise ValueError("Invalid sampling steps, guidance or shift")
    if args.blocks_to_swap < 0 or args.blocks_to_swap > 40:
        raise ValueError("blocks_to_swap must be between 0 and 40")
    if args.blocks_to_swap and device.type != "cuda":
        raise ValueError("MOSS block swapping requires CUDA")
    if args.lora_multiplier and len(args.lora_multiplier) != len(args.lora_weight or []):
        raise ValueError("Provide one lora_multiplier per lora_weight")
    if args.text_cache:
        if args.prompt is not None or args.negative_prompt:
            raise ValueError("text_cache already contains its prompt and empty negative prompt; remove prompt overrides")
        data = cache.load_cache(args.text_cache, "text")
        positive, mask = utils.pad_text_embeds([data[cache.TEXT_KEYS[0]]])
        negative, negative_mask = utils.pad_text_embeds([data[cache.TEXT_KEYS[1]]])
    else:
        if not args.text_encoder or args.prompt is None:
            raise ValueError("Provide --text_encoder and --prompt, or --text_cache")
        tokenizer = utils.load_tokenizer(args.tokenizer_path)
        encoder = utils.load_text_encoder(
            args.text_encoder, device=device, dtype=torch.bfloat16, disable_mmap=args.disable_numpy_memmap
        )
        rows = utils.encode_tokens(
            encoder, utils.tokenize_prompts(tokenizer, [args.prompt, args.negative_prompt], args.moss_max_token_length), device
        )
        positive, mask = utils.pad_text_embeds([rows[0].cpu()])
        negative, negative_mask = utils.pad_text_embeds([rows[1].cpu()])
        del encoder, tokenizer, rows
        clean_memory_on_device(device)
    loras = [load_file(path) for path in args.lora_weight or []]
    model = (
        utils.load_moss_image_model(
            device,
            args.dit,
            args.attn_mode,
            args.split_attn,
            "cpu" if args.blocks_to_swap else device,
            dtype,
            lora_weights_list=loras,
            lora_multipliers=args.lora_multiplier,
            disable_mmap=args.disable_numpy_memmap,
        )
        .eval()
        .requires_grad_(False)
    )
    del loras
    if args.blocks_to_swap:
        model.enable_block_swap(args.blocks_to_swap, BlockSwapConfig(device, supports_backward=False))
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    if args.compile:
        model_utils.compile_transformer(args, model, [model.layers], disable_linear=bool(args.blocks_to_swap))
    latents = sampling.generate_latents(
        model,
        positive,
        mask,
        negative,
        negative_mask,
        args.height,
        args.width,
        args.infer_steps,
        args.guidance_scale,
        args.flow_shift,
        sampling.parse_skip_layers(args.uncond_skip_layers),
        args.seed,
        device,
    ).cpu()
    model.wait_for_pending_block_moves()
    del model
    gc.collect()
    clean_memory_on_device(device)
    vae = load_vae(args.vae, torch.float32, device, disable_mmap=args.disable_numpy_memmap).eval()
    with torch.autocast(device_type=device.type, enabled=False):
        pixels = (vae.decode(latents.to(device)).float() / 2 + 0.5).clamp(0, 1)
    array = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    directory = Path(args.output_dir)
    name = args.output_name or f"moss_{time.strftime('%Y%m%d_%H%M%S')}_{args.seed}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("output_name must be a filename, not a path")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.lower().endswith(".png") else name + ".png")
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {path}; use --overwrite to replace it")
    metadata = PngInfo()
    metadata.add_text("prompt", args.prompt or "(cached prompt)")
    metadata.add_text("seed", str(args.seed))
    metadata.add_text("model", Path(args.dit).name)
    Image.fromarray(array).save(path, pnginfo=metadata)
    logging.getLogger(__name__).info("Saved %s", path)
    return path


def setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True, help="MOSS DiT first shard or full-model checkpoint")
    parser.add_argument("--vae", required=True, help="FLUX.2 VAE checkpoint")
    parser.add_argument("--text_encoder", default=None)
    parser.add_argument("--tokenizer_path", "--tokenizer_cache_dir", default=None)
    parser.add_argument("--moss_max_token_length", type=int, default=4096)
    parser.add_argument("--text_cache", default=None, help="Native MOSS text cache; replaces text encoder and prompt arguments")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--infer_steps", "--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument(
        "--uncond_skip_layers", default="22", help="Comma-separated main-block indices; empty string disables skipping"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "bf16", "float32", "fp32"], default="bfloat16")
    parser.add_argument("--attn_mode", choices=["torch", "flash", "flash3", "xformers", "sageattn"], default="torch")
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--disable_numpy_memmap", action="store_true")
    parser.add_argument("--lora_weight", nargs="*", default=None)
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    model_utils.setup_parser_compile(parser)
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    generate_image(setup_parser().parse_args())


if __name__ == "__main__":
    main()

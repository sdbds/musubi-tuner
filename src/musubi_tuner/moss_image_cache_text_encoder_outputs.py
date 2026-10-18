"""Cache unpadded BF16 MOSS-Image prompt and unconditional features."""

import logging

import torch

from musubi_tuner import cache_text_encoder_outputs as common
from musubi_tuner.moss_image import cache, utils


@torch.no_grad()
def encode_and_save_batch(tokenizer, text_encoder, batch, device, uncond, max_length=utils.MAX_TOKEN_LENGTH):
    tokens = utils.tokenize_prompts(tokenizer, [item.caption for item in batch], max_length)
    rows = utils.encode_tokens(text_encoder, tokens, device)
    for item, row in zip(batch, rows):
        cache.save_text(item, row, uncond, max_length=max_length)


def setup_parser():
    parser = common.setup_parser_common()
    parser.add_argument("--text_encoder", required=True, help="Qwen3.5-9B first shard or single-file checkpoint")
    parser.add_argument("--tokenizer_path", "--tokenizer_cache_dir", default=None)
    parser.add_argument("--moss_max_token_length", type=int, default=utils.MAX_TOKEN_LENGTH)
    parser.add_argument("--disable_numpy_memmap", action="store_true")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = setup_parser().parse_args()
    if not 1 <= args.moss_max_token_length <= utils.MAX_TOKEN_LENGTH:
        raise ValueError("moss_max_token_length must be between 1 and 4096")
    datasets = cache.load_datasets(args)
    files, paths = common.prepare_cache_files_and_paths(datasets)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = utils.load_tokenizer(args.tokenizer_path)
    text_encoder = utils.load_text_encoder(
        args.text_encoder, device=device, dtype=torch.bfloat16, disable_mmap=args.disable_numpy_memmap
    ).requires_grad_(False)
    with torch.no_grad():
        uncond = utils.encode_tokens(text_encoder, utils.tokenize_prompts(tokenizer, [""], args.moss_max_token_length), device)[0]
    common.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        files,
        paths,
        lambda batch: encode_and_save_batch(tokenizer, text_encoder, batch, device, uncond, args.moss_max_token_length),
        cache_is_current=lambda item: cache.cache_is_current(item, "text", args.moss_max_token_length),
    )
    common.post_process_cache_files(datasets, files, paths, args.keep_cache)


if __name__ == "__main__":
    main()

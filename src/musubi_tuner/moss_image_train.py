"""Full DiT fine-tuning for MOSS-Image; external text encoder and VAE stay frozen."""

import logging

import torch

from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_image_setup_parser
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common


class MossImageTrainer(MossImageNetworkTrainer):
    full_finetune = True

    def handle_model_specific_args(self, args):
        super().handle_model_specific_args(args)
        if args.block_swap_h2d_only:
            raise ValueError("H2D-only swapping requires frozen base weights and cannot be used for full fine-tuning")
        lora_options = {
            name: getattr(args, name)
            for name in (
                "network_module",
                "network_dim",
                "network_weights",
                "network_args",
                "network_dropout",
                "dim_from_weights",
                "scale_weight_norms",
                "base_weights",
                "base_weights_multiplier",
            )
        }
        if args.network_alpha != 1:
            lora_options["network_alpha"] = args.network_alpha
        if any(value not in (None, False, []) for value in lora_options.values()):
            raise ValueError("MOSS full fine-tuning does not accept LoRA/network arguments")
        if args.full_bf16 and args.mixed_precision != "bf16":
            raise ValueError("full_bf16 requires mixed_precision bf16")
        if args.fused_backward_pass:
            if args.optimizer_type.lower() != "adafactor" or args.gradient_accumulation_steps != 1 or args.max_grad_norm != 0:
                raise ValueError("fused_backward_pass requires Adafactor, gradient_accumulation_steps 1 and max_grad_norm 0")
        if args.blocks_to_swap:
            if args.gradient_accumulation_steps != 1:
                raise ValueError("MOSS full-model block swap currently requires gradient_accumulation_steps 1")
            if not args.fused_backward_pass and not args.block_swap_optimizer_patch_params:
                raise ValueError("Full-model block swap requires block_swap_optimizer_patch_params or fused_backward_pass")
            if args.optimizer_type.lower() not in ("", "adamw", "torch.optim.adamw", "adafactor"):
                raise ValueError("Full-model block swap supports AdamW and Adafactor only")
        self.dit_dtype = torch.bfloat16 if args.full_bf16 else torch.float32
        args.dit_dtype = "bfloat16" if args.full_bf16 else "float32"


def setup_parser():
    parser = moss_image_setup_parser(setup_parser_common())
    parser.add_argument("--full_bf16", action="store_true", help="Train DiT parameters in BF16 instead of FP32")
    parser.add_argument("--fused_backward_pass", action="store_true", help="Update Adafactor parameters during backward")
    parser.add_argument(
        "--block_swap_optimizer_patch_params",
        action="store_true",
        help="Synchronize full-model swap gradients before optimizer updates",
    )
    parser.add_argument("--mem_eff_save", action="store_true", help="Full checkpoints always use streaming saving")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    parser = setup_parser()
    MossImageTrainer().train(read_config_from_file(parser.parse_args(), parser))


if __name__ == "__main__":
    main()

"""MOSS-Image DiT LoRA training with frozen, cached text conditioning."""

import logging
import math

import torch
from safetensors import safe_open

from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE, ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.moss_image import cache, sampling, utils
from musubi_tuner.moss_image.vae import load_vae
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.timesteps import BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


class MossImageNetworkTrainer(NetworkTrainer):
    full_finetune = False

    def train(self, args):
        from musubi_tuner.moss_image.training import train

        return train(self, args, self.full_finetune)

    @property
    def architecture(self):
        return ARCHITECTURE_MOSS_IMAGE

    @property
    def architecture_full_name(self):
        return ARCHITECTURE_MOSS_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 3.0
        self.default_discrete_flow_shift = 5.0
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("MOSS-Image does not support FP8 weights")
        if args.mixed_precision == "fp16":
            raise ValueError("MOSS-Image supports bf16 or no mixed precision, not fp16")
        if args.vae_dtype not in (None, "float32", "fp32"):
            raise ValueError("MOSS-Image VAE requires FP32")
        if args.log_grad_metrics or args.show_timesteps:
            raise ValueError("MOSS does not support log_grad_metrics or show_timesteps")
        if args.gradient_checkpointing_cpu_offload and not args.gradient_checkpointing:
            raise ValueError("gradient_checkpointing_cpu_offload requires gradient_checkpointing")
        args.vae_dtype = "float32"
        self.dit_dtype = torch.float32 if args.mixed_precision == "no" else torch.bfloat16
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        if not 0 <= args.caption_dropout_rate <= 1:
            raise ValueError("caption_dropout_rate must be in [0, 1]")
        if not 1 <= args.moss_max_token_length <= 4096:
            raise ValueError("moss_max_token_length must be in [1, 4096]")
        if not math.isfinite(args.discrete_flow_shift) or args.discrete_flow_shift <= 0:
            raise ValueError("discrete_flow_shift must be finite and positive")
        if args.blocks_to_swap and not 0 <= args.blocks_to_swap <= 40:
            raise ValueError("MOSS blocks_to_swap must be between 0 and 40")
        if args.blocks_to_swap and args.gradient_checkpointing_cpu_offload:
            raise ValueError("MOSS block swap cannot be combined with CPU activation offloading")
        if args.compile and args.compile_fullgraph and args.split_attn:
            raise ValueError("compile_fullgraph cannot be used with split_attn")
        if args.sample_prompts and (not args.text_encoder or not args.vae):
            raise ValueError("MOSS sampling requires --text_encoder and --vae")

    def _build_dataset(self, args):
        group, collator, epoch = super()._build_dataset(args)
        cache.validate_datasets(group.datasets)
        return group, collator, epoch

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype, train_dataset_group):
        if args.network_weights and not args.network_weights.lower().endswith(".safetensors"):
            raise ValueError("MOSS adapter weights must use safetensors")
        network = super()._build_network(args, accelerator, transformer, vae, weight_dtype, train_dataset_group)
        if args.network_weights:
            with safe_open(args.network_weights, framework="pt", device="cpu") as handle:
                unhandled = set(handle.keys()) - network.state_dict().keys()
            if unhandled:
                raise ValueError(
                    f"MOSS network_weights contains unhandled keys: {sorted(unhandled)[:5]}. "
                    "Use matching network_args or dim_from_weights; external text encoder LoRA is unsupported."
                )
        return network

    def load_transformer(self, accelerator, args, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype):
        return utils.load_moss_image_model(
            accelerator.device,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype,
            disable_mmap=args.disable_numpy_memmap,
        )

    def load_vae(self, args, vae_dtype, vae_path):
        return load_vae(vae_path, dtype=torch.float32, device="cpu", disable_mmap=args.disable_numpy_memmap).eval()

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(args, transformer, [transformer.layers], disable_linear=self.blocks_to_swap > 0)

    def scale_shift_latents(self, latents):
        if latents.ndim != 4 or latents.shape[1] != 128 or latents.dtype != torch.float32:
            raise ValueError("MOSS training requires cached FP32 normalized [B, 128, H, W] latents")
        return latents

    def latent_tokens_for_timestep_sampling(self, latents):
        return latents.shape[-2] * latents.shape[-1]

    def get_noise_scheduler(self, args):
        scheduler = getattr(self, "_moss_noise_scheduler", None)
        if scheduler is None or scheduler.config.shift != args.discrete_flow_shift:
            scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
            scheduler.set_timesteps(scheduler.config.num_train_timesteps)
            self._moss_noise_scheduler = scheduler
        return scheduler

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        if args.timestep_sampling in BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS:
            sigma = self.sample_timesteps(args, noise.shape[0], timesteps, latents, device).float()
            times = sigma * 1000.0
            sigma = sigma.reshape(-1, *([1] * (latents.ndim - 1)))
            noisy = (1 - sigma) * latents.float() + sigma * noise.float()
        else:
            noisy, times = super().get_noisy_model_input_and_timesteps(
                args, noise.float(), latents.float(), timesteps, self.get_noise_scheduler(args), device, torch.float32
            )
        return noisy.float(), times.float()

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ):
        context, mask = cache.prepare_conditioning(
            batch["moss_embed"], batch["moss_uncond"], args.caption_dropout_rate, accelerator.device
        )
        prediction = transformer(noisy_model_input.to(accelerator.device), timesteps.float(), context, mask)
        return DiTOutput(prediction, noise.to(accelerator.device).float() - latents.to(accelerator.device).float())

    def compute_per_sample_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
        if output.pred.shape != output.target.shape:
            raise ValueError("MOSS prediction and target shapes differ")
        losses = (output.pred.float() - output.target.float()).square().flatten(1).mean(1)
        sigmas = timesteps.float() / 1000.0
        if args.weighting_scheme == "sigma_sqrt":
            if torch.any(sigmas <= 0):
                raise ValueError("sigma_sqrt weighting requires strictly positive sigmas")
            losses = losses * sigmas.reciprocal().square()
        elif args.weighting_scheme == "cosmap":
            losses = losses * (2 / (math.pi * (1 - 2 * sigmas + 2 * sigmas.square())))
        return losses

    @torch.no_grad()
    def process_sample_prompts(self, args, accelerator, sample_prompts):
        prompts = load_prompts(sample_prompts)
        tokenizer = utils.load_tokenizer(args.tokenizer_path)
        text_encoder = utils.load_text_encoder(
            args.text_encoder, device=accelerator.device, dtype=torch.bfloat16, disable_mmap=args.disable_numpy_memmap
        )
        encoded = {}
        result = []
        for prompt in prompts:
            row = dict(prompt)
            row.setdefault("negative_prompt", "")
            row.setdefault("sample_steps", 50)
            row.setdefault("width", 1024)
            row.setdefault("height", 1024)
            for name, key in (("prompt", "moss_embed"), ("negative_prompt", "moss_negative")):
                text = row.get(name) or ""
                if text not in encoded:
                    ids = utils.tokenize_prompts(tokenizer, [text], args.moss_max_token_length)
                    encoded[text] = utils.encode_tokens(text_encoder, ids, accelerator.device)[0].cpu()
                row[key] = encoded[text]
            result.append(row)
        del text_encoder, tokenizer
        clean_memory_on_device(accelerator.device)
        return result

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        was_training = transformer.training
        try:
            return super().sample_image_inference(
                accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
            )
        finally:
            transformer.train(was_training)
            vae.to("cpu")

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        if frame_count != 1 or image_path or control_video_path:
            raise ValueError("MOSS-Image supports text-to-image generation only")
        positive, mask = utils.pad_text_embeds([sample_parameter["moss_embed"]])
        negative, negative_mask = utils.pad_text_embeds([sample_parameter["moss_negative"]])
        latents = sampling.generate_latents(
            transformer,
            positive,
            mask,
            negative,
            negative_mask,
            height,
            width,
            sample_steps,
            guidance_scale if cfg_scale is None else cfg_scale,
            discrete_flow_shift,
            sampling.parse_skip_layers(args.sample_uncond_skip_layers),
            generator.initial_seed(),
            accelerator.device,
        )
        vae.to(accelerator.device).eval()
        try:
            with torch.no_grad(), torch.autocast(device_type=accelerator.device.type, enabled=False):
                pixels = vae.decode(latents.float())
            return (pixels.float().cpu() / 2 + 0.5).clamp(0, 1).unsqueeze(2)
        finally:
            vae.to("cpu")


def moss_image_setup_parser(parser):
    parser.add_argument("--text_encoder", default=None, help="Qwen3.5-9B checkpoint (needed for sampling)")
    parser.add_argument("--tokenizer_path", "--tokenizer_cache_dir", default=None)
    parser.add_argument("--moss_max_token_length", type=int, default=4096)
    parser.add_argument("--caption_dropout_rate", type=float, default=0.0)
    parser.add_argument("--sample_uncond_skip_layers", default="22", help="Comma-separated indices; empty string disables skipping")
    parser.add_argument("--fp8_scaled", action="store_true", help="Unsupported for MOSS; produces a validation error")
    parser.set_defaults(mixed_precision="bf16", timestep_sampling="sigmoid", discrete_flow_shift=1.0, vae_dtype="float32")
    return parser


def setup_parser():
    parser = moss_image_setup_parser(setup_parser_common())
    parser.set_defaults(network_module="musubi_tuner.networks.lora_moss_image", network_dim=32)
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    parser = setup_parser()
    args = read_config_from_file(parser.parse_args(), parser)
    MossImageNetworkTrainer().train(args)


if __name__ == "__main__":
    main()

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
)
from musubi_tuner.hunyuan_video_1_5 import (
    hunyuan_video_1_5_models,
    hunyuan_video_1_5_text_encoder,
    hunyuan_video_1_5_utils,
    hunyuan_video_1_5_vae,
)
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_models import (
    HunyuanVideo_1_5_DiffusionTransformer,
    detect_hunyuan_video_1_5_sd_dtype,
)
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_vae import VAE_LATENT_CHANNELS
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HunyuanVideo15NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        # HV1.5 supports both T2V and I2V, but mode is determined dynamically per batch
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 6.0

    # region model specific
    @property
    def architecture(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO_1_5

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        # HV1.5 defaults: bfloat16 DiT, no control training for now
        self._i2v_training = False  # determined per batch via cond latents
        self._control_training = False

        # Detect the original dtype from checkpoint to prevent incompatible dtype conversions
        # float16 checkpoints cannot be safely converted to bfloat16/float32
        sd_dit_dtype = detect_hunyuan_video_1_5_sd_dtype(args.dit)
        assert not (sd_dit_dtype is torch.float16 and args.dit_dtype in ["bfloat16", "float32"]), (
            "Loaded DiT checkpoint is float16, cannot override dit_dtype to bfloat16 or float32."
            " / DiTの重みがfloat16のため、dit_dtypeをbfloat16またはfloat32に設定できません。"
        )
        # Use checkpoint's native dtype if not explicitly specified to preserve precision
        if args.dit_dtype is None:
            args.dit_dtype = "float16" if sd_dit_dtype == torch.float16 else "bfloat16"
        # VAE defaults to float16 for VRAM efficiency while maintaining acceptable quality
        if args.vae_dtype is None:
            args.vae_dtype = "float16"

    @property
    def i2v_training(self) -> bool:
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        return self._control_training

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device
        logger.info("cache Text Encoder outputs for sample prompt: %s", sample_prompts)
        prompts = load_prompts(sample_prompts)

        # HV1.5 uses Qwen2.5-VL as the primary text encoder; fp8 optional for VRAM savings
        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        tokenizer_vlm, text_encoder_vlm = qwen_image_utils.load_qwen2_5_vl(args.text_encoder, vl_dtype, device, disable_mmap=True)
        # BYT5 is used as a secondary encoder for glyph/character-level understanding
        tokenizer_byt5, text_encoder_byt5 = hunyuan_video_1_5_text_encoder.load_byt5(
            args.byt5, dtype=torch.float16, device=device, disable_mmap=True
        )

        sample_outputs = {}
        with torch.no_grad():
            for prompt_dict in prompts:
                for key in ["prompt", "negative_prompt"]:
                    p = prompt_dict.get(key, " ")
                    if p is None or (p, key) in sample_outputs:
                        continue
                    embed_vlm, mask_vlm = hunyuan_video_1_5_text_encoder.get_qwen_prompt_embeds(tokenizer_vlm, text_encoder_vlm, p)
                    embed_byt5, mask_byt5 = hunyuan_video_1_5_text_encoder.get_glyph_prompt_embeds(
                        tokenizer_byt5, text_encoder_byt5, p
                    )

                    # Trim padding to reduce memory usage during inference
                    # The mask indicates valid (non-padded) positions
                    if mask_vlm is not None:
                        valid_len = mask_vlm.to(dtype=torch.bool).sum().item()
                        embed_vlm = embed_vlm[:valid_len]
                        mask_vlm = mask_vlm[:valid_len]
                    if mask_byt5 is not None and mask_byt5.numel() > 0:
                        valid_len = mask_byt5.to(dtype=torch.bool).sum().item()
                        embed_byt5 = embed_byt5[:valid_len]
                        mask_byt5 = mask_byt5[:valid_len]

                    sample_outputs[(p, key)] = (embed_vlm, mask_vlm, embed_byt5, mask_byt5)

        # Release text encoders immediately after caching to free VRAM for DiT inference
        del tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            pd = prompt_dict.copy()
            for key in ["prompt", "negative_prompt"]:
                p = pd.get(key, " ")
                embed_vlm, mask_vlm, embed_byt5, mask_byt5 = sample_outputs[(p, key)]
                pd[f"{key}_vl_embed"] = embed_vlm
                pd[f"{key}_vl_mask"] = mask_vlm
                pd[f"{key}_byt5_embed"] = embed_byt5
                pd[f"{key}_byt5_mask"] = mask_byt5
            sample_parameters.append(pd)

        return sample_parameters

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
        """architecture dependent inference for sampling"""
        device = accelerator.device
        cfg_scale = cfg_scale if cfg_scale is not None else 1.0
        # Skip CFG computation entirely when scale is 1.0 to save inference time
        do_cfg = do_classifier_free_guidance and cfg_scale != 1.0

        timesteps, sigmas = hunyuan_video_1_5_utils.get_timesteps_sigmas(sample_steps, discrete_flow_shift, device)

        # Latent dimensions are 1/16 of image dimensions spatially, and (frames-1)/4 + 1 temporally
        # This matches the VAE's compression ratio
        lat_f = 1 + (frame_count - 1) // 4
        lat_h = height // 16
        lat_w = width // 16
        latents = torch.randn((1, VAE_LATENT_CHANNELS, lat_f, lat_h, lat_w), generator=generator, device=device, dtype=dit_dtype)

        # cond_latents: extra channel (+1) is used as a mask for I2V conditioning
        # For T2V, this is all zeros (no conditioning image)
        cond_latents = torch.zeros((1, VAE_LATENT_CHANNELS + 1, lat_f, lat_h, lat_w), device=device, dtype=dit_dtype)

        def pad_and_mask(seq: torch.Tensor):
            # Create attention mask for single sequence (batch size 1)
            mask = torch.ones(seq.shape[0], device=device, dtype=torch.bool)
            return seq.to(device=device, dtype=dit_dtype), mask

        vl_embed, vl_mask = pad_and_mask(sample_parameter["prompt_vl_embed"])
        byt5_embed, byt5_mask = pad_and_mask(sample_parameter["prompt_byt5_embed"])

        if do_cfg:
            negative_vl_embed, negative_vl_mask = pad_and_mask(sample_parameter["negative_prompt_vl_embed"])
            negative_byt5_embed, negative_byt5_mask = pad_and_mask(sample_parameter["negative_prompt_byt5_embed"])
        else:
            negative_vl_embed = negative_vl_mask = negative_byt5_embed = negative_byt5_mask = None

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0])
                # Concatenate noise latents with conditioning latents along channel dimension
                # This is how HV1.5 architecture handles I2V conditioning
                latents_concat = torch.cat([latents, cond_latents], dim=1)
                with accelerator.autocast():
                    noise_pred = transformer(
                        hidden_states=latents_concat,
                        timestep=timestep,
                        text_states=vl_embed,
                        encoder_attention_mask=vl_mask,
                        vision_states=None,
                        byt5_text_states=byt5_embed,
                        byt5_text_mask=byt5_mask,
                        rotary_pos_emb_cache=None,
                    )

                    if do_cfg:
                        # CFG: predict noise for negative prompt, then interpolate
                        # noise_pred = negative + scale * (positive - negative)
                        latents_concat = torch.cat([latents, cond_latents], dim=1)
                        neg_noise_pred = transformer(
                            hidden_states=latents_concat,
                            timestep=timestep,
                            text_states=negative_vl_embed,
                            encoder_attention_mask=negative_vl_mask,
                            vision_states=None,
                            byt5_text_states=negative_byt5_embed,
                            byt5_text_mask=negative_byt5_mask,
                            rotary_pos_emb_cache=None,
                        )
                        noise_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                latents = hunyuan_video_1_5_utils.step(latents, noise_pred, sigmas, i)

        # VAE decode: move to device just before use to minimize VRAM usage during denoising
        vae.to(device)
        with torch.autocast(device_type=device.type, dtype=model_utils.str_to_dtype(args.vae_dtype)), torch.no_grad():
            decoded = vae.decode(latents / vae.scaling_factor)[0]
        # Convert to float32 for video saving to avoid precision issues
        decoded = decoded[0].to(torch.float32).cpu()
        # Handle single-frame case (image output) by adding temporal dimension
        decoded = decoded.unsqueeze(2) if decoded.dim() == 3 else decoded
        return decoded

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info("Loading VAE model from %s", vae_path)
        vae = hunyuan_video_1_5_vae.load_vae_from_checkpoint(
            vae_path, device="cpu", dtype=vae_dtype, sample_size=args.vae_sample_size
        )
        vae.eval()
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        # Select T2V or I2V model variant based on training mode
        task_type = "i2v" if self._i2v_training else "t2v"
        transformer = hunyuan_video_1_5_models.load_hunyuan_video_1_5_model(
            device=accelerator.device,
            task_type=task_type,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
        )
        return transformer

    def compile_transformer(self, args, transformer):
        transformer: HunyuanVideo_1_5_DiffusionTransformer = transformer
        # Disable linear compilation when block swapping is enabled
        # because torch.compile doesn't work well with dynamic module movement
        return model_utils.compile_transformer(
            args, transformer, [transformer.double_blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # HV1.5 VAE cache already stores pre-scaled latents, so no additional scaling needed
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer_arg,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        transformer: HunyuanVideo_1_5_DiffusionTransformer = transformer_arg

        # Check if this batch has I2V conditioning (first frame latents)
        cond_latents = batch.get("latents_image", None)
        is_i2v_batch = cond_latents is not None
        if cond_latents is None:
            # For T2V batches, create zero conditioning tensor
            # Extra channel (+1) is the conditioning mask, all zeros means "no conditioning"
            cond_latents = torch.zeros(
                (latents.shape[0], VAE_LATENT_CHANNELS + 1, *latents.shape[2:]),
                device=latents.device,
                dtype=latents.dtype,
            )
        # Update training mode flag to reflect current batch type
        self._i2v_training = is_i2v_batch

        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        cond_latents = cond_latents.to(device=accelerator.device, dtype=network_dtype)

        # HV1.5 concatenates noisy latents with conditioning along channel dim
        latents_concat = torch.cat([noisy_model_input, cond_latents], dim=1)

        def pad_varlen(seq_list: list[torch.Tensor]):
            """Pad variable-length sequences in batch to the maximum length.
            
            Different prompts have different token counts, so we need to pad
            to create uniform tensors for batched processing.
            """
            lengths = [t.shape[0] for t in seq_list]
            max_len = max(lengths)
            padded = []
            for t in seq_list:
                if t.shape[0] < max_len:
                    t = torch.nn.functional.pad(t, (0, 0, 0, max_len - t.shape[0]))
                padded.append(t)
            stacked = torch.stack(padded, dim=0)
            # Create attention mask: True for valid positions, False for padding
            mask = torch.zeros((len(seq_list), max_len), device=accelerator.device, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :l] = True
            return stacked.to(device=accelerator.device, dtype=network_dtype), mask

        vl_embed, vl_mask = pad_varlen(batch["vl_embed"])
        byt5_embed, byt5_mask = pad_varlen(batch["byt5_embed"])
        # SigLIP vision states for I2V image understanding (optional)
        vision_states = batch.get("siglip", None)
        if vision_states is not None:
            vision_states = vision_states.to(device=accelerator.device, dtype=network_dtype)

        # Enable gradient computation for inputs when using gradient checkpointing
        # Required because checkpointing recomputes forward pass during backward
        if args.gradient_checkpointing:
            latents_concat.requires_grad_(True)
            vl_embed.requires_grad_(True)
            byt5_embed.requires_grad_(True)

        with accelerator.autocast():
            model_pred = transformer(
                hidden_states=latents_concat,
                timestep=timesteps,
                text_states=vl_embed,
                encoder_attention_mask=vl_mask,
                vision_states=vision_states,
                byt5_text_states=byt5_embed,
                byt5_text_mask=byt5_mask,
                rotary_pos_emb_cache=None,
            )

        # Flow matching target: predict the velocity (noise - clean)
        # This is different from DDPM which predicts noise directly
        target = noise - latents
        return model_pred, target

    # endregion model specific


def hv1_5_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """HunyuanVideo-1.5 specific parser setup"""
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="text encoder (Qwen2.5-VL) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument("--byt5", type=str, default=None, required=True, help="BYT5 checkpoint path")
    parser.add_argument("--image_encoder", type=str, default=None, help="SigLIP image encoder path (for I2V cache compatibility)")
    parser.add_argument(
        "--vae_sample_size",
        type=int,
        default=128,
        help="VAE sample size (height/width). Default 128; set 256 if VRAM is sufficient for better quality.",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = hv1_5_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = HunyuanVideo15NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()

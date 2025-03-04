# Modified from official implementation

# Original source:
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import logging
import math
import os
import random
import sys

import torch
from tqdm import tqdm
from accelerate import Accelerator, init_empty_weights
from utils.safetensors_utils import load_safetensors

# from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from utils.device_utils import clean_memory_on_device, synchronize_device

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        device=None,
        dtype=None,
        dit_path=None,
        dit_attn_mode=None,
        t5_path=None,
        t5_fp8=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0) **IGNORED**:
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0) **IGNORED**:
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False) **IGNORED**:
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False) **IGNORED**:
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False) **IGNORED**:
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False) **IGNORED**:
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            device (`torch.device`, *optional*, defaults to None):
                Device to place the model on. If None, use the default device (cuda)
            dtype (`torch.dtype`, *optional*, defaults to None):
                Data type for DiT model parameters. If None, use the default parameter data type from config
            dit_path (`str`, *optional*, defaults to None):
                Path to DiT model checkpoint. checkpoint_dir is used if None.
            dit_attn_mode (`str`, *optional*, defaults to None):
                Attention mode for DiT model. If None, use "torch" attention mode.
            t5_path (`str`, *optional*, defaults to None):
                Path to T5 model checkpoint. checkpoint_dir is used if None.
            t5_fp8 (`bool`, *optional*, defaults to False):
                Enable FP8 quantization for T5 model
        """
        self.device = device if device is not None else torch.device("cuda")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.t5_fp8 = t5_fp8

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # shard_fn = partial(shard_model, device_id=device_id)
        checkpoint_path = None if checkpoint_dir is None else os.path.join(checkpoint_dir, config.t5_checkpoint)
        tokenizer_path = None if checkpoint_dir is None else os.path.join(checkpoint_dir, config.t5_tokenizer)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            weight_path=t5_path,
            fp8=t5_fp8,
            # shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        self.checkpoint_dir = checkpoint_dir
        self.dit_path = dit_path
        self.dit_dtype = dtype if dtype is not None else config.param_dtype
        self.dit_attn_mode = dit_attn_mode

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
        self,
        accelerator: Accelerator,
        input_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        blocks_to_swap=0,
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            blocks_to_swap (`int`, *optional*, defaults to 0):
                Number of blocks to swap (offload) to CPU. If 0, no blocks are offloaded.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        # self.vae.model.z_dim == 16
        target_shape = (16, (F - 1) // self.vae_stride[0] + 1, size[1] // self.vae_stride[1], size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.patch_size[1] * self.patch_size[2]) * target_shape[1])

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        self.text_encoder.model.to(self.device)
        with torch.no_grad():
            if self.t5_fp8:
                with accelerator.autocast():
                    context = self.text_encoder([input_prompt], self.device)
                    context_null = self.text_encoder([n_prompt], self.device)
            else:
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)

        del self.text_encoder
        clean_memory_on_device(self.device)

        # load DiT model
        with init_empty_weights():
            # if self.checkpoint_dir is not None:
            #     logger.info(f"Creating WanModel from {self.checkpoint_dir}")
            #     self.model = WanModel.from_pretrained(self.checkpoint_dir)
            # self.model = WanModel.from_config(config)
            # else:
            logger.info(f"Creating WanModel")
            self.model = WanModel(
                dim=self.config.dim,
                eps=self.config.eps,
                ffn_dim=self.config.ffn_dim,
                freq_dim=self.config.freq_dim,
                in_dim=16,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                out_dim=16,
                text_len=512,
                attn_mode=self.dit_attn_mode,
            )
            self.model.to(self.dit_dtype)

        loading_device = self.device if blocks_to_swap == 0 else "cpu"
        logger.info(f"Loading DiT model from {self.dit_path}, device={loading_device}, dtype={self.dit_dtype}")
        sd = load_safetensors(self.dit_path, loading_device, disable_mmap=True, dtype=self.dit_dtype)

        # remove "model.diffusion_model." prefix: 1.3B model has this prefix
        for key in list(sd.keys()):
            if key.startswith("model.diffusion_model."):
                sd[key[22:]] = sd.pop(key)

        info = self.model.load_state_dict(sd, strict=True, assign=True)
        logger.info(f"Loaded DiT model from {self.dit_path}, info={info}")

        if blocks_to_swap > 0:
            logger.info(f"Enable swap {blocks_to_swap} blocks to CPU from device: {self.device}")
            self.model.enable_block_swap(blocks_to_swap, self.device, supports_backward=False)
            self.model.move_to_device_except_swap_blocks(self.device)
            self.model.prepare_block_swap_before_forward()
        else:
            # make sure the model is on the right device
            self.model.to(self.device)

        self.model.eval().requires_grad_(False)
        clean_memory_on_device(self.device)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]

        # evaluation mode
        # with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        with accelerator.autocast(), torch.no_grad():
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
                )
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            del noise

            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                del noise_pred_cond, noise_pred_uncond

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
                )[0]
                del noise_pred
                latents = [temp_x0.squeeze(0)]
                del temp_x0

            x0 = latents

        del latents
        del sample_scheduler
        del self.model
        synchronize_device(self.device)
        clean_memory_on_device(self.device)

        # return latents
        return x0[0]

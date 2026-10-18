# MOSS-Image

MOSS-Image-Base supports native Musubi latent/text caching, DiT LoRA training, full DiT fine-tuning, training samples and standalone image generation. The external Qwen3.5-9B text encoder and FLUX.2 VAE stay frozen. Full fine-tuning includes the DiT's internal text projection/refiner, all main blocks, timestep modulation and output head.

The implementation adapts [sd-scripts' MOSS support](https://github.com/kohya-ss/sd-scripts/pull/2460) to Musubi's dataset and training APIs. Model math and sampling follow the [OpenMOSS reference](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/pipeline_moss_image.py).

## Weights and Dependencies

Prepare the MOSS DiT, Qwen3.5-9B text encoder and FLUX.2 VAE checkpoints from [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base). For sharded checkpoints, pass the first safetensors file; adjacent shards are discovered automatically. The DiT loader also accepts ComfyUI fused-qkv checkpoints. The VAE loader accepts BFL/ComfyUI and diffusers layouts, including the small decoder shipped with MOSS.

Use the dependencies in `pyproject.toml`, including transformers 5.17.0. The older minimum in the legacy `requirements.txt` is insufficient for Qwen3.5. No weights are downloaded automatically. Tokenizer assets are resolved from the model repository unless `--tokenizer_path path/to/tokenizer` supplies a local directory containing `tokenizer_config.json`.

Text encoding is BF16, one unpadded prompt at a time. VAE encoding and decoding are FP32. FP8 base/text encoding, text encoder LoRA, reference-image editing and video datasets are not supported by this integration.

## Dataset and Caches

Use an image dataset and a dedicated cache directory:

```toml
[general]
resolution = 1024
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
image_directory = "path/to/images"
cache_directory = "path/to/moss-cache"
num_repeats = 1
```

Run these commands from the repository root. They are shown on one line so they work in PowerShell and POSIX shells:

```console
python moss_image_cache_latents.py --dataset_config dataset.toml --vae path/to/vae.safetensors
python moss_image_cache_text_encoder_outputs.py --dataset_config dataset.toml --text_encoder path/to/model-00001-of-00004.safetensors --tokenizer_path path/to/tokenizer
```

The latent cache holds normalized, 2x2-patchified FP32 `[128,H/16,W/16]` tensors. Text caches hold unpadded BF16 `[L,4096]` prompt features and the encoded empty prompt. They use `_moss.safetensors` and `_moss_te.safetensors` suffixes, with architecture and format metadata. Both training modes consume the same caches. The text encoder and VAE need not be loaded during training unless training samples are requested.

These caches are **not interchangeable with sd-scripts' NPZ caches or FLUX.2 caches**. Do not rename another architecture's files. `--skip_existing` validates cache format; changed captions/token limits or bucket sizes are re-encoded, while invalid formats fail with a re-cache error. When changing the VAE or text encoder weights, regenerate the corresponding cache without `--skip_existing`.

`--moss_max_token_length` defaults to 4096 and includes the chat template. Cache generation preserves the official thinking-enabled template. Do not manually insert that template into captions.

## LoRA Training

```console
accelerate launch --num_cpu_threads_per_process 1 moss_image_train_network.py --dit path/to/diffusion_pytorch_model-00001-of-00003.safetensors --dataset_config dataset.toml --output_dir output/moss-lora --output_name moss_lora --sdpa --mixed_precision bf16 --network_dim 32 --network_alpha 32 --optimizer_type AdamW --learning_rate 1e-4 --max_train_steps 1000 --gradient_checkpointing --save_every_n_steps 100 --save_state
```

The default network module is `musubi_tuner.networks.lora_moss_image`. It targets the main blocks' attention and FFN projections with `lora_unet_*` keys. To include the internal text projection and refiner blocks, add:

```console
--network_args train_text_embedder=true
```

This option does not train Qwen3.5. External text encoder LoRA weights are rejected. Existing main-block/text-refiner MOSS LoRA safetensors with matching sd-scripts names can be loaded with `--network_weights`, or merged for generation with `--lora_weight`. Match the saved adapter's dimensions and text-refiner setting, or use `--dim_from_weights`; unhandled weights are rejected rather than silently ignored.

`--caption_dropout_rate 0.1` replaces selected conditioning rows with cached empty-prompt features. It never substitutes an all-zero embedding or mask.

## Full DiT Fine-Tuning

```console
accelerate launch --num_cpu_threads_per_process 1 moss_image_train.py --dit path/to/diffusion_pytorch_model-00001-of-00003.safetensors --dataset_config dataset.toml --output_dir output/moss-full --output_name moss_full --sdpa --mixed_precision bf16 --optimizer_type Adafactor --optimizer_args relative_step=False scale_parameter=False warmup_init=False --learning_rate 1e-5 --max_train_steps 1000 --gradient_checkpointing --save_every_n_steps 100 --save_state
```

FP32 trainable parameters are the default; BF16 autocast does not change parameter storage. Add `--full_bf16` to opt into BF16 parameters. This reduces memory but also changes optimizer-update precision. `--save_precision bf16` controls exported checkpoint precision independently; training-state checkpoints retain training precision.

Full checkpoints contain canonical DiT weights, model configuration and metadata, and can be supplied directly to the generation or training entry points. LoRA-only flags such as `--network_dim`, `--network_weights` and `--base_weights` are rejected by full fine-tuning. Full exports use streaming, atomic safetensors writes; `--mem_eff_save` is accepted for consistency and streaming saving is always enabled.

### Memory Options

- `--gradient_checkpointing` is supported in both modes.
- `--gradient_checkpointing_cpu_offload` also offloads checkpointed activations; it requires `--gradient_checkpointing` and cannot be combined with block swap.
- `--blocks_to_swap N` supports up to 40 main blocks and requires CUDA. LoRA can also use `--block_swap_h2d_only --gradient_checkpointing`.
- Full-model block swap forbids H2D-only mode. Use a single process, accumulation 1, AdamW or Adafactor, and `--block_swap_optimizer_patch_params` so pending copies and gradient/optimizer devices are synchronized before updates.
- Full training supports `--fused_backward_pass` with Adafactor, one process, accumulation 1 and `--max_grad_norm 0`. Use Adafactor's own clipping configuration. It can be combined with ordinary full-model block swap.
- Per-block `--compile` uses the existing Musubi compiler options. `--compile_fullgraph` cannot be combined with `--split_attn`. The separate Accelerate dynamo backend and Forward XM (`xm_best_of_k > 1`) are rejected for this training loop.
- xFormers requires `--split_attn` for variable-length conditioning. SDPA is the tested default. The inherited `--show_timesteps` visualization and `--log_grad_metrics` options are not supported and produce a validation error.

No fixed VRAM minimum is claimed. Full-model parameters, gradients and optimizer state can exceed a 24 GB GPU even when activations are checkpointed. Choose memory settings for the actual resolution and optimizer; a tiny-model test does not establish full-model memory feasibility.

## Sampling and Generation

Add the following to either training command for sample images:

```console
--sample_prompts prompts.txt --text_encoder path/to/text_encoder.safetensors --vae path/to/vae.safetensors --sample_at_first --sample_every_n_steps 100
```

For example, `prompts.txt` can contain:

```text
A studio photograph of a red ceramic cup --w 1024 --h 1024 --s 50 --d 42
```

Training samples and independent generation use the same sampler: 50 steps, CFG 3, inference shift 5, and unconditional skipped block 22 by default. The shifted schedule snaps to the nearest point on the 1000-step training grid. Noise, timesteps, CFG arithmetic and Euler updates are FP32. Training shift defaults to 1; it is independent of inference shift. Image dimensions must be positive multiples of 16.

```console
python moss_image_generate_image.py --dit output/moss-full/moss_full.safetensors --vae path/to/vae.safetensors --text_encoder path/to/text_encoder.safetensors --prompt "A studio photograph of a red ceramic cup" --width 1024 --height 1024 --seed 42 --output_dir output/images
```

For LoRA inference, use the original base DiT and add `--lora_weight output/moss-lora/moss_lora.safetensors --lora_multiplier 1.0`. Multiple adapters accept one multiplier each. Standard DiT LoRA keys and dimensions are validated before streaming weight merges.

`--uncond_skip_layers ""` disables unconditional block skipping in standalone generation. The corresponding training option is `--sample_uncond_skip_layers ""`. `--guidance_scale 1` disables CFG. A native cache can replace text encoder loading for independent generation: pass `--text_cache path/to/item_moss_te.safetensors` without `--prompt` or `--negative_prompt`. That mode uses the cached caption and cached empty negative prompt.

Generation writes PNGs with seed/model metadata. Existing output filenames require explicit `--overwrite`.

## Resume

`--network_weights` loads adapter weights; it does not resume optimizer or scheduler state. Likewise, passing a full trained DiT to `--dit` starts a new run unless `--resume` is used.

Use `--save_state` with periodic checkpointing, or `--save_state_on_train_end`. Resume either training mode by repeating its original command with:

```console
--resume output/moss-full/moss_full-step00000100-state
```

The state directory contains Accelerate's model/optimizer/scheduler/RNG files, `moss_progress.json` and per-rank progress files. MOSS restores the optimizer step, epoch, next data-loader batch and pending timestep buckets instead of restarting the data order. Keep the dataset/cache files, batch sizes, accumulation count, number of processes, seed, sampling settings, optimizer and model/network configuration unchanged. A changed cache fingerprint, training mode, seed, accumulation count, timestep-bucketing configuration or world size is rejected. Download remote state directories locally before resuming.

`--no_metadata` omits descriptive training/export metadata, retaining only fields needed to identify and reload the model or adapter. Full exports always retain their model configuration. Training-state files still contain the information required for resume.

## Validation Scope

Tests use actual small MOSS models and native cache files to verify model forward/backward, LoRA updates/merge, full-model updates, checkpoint export/reload, deterministic resume, BF16 CLI execution and PNG generation. CUDA tests exercise activation offloading and full-model block swap. They do not substitute for comparison with the official multi-billion-parameter checkpoints or a real-data training-quality evaluation.

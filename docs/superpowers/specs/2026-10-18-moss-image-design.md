# MOSS-Image Adaptation

Approved in chat on 2026-10-18, including DiT full fine-tuning. The user authorized writing this spec and the implementation plan, then proceeding without another approval round.

## Scope

Add MOSS-Image-Base text-to-image support to Musubi Tuner: latent caching, frozen text encoder caching, DiT LoRA training, DiT full fine-tuning, training samples, independent generation, checkpoint saving and training-state restoration. The external Qwen3.5-9B text encoder and FLUX.2 VAE remain frozen. Full fine-tuning trains all DiT parameters, including the internal text embedder/refiner, timestep modulation and output head.

Do not add text encoder LoRA, new quantization schemes, image editing, GUI integration, or changes to the parent scripts repository. Do not download model weights or start a long real-model training job automatically.

## Source Baseline

- Target: `0810ac1`, branch `qinglong`.
- Upstream: `kohya-ss/sd-scripts` dev at `6a3290182c786541ccf292106782b51568e46da5`.
- Model/inference: [717aef6](https://github.com/kohya-ss/sd-scripts/commit/717aef660b148b0feb49920af1289fcc36a7fabb).
- Cache strategies: [96d2c18](https://github.com/kohya-ss/sd-scripts/commit/96d2c1883bbb26ec350a07000832e22298d338ca).
- LoRA trainer: [cb5c24f](https://github.com/kohya-ss/sd-scripts/commit/cb5c24fc724f758fcb1e63969872f2f561ac1396).
- Numerical reference: [OpenMOSS pipeline](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/pipeline_moss_image.py).

Preserve source attribution and licenses. Adapt the model implementation; do not import sd-scripts' training framework into this repository.

## Integration

Use a `musubi_tuner.moss_image` package for the model, loading/text helpers, VAE compatibility and sampler. Add root forwarding scripts and package entry points:

- `moss_image_cache_latents.py`
- `moss_image_cache_text_encoder_outputs.py`
- `moss_image_train_network.py`
- `moss_image_train.py`
- `moss_image_generate_image.py`

LoRA uses the existing `NetworkTrainer` and `networks.lora` infrastructure through a small `lora_moss_image` adapter. Full fine-tuning inherits MOSS model-specific behavior and uses a dedicated loop following the repository's current full-model trainers. It must not inherit unrelated model behavior or change other trainers' timestep semantics. Share MOSS loss, sampling and checkpoint helpers where both entry points need them.

Register short architecture `moss` and full architecture `moss_image`, 16-pixel resolution buckets, cache metadata and SAI metadata. Keep shared edits additive. Reuse the local attention implementation and `BlockSwapConfig`/offloader API, not sd-scripts' similarly named but incompatible API.

## Model and Weights

The base DiT has dimension 3072, FFN dimension 8192, 24 heads, 42 main blocks, text dimension 4096, frequency dimension 256 and 128 latent channels. Preserve separate `layers.N.proj.q/k/v` state keys. Accept official shards and ComfyUI single-file weights; strip known prefixes and split only main-block fused qkv weights. Reject missing, duplicate or unexpected trainable weights rather than silently accepting a partial model. Preserve derived RoPE frequencies in FP32 even after dtype conversion and loading.

Load only `Qwen3_5TextModel`, not the visual tower or language-model head. Bundle its configuration inside the package so installed wheels work. Encode prompts individually without token padding through the text encoder, then right-pad hidden states and construct a valid-token mask. Use the official thinking-enabled chat template and a maximum of 4096 tokens. An empty prompt is a real encoded prompt; a zero-length or all-zero attention mask is invalid.

Keep external text encoder weights/outputs BF16 and VAE FP32. Unsupported FP8 options fail early. Support a local tokenizer directory; otherwise only tokenizer assets may be resolved from the model's Hugging Face repository.

## VAE and Cache Contract

The local FLUX.2 VAE `encode()` already performs 2x2 patchification and BatchNorm normalization. Upstream MOSS caches raw 32-channel latents and transforms them later. Musubi MOSS will instead store normalized, patchified FP32 tensors `[128, H/16, W/16]`, so training never needs to load the VAE. Decoding uses the existing inverse-normalize/unpatchify contract.

Reuse local VAE layers. Add optional decoder width support with the old default unchanged, and a MOSS loader for standard BFL/ComfyUI and diffusers VAE names. Support the small decoder bundled with MOSS, retain integer BN counters and load strictly.

Use separate `moss` cache filenames and explicit version/format metadata. Reject foreign architecture, raw 32-channel latents, wrong feature dimensions, non-finite values and incompatible cache versions. Existing caches from sd-scripts are not read implicitly. Validate caches on skip-existing as well as on consumption.

Text caches contain unpadded BF16 `[L,4096]` prompt features and encoded-empty-prompt features. Reuse the dataset's `varlen_` convention. A MOSS `caption_dropout_rate` option defaults to zero; selected rows use the cached empty-prompt features, never zero embeddings. Re-pad only when constructing a training batch. LoRA and full training consume the identical cache format.

## Training Semantics

Construct noisy latents as `(1 - sigma) * clean + sigma * noise` and target `noise - clean`. Keep noise interpolation, sigmas and model timesteps FP32. Pass `sigma * 1000` to MOSS with no additional `+1` and no division by 1000. Compute MSE in FP32. Training shift defaults to 1; sampling shift defaults to 5. Use existing timestep options only where their noise coefficient semantics are preserved, and calculate weighting from the actual sigma.

LoRA defaults to main `MossImageBlock` linear projections with `lora_unet` names. An explicit `train_text_embedder=true` includes the DiT input projection and two refiner blocks; this is not text encoder training. Imported adapters must map to actual supported modules; unsupported text encoder adapters must not be silently discarded.

Full training defaults to FP32 trainable parameters with BF16 mixed precision; `--full_bf16` explicitly permits BF16 parameters. Reject LoRA-only flags, quantized trainable weights and H2D-only swapping. Support gradient accumulation, gradient checkpointing, per-block compilation and ordinary AdamW/Adafactor workflows. Block swapping needs a validated optimizer path: wait for transfers and keep gradients and optimizer state on the appropriate parameter device. Reject unsupported combinations before loading large weights. Fused Adafactor, if enabled, requires one accumulation step and a supported execution mode.

Save full DiT checkpoints in canonical model keys, excluding compile wrapper prefixes. Honor save precision without degrading derived FP32 buffers. The inference loader must load these files strictly. Accelerate state saves must restore optimizer, scheduler and RNG as well as global step, epoch and data-loader position. An interrupted and resumed tiny-model run must match an uninterrupted run. Exporting a full checkpoint and resuming a training state are separate operations.

## Sampling

Use one sampler for standalone output and training samples. Default to 50 steps, CFG 3, flow shift 5 and unconditional skipped block 22 (zero-based). Permit explicitly disabling skipped layers and validate indices. Snap the shifted inference sigmas to the nearest point on the 1000-step training grid, remove duplicate consecutive values and finish at zero. Initial noise, CFG arithmetic and Euler updates use FP32. Validate positive image dimensions divisible by 16.

Load/merge LoRA with canonical names and preserve base-model output when the adapter is disabled or zero initialized. Return training samples through the existing image-saving interface. Restore training mode and offloader mode after sampling, including exceptions.

## Verification

Use the existing environment, without dependency upgrades. The inspected versions are torch 2.13.0+cu130, transformers 5.17.0, diffusers 0.40.0 and accelerate 1.6.0. Check optional dependencies lazily so CLI help and CPU tests work without model downloads.

Tests must cover batch versus independent-example forward, padding independence, checkpointed backward, main qkv conversion, VAE conversion and unchanged FLUX.2 defaults, BF16 text preservation, cache rejection, exact timestep/noise relationships, caption dropout, LoRA parameter updates/merge, full-model updates across module groups, save/reload and interrupted/resumed optimization. Exercise actual cache and trainer paths with tiny models rather than relying only on parser or source-text assertions. Run existing regression tests and report unrelated failures explicitly.

The local GPU is an RTX 4090 with 24 GB. The checked `ckpts` and `huggingface` directories contain a FLUX.2 VAE but no identified MOSS/Qwen3.5 weights. CPU/tiny-model CUDA checks and full-weight numerical/image checks must be reported separately. Small-model tests do not prove large-model memory feasibility or image quality.

## Alternatives

Copying the complete upstream trainer would retain incompatible dataset, latent and offloader contracts. Unifying both repositories' training frameworks would expand the regression surface beyond this model. The chosen native adapter approach preserves upstream numerical behavior while reusing Musubi's established workflow.

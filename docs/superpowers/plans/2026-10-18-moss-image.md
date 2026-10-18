# MOSS-Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Complete native MOSS-Image caching, LoRA/full DiT training, sampling, export and resume workflows.

**Architecture:** Port the numerical model and sampler, adapt I/O to Musubi, and reuse its attention, datasets, LoRA and optimizer utilities. Use a MOSS full-model trainer with shared model-specific behavior; do not restructure unrelated trainers.

**Tech Stack:** Python, PyTorch, Accelerate, transformers, safetensors, pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-moss-image-design.md`

## Global Constraints

- External Qwen3.5 text encoder and FLUX.2 VAE remain frozen.
- Architecture names: `moss` / `moss_image`; resolution step 16.
- Cache normalized FP32 `[128,H/16,W/16]` latents, unpadded BF16 `[L,4096]` text.
- MOSS model time is FP32 `sigma * 1000`, without `+1` or `/1000`.
- No text encoder LoRA, new quantization, GUI or parent-repository changes.
- No large weight downloads or long real-model training jobs.
- Preserve old model behavior and run existing regressions.
- User explicitly authorized direct execution after documents; no further approval handoff.

## Review Focus

- Variable text lengths must not change per-example image RoPE or attention semantics (Task 1).
- Stale/wrong-architecture caches must fail even with skip-existing enabled (Task 2).
- A full checkpoint must reload without missing keys or BF16-rounded RoPE frequencies (Tasks 1/4).
- Sampling exceptions and resumed runs must not leave wrong model/offloader modes or reset training progress (Tasks 3/4).
- Trainable-weight swapping cannot use frozen CPU snapshots or incompatible optimizer state (Task 4).

### Task 1: Numerical Model, Loader and VAE

Files: create `src/musubi_tuner/moss_image/{__init__,model,utils,vae,sampling}.py`, packaged Qwen3.5 config, and `tests/test_moss_image_model.py`; modify `flux_2/flux2_models.py` only for optional decoder width.

Interfaces: `MossImage.forward(x,timesteps,context,context_mask,skip_layers)` returns `[B,C,H,W]`; `load_moss_image_model(...)` accepts canonical/sharded/fused weights; `load_vae(path,device,dtype)` returns the existing normalized-latent AE contract; `get_sampling_sigmas(steps,shift)` returns FP32 sigmas ending at zero.

- [x] Add tests for batch/per-example parity, padding independence, checkpointed gradients, qkv split and strict small-model checkpoint round-trip.
- [x] Run `python -m pytest tests/test_moss_image_model.py -q`; expect missing-MOSS failures before implementation.
- [x] Adapt upstream model using local attention and offloader APIs; add strict weight transformations and packaged text configuration.
- [x] Add tests and implementation for small-decoder/default VAE contracts and nearest-grid sampler defaults.
- [x] Run the model tests and relevant existing FLUX.2/cache tests; expect pass.

### Task 2: Native Cache and Text Encoding

Files: create both MOSS cache entry points/root wrappers, `moss_image/cache.py`, `tests/test_moss_image_cache.py`; modify dataset architecture/bucket/cache exports as required.

Interfaces: cache writers consume `ItemInfo`; latent tensors are `[128,h,w]`; text tensors are `[L,4096]`. `encode_tokens` returns unpadded rows; `pad_text_embeds` returns padded embeddings plus a binary mask. Cache metadata identifies architecture, version and latent representation.

- [x] Add failing tests for actual safetensors round-trips, BF16 bit preservation, invalid/raw/foreign cache rejection, and skipped-existing validation.
- [x] Implement local tokenizer/text-only loading and unpadded encoding, native cache writers and real dataset loops.
- [x] Add dropout tests: probability 0 preserves prompts, probability 1 uses encoded empty prompts, source caches remain unchanged.
- [x] Run `python -m pytest tests/test_moss_image_cache.py tests/test_image_cache_compatibility.py -q`; expect pass.

### Task 3: LoRA and Shared Training/Sampling

Files: create `networks/lora_moss_image.py`, `moss_image_train_network.py` and root wrapper, `tests/test_moss_image_training.py`; update model metadata registrations.

Interfaces: `MossImageNetworkTrainer` subclasses `NetworkTrainer`; `call_dit` returns `DiTOutput`; MOSS overrides time/noise construction. The sampler accepts cached positive/negative embeddings and returns normalized FP32 latents.

- [x] Add failing tests that catch the base trainer's timestep `+1`, BF16 timestep rounding and wrong flow target.
- [x] Add LoRA training/update, target selection and load/merge tests against a small real MOSS model.
- [x] Implement the adapter and trainer hooks, cache checks, sample prompt encoding and shared sampler.
- [x] Test CFG skip behavior, invalid dimensions and restoration after a sampling exception.
- [x] Run `python -m pytest tests/test_moss_image_training.py tests/test_moss_image_model.py tests/test_moss_image_cache.py -q`; expect pass.

### Task 4: Full DiT Training, Export and Resume

Files: create `moss_image_train.py` and root wrapper, MOSS checkpoint/training helpers as needed, `tests/test_moss_image_finetune.py`.

Interfaces: `MossImageTrainer` shares MOSS model hooks, optimizes all DiT parameters and exports canonical full weights. Training state includes next data-loader position, epoch and global optimizer step in addition to Accelerate model/optimizer/scheduler/RNG state.

- [x] Add failing tests for full-model updates across main/refiner/modulation/head parameters and direct full-checkpoint reload.
- [x] Implement full training using local optimizer/scheduler/Accelerate utilities and shared MOSS loss.
- [x] Add rejection tests for quantized training, H2D-only swapping, LoRA-only flags and unsafe fused optimizer combinations.
- [x] Add an interrupted/resumed tiny dataset run and compare subsequent weights, LR and global step against an uninterrupted run.
- [x] Exercise supported block-swap optimizer behavior on CUDA where available; otherwise mark that runtime test unavailable, not passed.
- [x] Run `python -m pytest tests/test_moss_image_finetune.py -q`; expect pass.

### Task 5: Independent Generation and End-to-End Validation

Files: create `moss_image_generate_image.py` and root wrapper, `docs/moss_image.md`, `tests/test_moss_image_entrypoints.py`; update README and entrypoint tests where appropriate.

Interfaces: generation uses Task 1 loaders and Task 3 sampler, accepts full checkpoints and optional canonical DiT LoRA, and writes PNG output. Five entry points support `--help` without loading weights.

- [x] Add failing CLI and tiny generation tests, including LoRA and a saved full-model checkpoint.
- [x] Implement generation and usage examples for cache, LoRA, full fine-tuning, sampling and resume.
- [x] Run all MOSS tests, CPU/CUDA tiny smoke tests and `python -m pytest`; report any unrelated baseline failures by name.
- [x] Run lint/compile checks and `git diff --check`.
- [x] Obtain a fresh-context final review, fix important findings with regression tests, and report real-weight validation limits.

## Execution Record

Work remains in the user's clean `qinglong` submodule checkout so changes are immediately usable by the surrounding launcher repository. No branch switch, push or merge is part of this task. Track task results and deviations in the execution ledger. Product changes remain reviewable locally.

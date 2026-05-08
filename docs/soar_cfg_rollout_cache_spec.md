# SOAR CFG Rollout and Empty Prompt Cache Spec

## Status

- Implemented for LoRA/network training, except real one-step smoke verification.
- Scope: add official-style CFG rollout support to the existing SOAR-lite implementation.
- Target trainers: `zimage`, `flux_2`, and standard text-to-image `qwen_image`.
- Primary goal: make the rollout velocity use `v_uncond + scale * (v_cond - v_uncond)` when requested.
- Secondary goal: provide empty prompt embeddings without loading text encoders during training.
- Non-goal: expand SOAR to every architecture.
- Non-goal: change baseline training or existing cond-only SOAR behavior when CFG rollout is not explicitly requested.
- Non-goal: add CFG auxiliary loss. CFG is used only to construct rollout points; auxiliary supervision still uses the conditional training forward.

## 1. Official Reference

HY-SOAR computes empty prompt embeddings once, then uses them for rollout CFG:

- `uncond_hidden_states, uncond_pooled_embeds = compute_text_embeddings("")`
- rollout batch concatenates unconditional and conditional embeddings
- `v_cfg = v_uncond + cfg_scale_sampling * (v_cond - v_uncond)`
- `v_cfg` is used only for `single_step_aux_points(...)`
- auxiliary points are still supervised with conditional prompts

Official default values relevant to this spec:

- `cfg_scale_sampling = 4.5`
- `trajectory_length = 6`
- `num_sampling_steps = 40`
- `lambda_aux = 1.0`

References:

- `https://github.com/Tencent-Hunyuan/HY-SOAR/blob/main/soar/train_soar_sd3_5m.py`
- `https://github.com/Tencent-Hunyuan/HY-SOAR/blob/main/soar/utils/algorithm.py`

## 2. Current Local Limitation

The local SOAR implementation currently uses the conditional main prediction as rollout velocity:

```text
v_rollout = v_cond
```

That is mathematically equivalent to CFG scale `1.0`, not official SOAR's default rollout behavior.

The missing piece is not just a CLI flag. The training batches only contain positive prompt embeddings:

- `zimage`: `batch["llm_embed"]`
- `flux_2`: `batch["ctx_vec"]`
- `qwen_image`: `batch["vl_embed"]`

Therefore real CFG rollout requires an unconditional embedding source that is available during training.

## 3. Linus Review

### 3.1 Real Problem

The user's blur reports are plausible if SOAR rollout is cond-only. Official SOAR uses CFG during rollout, so local SOAR-lite may generate weaker or biased auxiliary points even if the loss math is correct.

### 3.2 Simpler Path

Do not load text encoders during training. The project already has text encoder cache scripts. The cheaper and safer path is to cache the empty prompt embedding once per cache directory while running those scripts.

### 3.3 Main Breakage Risk

Do not store the empty prompt embedding in every sample cache. `flux_2` text embeddings are large (`ctx_vec`, fixed length), and duplicating the same tensor into every item would inflate cache size badly.

Do not silently switch existing `--soar` runs from cond-only to CFG `4.5`. That would change training behavior and increase compute/VRAM without an explicit user choice.

## 4. User-Facing API

Add one SOAR-specific rollout argument:

```text
--soar_cfg_scale_sampling FLOAT
```

Default:

```text
1.0
```

Rationale:

- `1.0` preserves current SOAR-lite behavior.
- Any value other than `1.0` enables CFG rollout and requires empty prompt cache.
- Official-style profile uses `--soar_cfg_scale_sampling 4.5`.
- Use the `soar_` prefix to avoid confusion with sample-prompt `cfg_scale`.

Validation:

- `--soar_cfg_scale_sampling <= 0` is invalid.
- `--soar_cfg_scale_sampling != 1.0` requires a supported trainer and a readable empty prompt cache.
- If `--soar` is disabled, this argument has no effect.

## 5. Empty Prompt Cache Design

### 5.1 Cache Location

Write one sidecar file per dataset cache directory and architecture, inside a dedicated subdirectory so it does not match normal latent or text-encoder cache globs:

```text
<cache_directory>/__soar_empty_prompt/<architecture>.safetensors
```

Examples:

- `__soar_empty_prompt/zi.safetensors`
- `__soar_empty_prompt/f2k4b.safetensors`
- `__soar_empty_prompt/qi.safetensors`

The path intentionally must not match the existing per-item patterns:

```text
*_<architecture>.safetensors
*_<architecture>_te.safetensors
```

Otherwise latent-cache discovery may parse it as a training sample, or `post_process_cache_files(...)` may treat it as stale per-item text cache.

### 5.2 Tensor Keys

Use architecture-native tensor names so adapter loading stays simple:

```text
zimage:
  varlen_llm_embed_<dtype>

flux_2:
  ctx_vec_<dtype>

qwen_image:
  varlen_vl_embed_<dtype>
```

Metadata:

```text
cache_kind = "soar_empty_prompt"
prompt = ""
architecture = <architecture full name or model architecture>
model_version = <model version when available>
format_version = "1"
```

### 5.3 Cache Script Changes

Modify exactly these scripts:

- `src/musubi_tuner/zimage_cache_text_encoder_outputs.py`
- `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`
- `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py`

After loading the text encoder and resolving datasets:

1. encode `""` once with the same dtype/path/model version as normal caching
2. trim variable-length embeddings using the same rules as normal item caches
3. write the sidecar file into every unique dataset `cache_directory`
4. if `--skip_existing` is set and the sidecar exists, leave it unchanged

No per-item cache format migration is required.

## 6. Training Adapter Contract

Add a small SOAR CFG adapter surface to `NetworkTrainer`:

```python
def load_soar_empty_prompt_cache(self, args, train_dataset_group) -> None:
    ...

def get_soar_rollout_velocity_standard(
    self,
    args,
    accelerator,
    transformer,
    batch,
    noisy_model_input,
    timesteps,
    network_dtype,
    cond_model_pred,
) -> torch.Tensor:
    ...
```

Default behavior:

- `load_soar_empty_prompt_cache(...)` is a no-op when `soar_cfg_scale_sampling == 1.0`.
- If `soar_cfg_scale_sampling != 1.0`, the default trainer raises `NotImplementedError`.
- `get_soar_rollout_velocity_standard(...)` returns `self.soar_velocity_to_standard(cond_model_pred)` when CFG scale is `1.0`.

The shared SOAR branch in `hv_train_network.py` should call:

```text
v_rollout_standard = self.get_soar_rollout_velocity_standard(...)
```

Then pass `v_rollout_standard` directly to `single_step_aux_points(...)`.

This replaces the current hard-coded use of conditional-only velocity:

```text
self.soar_velocity_to_standard(model_pred)
```

## 7. CFG Formula

CFG must be combined in standard velocity space:

```text
v_cfg = v_uncond + scale * (v_cond - v_uncond)
```

Adapters should:

1. compute `v_uncond_local`
2. convert both predictions to standard SOAR velocity
3. combine in standard space

This avoids sign mistakes for `zimage`, whose local training target is reversed.

Do not use architecture-specific inference CFG formulas if they differ from the official SOAR formula. This spec is for rollout supervision, not final image sampling.

## 8. Implementation Details by Model

### 8.1 Z-Image

Supported v1 mode:

- standard text-to-image LoRA

Empty cache:

- load `varlen_llm_embed_<dtype>`
- keep on CPU
- for each rollout batch, build `batch_uncond = batch.copy()`
- replace `batch_uncond["llm_embed"]` with a list containing one copy per sample

Forward:

- run `predict_velocity_for_soar(...)` once with `batch_uncond`
- convert `cond_model_pred` and `uncond_model_pred` with `soar_velocity_to_standard(...)`
- return CFG-combined standard velocity

Reject CFG rollout for Z-Image control/Omni paths until a dedicated test proves the control latents and empty text conditioning are handled correctly.

### 8.2 Flux.2

Supported v1 mode:

- non-guidance-distilled Flux.2 text-to-image

Empty cache:

- load `ctx_vec_<dtype>`
- expand or repeat to `[B, T, D]`
- replace `batch_uncond["ctx_vec"]`

Forward:

- run the same adapter forward used by SOAR auxiliary prediction
- combine in standard velocity space

Validation:

- if `model_version_info.guidance_distilled` is true and `soar_cfg_scale_sampling != 1.0`, raise early
- cond-only SOAR can remain available for guidance-distilled models

### 8.3 Qwen-Image

Supported v1 mode:

- standard text-to-image only

Unsupported v1 modes:

- edit
- layered
- `remove_first_image_from_target`

Empty cache:

- load `varlen_vl_embed_<dtype>`
- keep one variable-length tensor on CPU
- replace `batch_uncond["vl_embed"]` with one copy per sample

Forward:

- run `predict_velocity_for_soar(...)` with unconditional embeddings
- combine in standard velocity space

Do not apply Qwen inference CFG norm-rescale in SOAR rollout unless a later spec explicitly asks for it. Official HY-SOAR uses the plain CFG velocity formula.

## 9. Full-Finetune Z-Image

`src/musubi_tuner/zimage_train.py` currently has its own SOAR branch outside `NetworkTrainer`.

This implementation limits CFG rollout to LoRA/network training. If full-finetune parity is required, implement the same logic after the adapter path is stable:

- load the same sidecar empty prompt cache
- build unconditional `llm_embed` batch
- run one unconditional forward under `torch.no_grad()`
- combine in standard velocity space
- use the result only for `single_step_aux_points(...)`

Do not block the LoRA implementation on this unless the user explicitly needs full-finetune CFG SOAR.

## 10. Compute and VRAM Impact

With `--soar_cfg_scale_sampling 1.0`:

- no extra uncond forward
- no behavior change from current SOAR-lite

With `--soar_cfg_scale_sampling != 1.0`:

- one extra no-grad unconditional model forward per SOAR training step
- extra CPU memory for one empty prompt embedding
- extra GPU memory for temporary unconditional embeddings and model inputs
- no extra backward graph for the uncond rollout forward

Implementation should prefer reusing the already-computed conditional `model_pred` and only running the unconditional forward. That is mathematically equivalent to official CFG under deterministic model forward and costs less than the official concatenated uncond+cond rollout batch.

## 11. Test Plan

### 11.1 Unit Tests

Add tests for:

- `add_soar_arguments(...)` exposes `--soar_cfg_scale_sampling` and defaults to `1.0`
- validation rejects non-positive CFG scale
- validation rejects CFG rollout when empty prompt cache is missing
- empty prompt sidecar path does not match per-item text encoder cache glob
- loading each architecture's sidecar finds the right tensor key
- CFG combination happens in standard velocity space
- scale `1.0` returns the same rollout velocity as current cond-only SOAR

### 11.2 Adapter Tests

Add or extend:

- `tests/test_zimage_soar_lora.py`
- `tests/test_flux2_soar_lora.py`
- `tests/test_qwen_image_soar_lora.py`

Required cases:

- adapters replace only text conditioning, not latents/noise/timesteps
- zimage sign conversion uses standard-space CFG formula
- flux2 rejects CFG rollout for guidance-distilled model versions
- qwen rejects CFG rollout for edit/layered modes

### 11.3 Smoke Tests

Minimum local smoke command shape:

```bash
--soar --soar_trajectory_length 1 --soar_lambda_aux 0.1 --soar_cfg_scale_sampling 4.5
```

Expected log additions:

- `soar/cfg_scale_sampling`
- `soar/cfg_rollout_enabled`
- existing `loss/main`, `loss/aux`, `soar/aux_points_per_sample`

## 12. Implementation Phases

### Phase 1: Cache Infrastructure

- [x] add sidecar path helpers
- [x] add sidecar save/load helpers
- [x] update three text encoder cache scripts
- [x] add unit tests for paths and tensor-key loading

### Phase 2: Shared Network Training Hook

- [x] add `--soar_cfg_scale_sampling`
- [x] validate scale and supported modes
- [x] load empty prompt cache when needed
- [x] replace conditional-only rollout velocity with adapter-provided standard rollout velocity
- [x] keep aux loss and main loss normalization unchanged

### Phase 3: Model Adapters

- [x] implement Z-Image empty-conditioning rollout
- [x] implement Flux.2 empty-conditioning rollout
- [x] implement Qwen-Image empty-conditioning rollout
- [x] reject unsupported modes early

### Phase 4: Docs and Verification

- [x] document official-style profile in `docs/zimage.md`, `docs/flux_2.md`, and `docs/qwen_image.md`
- [x] run focused unit tests
- [x] run `ruff check` on touched files
- [ ] run one real one-step smoke test when local model/cache paths are available

## 13. Recommended User Profile After Implementation

For Z-Image Base LoRA after empty prompt cache exists:

```bash
--timestep_sampling sigma \
--weighting_scheme logit_normal \
--logit_mean 0 \
--logit_std 1 \
--soar \
--soar_cfg_scale_sampling 4.5 \
--soar_lambda_aux 0.1 \
--soar_trajectory_length 1 \
--soar_sigma_upper_ratio 1.2
```

Only increase `soar_lambda_aux` or `soar_trajectory_length` after sharpness is stable.

# SOAR-lite LoRA Targeted Model Extension Spec

## Status

- Implemented except real one-step smoke verification, which requires local model and dataset configs
- Scope: extend the existing `zimage` SOAR-lite work to LoRA training for exactly three targets: `zimage`, `flux_2`, and `qwen_image`
- Primary goal: make SOAR usable in `*_train_network.py` without silently changing baseline training when `--soar` is absent
- Secondary goal: define a small model-adapter contract that is only as generic as these three targets require
- Non-goal: reproduce official CFG-based SOAR in this phase
- Non-goal: support `flux_kontext`, Wan, HunyuanVideo, FramePack, or other trainers in this phase

## 1. Problem Statement

The current SOAR-lite implementation is wired into full finetuning only:

- `src/musubi_tuner/zimage_train.py` registers `--soar`
- `src/musubi_tuner/zimage_train.py` owns the auxiliary loss branch
- `src/musubi_tuner/zimage_train_network.py` only exposes `predict_velocity(...)` and the normal `call_dit(...)`

LoRA training uses `src/musubi_tuner/hv_train_network.py::NetworkTrainer.train(...)` and model-specific subclasses such as:

- `src/musubi_tuner/zimage_train_network.py`
- `src/musubi_tuner/flux_2_train_network.py`
- `src/musubi_tuner/qwen_image_train_network.py`

The real question is not "can we add another CLI flag". The question is whether the shared network training loop has enough information to generate SOAR auxiliary points and run extra DiT forwards while updating only the network parameters.

## 2. Linus Review Rules

This extension must pass three checks before implementation:

1. Is this a real problem?
   LoRA users cannot currently test SOAR without switching to full finetuning, so the feature is not available to the most common low-VRAM training path.

2. Is there a simpler way?
   The simplest honest path is `zimage` LoRA first, because `zimage_train_network.py` already has `predict_velocity(...)`. Do not build a broad framework beyond the three requested targets.

3. What can this break?
   It can break baseline LoRA training if the normal loss path changes, if optimizer stepping changes under gradient accumulation, or if the auxiliary forward accidentally updates the full transformer instead of only the LoRA network.

## 3. Current Facts

### 3.1 Network Training Loop

`NetworkTrainer.train(...)` currently does the common LoRA step:

1. load cached latents
2. sample `noise`
3. call `get_noisy_model_input_and_timesteps(...)`
4. compute optional weighting via `compute_loss_weighting_for_sd3(...)`
5. call model-specific `call_dit(...)`
6. compute MSE loss
7. `accelerator.backward(loss)`
8. reduce, clip, `optimizer.step()`, `lr_scheduler.step()`, `optimizer.zero_grad(...)`

The optimizer is built from network trainable parameters, not the frozen base model. Therefore an auxiliary SOAR forward can update LoRA if it runs through the transformer with the network applied and calls `accelerator.backward(...)` before the same optimizer step.

### 3.2 Sign Conventions Differ

Architectures do not all use the same local velocity sign:

| Architecture | Local target in `call_dit(...)` | Standard SOAR sign match |
| --- | --- | --- |
| `zimage` | `latents - noise` | reversed |
| `flux_2` | `noise - latents` | matches |
| `qwen_image` | `noise - latents` | matches |

This means any shared SOAR helper must not assume one sign. It must either receive a sign convention or ask the model adapter to build the auxiliary target.

### 3.3 Latent Shapes Differ

The current full-finetune `zimage` path uses 4D image latents: `[B, C, H, W]`.

The target trainers use:

- 4D image latents: `zimage`, `flux_2`
- 5D image latents: `qwen_image`
- packed internal model tokens inside `call_dit(...)`

SOAR auxiliary points should be generated in the external latent shape used by `get_noisy_model_input_and_timesteps(...)`, not in the packed token shape internal to a model forward.

### 3.4 Conditioning Is Model-Specific

Each model-specific `call_dit(...)` knows how to read conditioning from `batch`:

- `zimage`: `llm_embed`
- `flux_2`: `ctx_vec` and optional control latents
- `qwen_image`: `vl_embed`, optional edit or layered control latents

The shared SOAR loop should not unpack all of these directly. It should call an adapter-owned forward helper.

## 4. User-Facing API

Use the same CLI names for full finetune and LoRA:

- `--soar`
- `--soar_lambda_aux`
- `--soar_trajectory_length`
- `--soar_num_sampling_steps`

For network training these flags should be registered through a shared SOAR parser helper, not copied into every model parser.

Do not add these flags blindly to `setup_parser_common()`. That parser is also used by full-finetune scripts such as `zimage_train.py`, which already register SOAR flags. The helper must be idempotent or called only from scripts that do not already provide SOAR args.

Initial defaults remain:

- `--soar_lambda_aux 1.0`
- `--soar_trajectory_length 6`
- `--soar_num_sampling_steps 40`

Smoke-test profile:

```bash
--soar --soar_trajectory_length 1
```

Unsupported combinations must fail early:

- `--soar` with fused per-parameter backward optimizers where optimizer step timing is not controlled by the normal loop
- `--soar` on a model whose adapter does not implement the SOAR forward contract
- `--soar` on a model mode where auxiliary latent reconstruction is ambiguous, such as Qwen layered training before that path has a dedicated test

## 5. Architecture

### 5.1 Keep `soar_utils.py` Math-Only

`src/musubi_tuner/soar_utils.py` should stay independent of model classes:

- sigma and timestep mapping
- ODE single-step rollout
- auxiliary point sampling
- float32 scheduler math

It should not know about text embeddings, LoRA networks, or packed token layouts.

### 5.2 Add a Training Helper Layer

Add a small helper module, proposed name:

```text
src/musubi_tuner/soar_train_utils.py
```

Responsibilities:

- validate shared SOAR args
- register shared SOAR CLI args through an idempotent helper
- compute per-sample weighted MSE with shape-safe weighting
- normalize main and auxiliary losses
- run serialized backward:
  1. backward main loss
  2. build aux points under `torch.no_grad()`
  3. forward and backward each auxiliary point
  4. return log values

This helper should not call a specific model. It receives a callback:

```python
predict_velocity_fn(aux_latents: torch.Tensor, aux_timesteps: torch.Tensor) -> torch.Tensor
```

### 5.3 Define a Model Adapter Contract

Each supported trainer should provide a small SOAR adapter surface:

```python
def supports_soar(self, args) -> bool:
    ...

def predict_velocity(
    self,
    args,
    accelerator,
    transformer,
    batch,
    noisy_model_input,
    timesteps,
    network_dtype,
):
    ...

def soar_velocity_to_standard(self, model_pred):
    ...

def soar_standard_to_local_target(self, clean_latents, aux_latents, aux_sigmas):
    ...

def compute_soar_weighting(self, args, noise_scheduler, timesteps_or_sigmas, device, latent_ndim):
    ...
```

The first implementation can keep these as normal methods on `ZImageNetworkTrainer`. Do not introduce a formal abstract base class until at least two architectures use the contract.

### 5.4 Network Loop Integration

The shared LoRA path should branch only after the normal model forward:

```text
normal input/noise/timestep creation
normal model forward
if not use_soar:
    existing loss path unchanged
else:
    serialized SOAR main + aux path
same optimizer step as today
```

The optimizer step, gradient accumulation boundary, gradient clipping, distributed gradient reduction, and `scale_weight_norms` handling must remain owned by `NetworkTrainer.train(...)`.

The SOAR branch may call `accelerator.backward(...)` multiple times before the same optimizer step. It must not call `optimizer.step()` internally.

## 6. LoRA-Specific Rules

### 6.1 Trainable Parameter Boundary

SOAR must update the same parameters as normal LoRA training:

- base transformer remains frozen except for the injected network behavior
- optimizer params remain `network.prepare_optimizer_params(...)`
- gradient clipping should use `network.get_trainable_params()`

No SOAR helper should change `requires_grad` on base model weights.

### 6.2 Network Hooks

The current network training loop calls:

- `accelerator.unwrap_model(network).on_epoch_start(transformer)`
- `accelerator.unwrap_model(network).on_step_start()`

SOAR auxiliary forwards happen inside the same training step. They should not call `on_step_start()` again, because that hook semantically belongs to one optimizer step, not each auxiliary point.

### 6.3 Gradient Accumulation

Under `accelerator.accumulate(training_model)`, the loss normalization must preserve the existing update scale:

```text
total_count = batch_count + lambda_aux * aux_count
loss_main = sum(main_per_sample) / total_count
loss_aux_i = lambda_aux * sum(aux_per_sample_i) / total_count
```

This keeps the effective gradient magnitude tied to samples and auxiliary points rather than to the number of backward calls.

## 7. Target Model Rollout Order

### Phase 1: `zimage` LoRA

Implement first.

Reasons:

- `zimage_train_network.py` already exposes `predict_velocity(...)`
- existing full-finetune SOAR is the reference behavior
- sign convention is known to be reversed and already handled in the v1 code
- latent shape is 4D, so testing is cheap

Acceptance criteria:

- `--soar` appears in `zimage_train_network.py --help`
- `--soar` absent: generated loss tensor shape and baseline code path are unchanged
- `--soar --soar_trajectory_length 1`: one training step runs and updates only network parameters
- `--weighting_scheme sigma_sqrt/cosmap`: no 5D-to-4D accidental broadcast

### Phase 2: `flux_2` LoRA

Implement second if Phase 1 is stable.

Reasons:

- local target uses standard sign: `noise - latents`
- external latents are 4D
- conditioning is more complex than `zimage` but still contained in `call_dit(...)`

Special care:

- auxiliary points must be generated before `flux2_utils.prc_img(...)`
- control latents must stay unchanged while only noisy target latents vary
- model output must be unpacked back to the same external latent shape before loss

### Phase 3: `qwen_image` Text-to-Image Only

Implement after `flux_2`, and only for non-edit, non-layered mode first.

Reasons:

- local target uses standard sign
- non-layered text-to-image has the smallest state surface

Explicitly reject initially:

- edit mode with `latents_control_*`
- layered mode
- `--remove_first_image_from_target`

These modes change which latent frames are target versus conditioning, so auxiliary target construction needs dedicated tests.

### Out of Scope for This Spec

Do not implement these in the current plan:

- `flux_kontext`
- Wan
- HunyuanVideo
- FramePack
- any other image or video trainer

These may be reconsidered only after the three target trainers have real one-step smoke-test evidence.

## 8. Loss and Weighting Rules

The helper must never blindly multiply an architecture-independent weight tensor into an arbitrary loss tensor.

Required invariant:

```text
weighting.ndim == loss.ndim
weighting.shape[0] == loss.shape[0]
all non-batch weighting dims are 1 or match loss dims
```

For `zimage`, use `get_sigmas(..., n_dim=latents.ndim)` and a shape-normalizing helper.

For standard-sign architectures, the same helper can be used if sigmas are derived from the current scheduler and expanded to the model output rank.

Any architecture whose loss rank differs from latent rank must implement architecture-local weighting.

## 9. Logging and Metadata

Add shared logs only when SOAR is actually active:

- `soar/enabled`
- `soar/lambda_aux`
- `soar/trajectory_length`
- `soar/aux_points_per_sample`
- `loss/main`
- `loss/aux`
- `loss/total`

Checkpoint metadata should record CLI intent:

- `ss_soar`
- `ss_soar_lambda_aux`
- `ss_soar_trajectory_length`
- `ss_soar_num_sampling_steps`

Runtime logs should record actual activation. For example, `--soar --soar_lambda_aux 0` may be metadata-enabled but runtime-disabled.

## 10. Tests

### Unit Tests

Add narrow tests for shared SOAR helper behavior:

- loss normalization with `lambda_aux`
- no rank-changing weighting broadcast
- reversed-sign target for `zimage`
- standard-sign target for `flux_2`-style tensors
- standard-sign target for `qwen_image`-style 5D tensors
- float32 scheduler math retained for bf16/fp16 latents

### Trainer Contract Tests

For each supported model adapter:

- parser accepts `--soar`
- unsupported model modes reject early
- `predict_velocity(...)` returns the same shape as `noisy_model_input`
- auxiliary target returns the same shape as `model_pred`

### Smoke Tests

Minimum smoke runs:

```bash
python src/musubi_tuner/zimage_train_network.py ... --soar --soar_trajectory_length 1 --max_train_steps 1
python src/musubi_tuner/flux_2_train_network.py ... --soar --soar_trajectory_length 1 --max_train_steps 1
python src/musubi_tuner/qwen_image_train_network.py ... --soar --soar_trajectory_length 1 --max_train_steps 1
```

Do not mark a model supported until it has a real one-step run in the intended environment.

## 11. Rejection Rules

Reject `--soar` early when:

- the trainer does not implement `supports_soar(...)`
- the model mode mutates target latent layout in a way the adapter does not explicitly handle
- fused backward or per-parameter optimizer hooks prevent a single normal optimizer step after all auxiliary backward calls
- `points_per_path < 1`
- `num_sampling_steps < 2`
- auxiliary point shape does not match clean latent shape

## 12. Implementation Sequence

1. Move zimage SOAR loss helpers from `zimage_train.py` into `soar_train_utils.py` without changing behavior.
2. Add an idempotent shared parser helper and validation to the network training path.
3. Add `zimage` LoRA support using the existing `predict_velocity(...)`.
4. Run zimage LoRA unit tests and a real one-step smoke test.
5. Add `flux_2` adapter support.
6. Run flux2 LoRA unit tests and a real one-step smoke test.
7. Add Qwen text-to-image adapter support.
8. Run Qwen text-to-image LoRA unit tests and a real one-step smoke test.

## 13. Open Questions

- Should `--soar_lambda_aux 0` mean "parse and record metadata, but do not run auxiliary forwards" or should it be rejected as user error?
- Should LoRA SOAR be allowed with `scale_weight_norms`, or should the first pass reject it until the post-step regularization behavior is checked?
- Should each architecture define its own weighting helper, or should a generic shape-safe weighting helper be sufficient for all flow-matching models?
- Should unsupported edit/control modes reject at parser validation time or at trainer setup time after model mode is known?

## 14. Non-Negotiable Invariants

- `--soar` absent means baseline training behavior does not change.
- SOAR must not change optimizer ownership.
- SOAR must not step the optimizer internally.
- SOAR must not update frozen base weights during LoRA training.
- Auxiliary points must stay in external latent space, not model-packed token space.
- Scheduler math stays in float32.
- Weighting must never change loss rank.
- A model is not documented as supported until a real one-step smoke test exists.

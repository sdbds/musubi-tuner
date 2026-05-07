# SOAR-lite LoRA Targeted Model Implementation Plan

> **For agentic workers:** REQUIRED: Use checkbox (`- [ ]`) syntax for tracking. Keep each phase independently reviewable. Do not start the next model phase until the current phase has unit tests and at least one real one-step smoke run.

**Goal:** Extend SOAR-lite from `zimage` full finetuning to LoRA/network training, then add adapters for exactly three targets: `zimage`, `flux_2`, and `qwen_image`.

**Spec:** `docs/soar_lora_cross_model_spec.md`

**Linus review result:** PASS after tightening the parser rule. Do not add SOAR args blindly to `setup_parser_common()` because full-finetune scripts also use it and `zimage_train.py` already registers SOAR args.

**Architecture:** Keep scheduler and rollout math in `soar_utils.py`. Add `soar_train_utils.py` for shared training concerns: idempotent parser registration, validation, shape-safe weighting, per-sample loss, serialized main/aux backward orchestration. Model trainers expose small adapter methods only where needed.

**Implementation status (2026-04-29):** code and unit-test tasks are complete for `zimage`, `flux_2`, and standard text-to-image `qwen_image`. Real one-step smoke tests are not complete because this workspace does not include the required model and dataset configs.

**Tech Stack:** Python, PyTorch, Accelerate, stdlib `unittest`, Ruff.

---

## File Structure

**Create:**

- `src/musubi_tuner/soar_train_utils.py`
- `tests/test_soar_train_utils.py`
- `tests/test_zimage_soar_lora.py`
- `tests/test_flux2_soar_lora.py`
- `tests/test_qwen_image_soar_lora.py`
- `docs/superpowers/plans/2026-04-29-soar-lora-cross-model-implementation.md`

**Modify Phase 1:**

- `src/musubi_tuner/hv_train_network.py`
- `src/musubi_tuner/zimage_train.py`
- `src/musubi_tuner/zimage_train_network.py`
- `tests/test_support.py`
- `docs/zimage.md`
- `docs/soar_lora_cross_model_spec.md`

**Modify Later Phases:**

- `src/musubi_tuner/flux_2_train_network.py`
- `src/musubi_tuner/qwen_image_train_network.py`

## Phase 0: Guardrails

- [x] Confirm the current full-finetune SOAR tests still pass before touching LoRA.

Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
ruff check src/musubi_tuner/soar_utils.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py tests
```

Expected:

- Existing tests pass.
- Any existing failure must be fixed before adding LoRA code.

- [ ] Record the current baseline diff for files that will be touched.

Run:

```bash
git diff -- src/musubi_tuner/hv_train_network.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py src/musubi_tuner/soar_utils.py tests docs
```

## Phase 1: Extract Shared Training Helpers

**Files:**

- Create: `src/musubi_tuner/soar_train_utils.py`
- Create: `tests/test_soar_train_utils.py`
- Modify: `src/musubi_tuner/zimage_train.py`
- Modify: `tests/test_zimage_soar_aux.py`

- [x] Move shape-safe loss helpers out of `zimage_train.py`.

Move or duplicate first, then delete after tests pass:

- `_compute_zimage_loss_weighting_from_sigma(...)`
- `_compute_per_sample_zimage_loss(...)`
- `validate_soar_args(...)`
- `_run_soar_auxiliary_pass(...)`

Target names in `soar_train_utils.py`:

- `add_soar_arguments(parser)`
- `validate_soar_args(args, *, allow_fused_backward: bool = False)`
- `compute_loss_weighting_from_sigma(weighting_scheme, sigmas, target_ndim)`
- `compute_per_sample_loss(model_pred, target, weighting)`
- `run_soar_auxiliary_pass(...)`

- [x] Make `add_soar_arguments(parser)` idempotent.

Required behavior:

- If `--soar` is not present, add all SOAR args.
- If `--soar` is already present, return the parser unchanged.
- Do not break `zimage_train.py`, which currently registers SOAR args through `zimage_finetune_setup_parser(...)`.

- [x] Add unit tests for parser idempotency.

Test cases:

- calling `add_soar_arguments(...)` once adds defaults
- calling it twice does not raise argparse conflict
- calling it after `zimage_finetune_setup_parser(...)` does not raise conflict

- [x] Add unit tests for shape-safe weighting.

Test cases:

- 4D loss with `[B,1,1,1]` weighting keeps 4D shape
- 5D loss with `[B,1,1,1,1]` weighting keeps 5D shape
- mismatched non-batch dimensions raise or are normalized only when safe

- [x] Update `zimage_train.py` to import shared helpers.

Rules:

- Preserve current full-finetune behavior.
- Do not change full-finetune CLI defaults.
- Keep `--soar` incompatible with `--fused_backward_pass`.

- [x] Verify Phase 1.

Run:

```bash
python -m unittest tests.test_soar_train_utils tests.test_zimage_soar_aux -v
python -m unittest discover -s tests -p "test_*.py" -v
ruff check src/musubi_tuner/soar_utils.py src/musubi_tuner/soar_train_utils.py src/musubi_tuner/zimage_train.py tests
```

## Phase 2: Add Shared Network-Loop SOAR Branch

**Files:**

- Modify: `src/musubi_tuner/hv_train_network.py`
- Modify: `tests/test_support.py`
- Create or modify: `tests/test_zimage_soar_lora.py`

- [x] Add shared SOAR args to network scripts through the idempotent helper.

Implementation rule:

- Call `add_soar_arguments(parser)` from `setup_parser_common()` only if it is proven not to conflict with full-finetune scripts, or call it from each `*_train_network.py` main setup path.
- Preferred first pass: call it from `setup_parser_common()` only after full-finetune duplicate handling is covered by tests.

- [x] Add `NetworkTrainer` default SOAR adapter methods.

Default methods should reject clearly:

- `supports_soar(self, args) -> False`
- `predict_velocity_for_soar(...)` raises `NotImplementedError`
- `soar_velocity_to_standard(...)` raises `NotImplementedError`
- `soar_standard_to_local_target(...)` raises `NotImplementedError`

The shared loop must call validation after `self.handle_model_specific_args(args)` because model mode is known there.

- [x] Add `use_soar` calculation in `NetworkTrainer.train(...)`.

Required logic:

```python
use_soar = args.soar and args.soar_lambda_aux > 0 and args.soar_trajectory_length > 0
```

If `args.soar` and the trainer does not support SOAR, fail before loading unnecessary training state when possible.

- [x] Integrate serialized backward without changing optimizer ownership.

Current non-SOAR path must remain:

```text
loss.mean()
accelerator.backward(loss)
reduce/clip
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad()
```

SOAR path must:

1. compute normal `model_pred, target`
2. compute per-sample main loss
3. backward normalized main loss
4. build auxiliary points with detached rollout under no grad
5. forward each auxiliary point through adapter callback
6. backward normalized auxiliary loss
7. return `loss` for logging only
8. reuse the existing reduce/clip/step/zero_grad block

- [x] Preserve distributed gradient reduction.

The existing reduction iterates `network.parameters()` after backward. Keep it after all SOAR backward calls so main and aux gradients are reduced together.

- [x] Preserve `scale_weight_norms`.

First pass decision:

- Allow it only if the post-step call remains unchanged.
- If behavior is uncertain in tests, reject `--soar --scale_weight_norms` early and document the rejection.

- [x] Add network-loop contract tests.

Use narrow stubs rather than importing all heavy model dependencies:

- parser has SOAR defaults for network training
- unsupported default trainer rejects `--soar`
- helper calls `accelerator.backward(...)` once for main and once per aux point
- optimizer step is not called from SOAR helper

## Phase 3: Add `zimage` LoRA Adapter

**Files:**

- Modify: `src/musubi_tuner/zimage_train_network.py`
- Modify: `tests/test_zimage_predict_velocity.py`
- Create or modify: `tests/test_zimage_soar_lora.py`
- Modify: `docs/zimage.md`

- [x] Implement `supports_soar(...)` for `ZImageNetworkTrainer`.

Return true for the current `zimage` LoRA mode.

- [x] Reuse existing `predict_velocity(...)`.

Adapter wrapper should accept the shared signature:

```python
predict_velocity_for_soar(args, accelerator, transformer, batch, aux_latents, aux_timesteps, network_dtype)
```

It should call existing `predict_velocity(...)` with `batch["llm_embed"]`.

- [x] Implement zimage sign conversion.

Rules:

- Local prediction target is `latents - noise`.
- Standard SOAR velocity is `noise - latents`.
- Rollout velocity should use `-model_pred.detach()`.
- Auxiliary local target should be `(clean_latents - aux_latents) / sigma`.

- [x] Add zimage LoRA unit tests.

Test cases:

- sign conversion matches current full-finetune SOAR behavior
- `predict_velocity_for_soar(...)` returns image-shaped output
- auxiliary target has same shape as model output
- `--weighting_scheme sigma_sqrt/cosmap` does not change loss rank

- [x] Update docs.

In `docs/zimage.md`:

- change SOAR limitation from "full finetuning only" to current support status
- if only `zimage` LoRA is implemented, say only `zimage` LoRA is supported
- keep CFG limitation explicit

- [x] Verify Phase 3.

Run:

```bash
python -m unittest tests.test_zimage_predict_velocity tests.test_zimage_soar_lora tests.test_soar_train_utils -v
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall src/musubi_tuner/soar_utils.py src/musubi_tuner/soar_train_utils.py src/musubi_tuner/hv_train_network.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py
ruff check src/musubi_tuner/soar_utils.py src/musubi_tuner/soar_train_utils.py src/musubi_tuner/hv_train_network.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py tests
```

- [ ] Run real zimage LoRA smoke test when model and dataset configs are available.

Use the existing project environment and a real small config:

```bash
python src/musubi_tuner/zimage_train_network.py ... --soar --soar_trajectory_length 1 --max_train_steps 1
```

Required evidence:

- training starts
- one step completes
- no full transformer parameters are included in optimizer params
- logs include `soar/enabled=1`

## Phase 4: Flux.2 Adapter

**Gate:** Start only after Phase 3 unit tests and real smoke test pass.

**Files:**

- Modify: `src/musubi_tuner/flux_2_train_network.py`
- Add tests for Flux.2 adapter contract
- Modify: `docs/flux_2.md`

- [x] Add `supports_soar(...)` for Flux.2 LoRA.

First supported mode:

- text-to-image or base Flux.2 network training path
- standard target sign
- 4D external latents

- [x] Extract or wrap Flux.2 forward-only prediction.

Keep auxiliary points in external `[B,C,H,W]` latent space and call existing pack/unpack logic inside the forward helper.

- [x] Add Flux.2 tests.

Test cases:

- standard sign target `(aux_latents - clean_latents) / sigma`
- output shape equals external latent shape
- control latents, if present, are passed through unchanged

- [ ] Run Flux.2 smoke test when model and dataset configs are available.

## Phase 5: Qwen Image Adapter

**Gate:** Start only after Flux.2 unit tests and real smoke test pass.

**Files:**

- Modify: `src/musubi_tuner/qwen_image_train_network.py`
- Add tests for Qwen Image adapter contract
- Modify: `docs/qwen_image.md`

First supported mode:

- non-edit
- non-layered
- no `--remove_first_image_from_target`

Early reject:

- edit mode with `latents_control_*`
- layered mode
- `--remove_first_image_from_target`

Must test:

- 5D external latent shape
- pack/unpack round trip via adapter
- standard sign auxiliary target

Do not implement Qwen edit, Qwen layered, or `--remove_first_image_from_target` in this phase.

- [x] Add `supports_soar(...)` for standard Qwen-Image LoRA.
- [x] Reject edit, layered, and `--remove_first_image_from_target`.
- [x] Wrap Qwen-Image forward-only prediction.
- [x] Add Qwen-Image adapter tests.
- [ ] Run Qwen-Image smoke test when model and dataset configs are available.

## Out of Scope for This Plan

Do not implement these adapters in this plan:

- `flux_kontext`
- Wan
- HunyuanVideo
- FramePack
- any other image or video trainer

## Final Verification

Run after each implemented phase:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall src/musubi_tuner/soar_utils.py src/musubi_tuner/soar_train_utils.py src/musubi_tuner/hv_train_network.py
ruff check src/musubi_tuner/soar_utils.py src/musubi_tuner/soar_train_utils.py src/musubi_tuner/hv_train_network.py tests
```

Additional compile targets per phase:

- Phase 3: `src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py`
- Phase 4: `src/musubi_tuner/flux_2_train_network.py`
- Phase 5: `src/musubi_tuner/qwen_image_train_network.py`

## Stop Conditions

Stop and review before continuing if:

- `--soar` absent changes the baseline loss path
- weighting changes loss rank
- optimizer step moves into a SOAR helper
- full transformer parameters appear in LoRA optimizer groups
- gradient accumulation changes the number of optimizer steps
- a model adapter needs to inspect model-packed token state outside its own forward helper
- a real one-step smoke run fails after unit tests pass

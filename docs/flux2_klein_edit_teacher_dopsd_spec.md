# FLUX.2 Klein Edit-Teacher D-OPSD Spec

## Linus Review

Answer the design questions before changing code:

1. Is this a real problem?
   Yes. The current FLUX.2 Klein D-OPSD path distills from cached Qwen3-VL
   teacher context. That gives an image-aware soft condition, but it still does
   not directly ask the FLUX.2 edit branch to reconstruct the target image. If
   the observed failure mode is weak fitting, an edit teacher that sees the
   target image as a reference is a more direct supervision source.

2. Is there a simpler way?
   Yes. Do not add a second teacher DiT and do not cache teacher velocities.
   Reuse the existing FLUX.2 transformer, EMA adapter teacher, and control
   latent path. The teacher condition becomes:

   ```text
   clean target latent as reference/control image
   + fixed identity-edit text context
   ```

   The student condition remains the normal text-only FLUX.2 Klein condition.

3. What will this break?
   It changes the meaning of D-OPSD for FLUX.2. This is no longer the paper's
   Qwen3-VL hidden-state teacher; it is privileged edit-branch distillation.
   It will be slower than the current FLUX.2 D-OPSD teacher because the teacher
   forward includes reference image tokens. It can improve reconstruction while
   hurting prompt-only generalization if the teacher exposes details the student
   prompt cannot infer.

## Goal

Replace the current FLUX.2 Klein D-OPSD teacher with an edit/reference teacher
for:

```text
--model_version klein-4b
--model_version klein-9b
```

The teacher uses the FLUX.2 edit/reference path to reconstruct the target image
from the student's current on-policy rollout state. This is the FLUX.2 Klein
D-OPSD path; do not add a user-facing teacher-mode switch. Z-Image is out of
scope because it has no matching edit/reference branch in this repository.

## Non-Goals

- No Z-Image support.
- No FLUX.2 dev/base or Klein base support.
- No external edit dataset requirement.
- No independent teacher model loaded alongside the student model.
- No teacher velocity cache. Teacher velocities depend on the current student
  rollout state and must be computed online.
- No Qwen3-VL teacher processor or VLM reweighting for FLUX.2 D-OPSD.
- No `--dopsd_teacher_mode` option. FLUX.2 Klein D-OPSD has one teacher
  definition in this branch.
- No support for datasets that already contain `latents_control_*` keys in the
  first implementation. External controls make the identity-reference contract
  ambiguous.

## FLUX.2 D-OPSD Contract

The existing FLUX.2 D-OPSD implementation is replaced in place. The user-facing
training switch remains:

```bash
--dopsd
```

The supported variants remain:

```text
--model_version klein-4b
--model_version klein-9b
```

FLUX.2 D-OPSD no longer uses a Qwen3-VL hidden-state teacher. It uses a normal
Qwen3 text context for a fixed identity edit instruction and uses the target
latent as the teacher reference image.

## Core Algorithm

For each training batch:

1. Load clean target latents and normal student `ctx_vec`.
2. Initialize the D-OPSD rollout state from Gaussian noise with the same shape
   as the target latents.
3. For each FLUX.2 Klein schedule step:
   - Build a teacher batch:

     ```text
     ctx_vec = dopsd_teacher_ctx_vec
     latents_control_0 = clean target latents
     ```

   - Swap the trainable adapter weights to EMA weights.
   - Run the teacher prediction under no-grad/inference mode:

     ```text
     teacher_v = DiT(state_i, timestep_i, edit_teacher_condition)
     ```

   - Restore live student adapter weights.
   - Run the student prediction with the normal text-only condition:

     ```text
     student_v = DiT(state_i, timestep_i, normal_text_condition)
     ```

   - Backward the existing low-allocation `x0` objective:

     ```text
     loss_i = sigma_i ** 2 * mse(student_v, stopgrad(teacher_v))
     ```

   - Advance the rollout with the student prediction:

     ```text
     state_{i+1} = state_i + (sigma_{i+1} - sigma_i) * student_v
     ```

The critical D-OPSD invariant remains unchanged: teacher supervision is computed
on the student's own on-policy state, not on an offline teacher trajectory.

## Cache Contract

The mode does not cache teacher velocities.

Keep the existing FLUX.2 D-OPSD teacher cache key:

```text
dopsd_teacher_ctx_vec_<dtype>
batch key: dopsd_teacher_ctx_vec
```

The key is reused because this branch has not been released and no compatibility
migration is needed. Its contents change: it is now produced by the normal
FLUX.2 Klein Qwen3 text encoder, not by Qwen3-VL. The fixed initial instruction
is:

```text
Reconstruct the reference image exactly. Do not change its content, composition, style, or colors.
```

The same cached tensor can be stored for every sample. This is wasteful in disk
terms, but it matches the repository's per-sample safetensors cache contract and
keeps training text-encoder-free.

Do not add an extra latent cache key for the reference image in the first
implementation. Use the already loaded clean target `latents` as:

```text
teacher_batch["latents_control_0"] = latents.detach()
```

## Variant Mapping

Use the existing FLUX.2 Klein model metadata:

| Model | Student text encoder | Edit teacher text encoder | Teacher reference |
| --- | --- | --- | --- |
| `klein-4b` | Qwen3 4B | same cached Qwen3 4B path | clean target latent |
| `klein-9b` | Qwen3 8B | same cached Qwen3 8B path | clean target latent |

The teacher context dimension must equal:

```text
self.model_version_info.params.context_in_dim
```

## Implementation Plan

1. Replace FLUX.2 D-OPSD text cache generation.
   - Keep the existing `dopsd_teacher_ctx_vec` cache key.
   - Use the normal Qwen3 embedder for the current model version.
   - Encode the fixed identity instruction without images.
   - Save the context under `dopsd_teacher_ctx_vec_<dtype>`.
   - Remove the FLUX.2 requirement for `--dopsd_teacher_text_encoder` because
     FLUX.2 D-OPSD no longer uses Qwen3-VL.

2. Replace FLUX.2 trainer teacher batch construction.
   - Require `dopsd_teacher_ctx_vec` and reject any existing
     `latents_control_*` keys from the dataset.
   - Inject `latents_control_0` from the clean target latents through a closure
     around `make_dopsd_teacher_batch`, so the shared runner does not need a
     broad API change.
   - Keep the student batch unchanged and text-only.

3. Keep the existing schedule and loss.
   - Use `flux2_utils.get_schedule(...)` exactly as the current FLUX.2 D-OPSD
     path does.
   - Keep `dopsd_flow_x0_loss`, currently implemented as
     `sigma ** 2 * velocity MSE`.
   - Keep stepwise backward.

4. Metadata and logging.
   - No new teacher-mode metadata is needed.
   - Log that FLUX.2 D-OPSD uses edit/reference teacher conditioning.
   - Log that target latents are injected as teacher reference controls.

## Validation

Minimum tests before implementation is considered complete:

- Cache script:
  - `--dopsd_cache_teacher_outputs` creates `dopsd_teacher_ctx_vec_<dtype>`.
  - The saved context last dimension matches `context_in_dim` for both
    `klein-4b` and `klein-9b`.
  - FLUX.2 D-OPSD cache generation does not require a Qwen3-VL path.

- Trainer:
  - `--dopsd` rejects `dev`, `klein-base-4b`, and `klein-base-9b`.
  - `--dopsd` rejects batches that already contain `latents_control_*`.
  - Teacher batch uses `dopsd_teacher_ctx_vec` as `ctx_vec`.
  - Teacher batch injects clean target latents as `latents_control_0`.
  - Student batch does not receive control latents.

- Smoke:
  - One LoRA step reaches teacher forward, student forward, backward, optimizer
    step, and EMA update.
  - The returned loss is finite for both 4-step and 8-step schedules.
  - No text encoder or VLM is loaded during training.

- Performance:
  - Record iteration time and peak VRAM against the previous Qwen3-VL-context
    implementation if a local baseline exists.
  - Expect the teacher forward to be slower because reference tokens roughly
    add another image-token sequence to the teacher pass.

## Risk Register

- This is not paper-equivalent D-OPSD. It is an edit-teacher variant that uses
  privileged target-image information.
- It may increase reconstruction and reduce generalization. The student cannot
  infer image details that are absent from the prompt except by memorizing them
  in adapter weights.
- Teacher attention cost increases because the teacher sees both rollout image
  tokens and reference image tokens.
- The fixed identity instruction may matter. If quality is sensitive to wording,
  the instruction should become a cache-time parameter, but not before the first
  controlled experiment.
- Mixed SFT loss is orthogonal. It can be combined later, but the first edit
  teacher implementation should isolate the teacher-condition variable.

## Rejected Alternatives

- Cache teacher velocities.
  This breaks on-policy distillation because teacher velocity depends on the
  current student rollout state and current EMA weights.

- Use the original caption as the teacher instruction.
  That turns the teacher into a reference-guided edit/generation hybrid. The
  first experiment should test the clean identity-edit hypothesis.

- Add Z-Image support by fabricating a reference branch.
  Z-Image has no matching edit/control input path here. That would be a separate
  architecture change, not an adaptation of this idea.

- Allow existing control-image datasets immediately.
  The teacher already needs the target image as privileged reference. Mixing
  external controls into the first implementation makes the supervision contract
  unclear.

- Keep both FLUX.2 teacher paths behind a mode flag.
  This branch has not shipped, and the requested behavior is to replace the
  current FLUX.2 Klein D-OPSD teacher in place. A mode flag would preserve a
  path we do not intend to test or support.

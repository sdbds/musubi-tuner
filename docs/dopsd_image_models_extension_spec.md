# D-OPSD Extension Spec for Qwen3-Text Models

## Goal

Extend the existing D-OPSD path to image models whose normal text encoder is
Qwen3 and whose teacher condition can be projected into the same DiT condition
space. Z-Image uses the paper-style Qwen3-VL teacher. FLUX.2 Klein uses its
edit/reference branch as the teacher condition.

Supported model families in this implementation:

```text
Z-Image
FLUX.2 Klein 4B
FLUX.2 Klein 9B
```

Qwen-Image is intentionally not supported by D-OPSD in this implementation. It
uses Qwen2.5-VL rather than Qwen3 as its text encoder, so the paper's
Qwen3-VL/Qwen3 feature-space alignment assumption does not apply.

## Core Requirements

D-OPSD is valid here only when all of these are true:

- The model's few-step inference update can be reproduced during training.
- The student condition is the normal text-only inference condition.
- The teacher condition has the same shape expected by the DiT but carries
  target-image information.
- For adapter training, student and teacher use the same frozen base DiT and
  the teacher weights are an EMA of trainable adapter parameters.
- For Z-Image full-parameter fine-tuning, student and teacher use the same live
  DiT and the teacher weights are an EMA of all trainable DiT parameters. EMA
  storage defaults to `auto`: GPU when reported free CUDA memory appears
  sufficient, otherwise CPU.
- The rollout step uses the same velocity sign convention as inference.

## Target Matrix

| Model | Status | Notes |
| --- | --- | --- |
| Z-Image | supported | Paper path; Qwen3-4B student TE and Qwen3-VL-4B teacher. |
| FLUX.2 Klein 4B | supported | Qwen3-4B student TE; D-OPSD teacher uses identity-edit Qwen3 ctx plus target latent reference. |
| FLUX.2 Klein 9B | supported | Qwen3-8B student TE; D-OPSD teacher uses identity-edit Qwen3 ctx plus target latent reference. |
| FLUX.2 dev/base | unsupported | Not the few-step Klein setup targeted here. |
| FLUX.2 Klein base 4B/9B | unsupported | Base checkpoints use the slower non-distilled setup and need a separate design. |
| Qwen-Image / Qwen-Image Edit / Layered | unsupported | Different TE family: Qwen2.5-VL, not Qwen3. |
| Video/control families | unsupported | Need separate condition and rollout designs. |

## Shared Training Shape

Each supported trainer supplies:

```text
supports_dopsd(args)
get_dopsd_schedule(args, device, batch, latents)
make_dopsd_teacher_batch(args, batch)
predict_velocity_for_dopsd(...)
dopsd_rollout_step(...)
```

`get_dopsd_schedule` is per batch. Z-Image can ignore `batch` and `latents`,
but FLUX.2 Klein derives its schedule from the current packed image sequence
length, so a global schedule would be wrong for mixed-resolution buckets.

## Teacher Cache Contract

Z-Image teacher cache:

```text
varlen_dopsd_teacher_llm_embed_<dtype>
batch key: dopsd_teacher_llm_embed
```

FLUX.2 Klein teacher cache:

```text
dopsd_teacher_ctx_vec_<dtype>
batch key: dopsd_teacher_ctx_vec
```

For FLUX.2 Klein this key stores a normal Qwen3 identity-edit instruction
embedding, not a Qwen3-VL output. The teacher reference image is injected during
training from the clean target latent. Details are specified in
`docs/flux2_klein_edit_teacher_dopsd_spec.md`.

Teacher cache keys are fixed model contracts, not user-facing knobs.

## Teacher Processor and Reweighting

For Z-Image, the VLM weight source is provided by:

```bash
--dopsd_teacher_text_encoder path/to/qwen3-vl-weights-or-dir
```

The processor/tokenizer is not loaded from that safetensors path. It is loaded
from the official Qwen3-VL repo:

```text
Z-Image: Qwen/Qwen3-VL-4B-Instruct
```

The teacher VLM language-model reweight source is the same `--text_encoder`
used by the Z-Image student path. There is intentionally no separate
reweight-source option. Supplying a different Qwen3 model would create a silent
feature-space mismatch risk.

If the VLM has already been externally reweighted, pass:

```bash
--dopsd_teacher_already_reweighted
```

For ablations only:

```bash
--dopsd_teacher_allow_raw_vlm
```

Qwen3-VL teacher cache generation requires the project-pinned
`transformers==4.57.6`.

FLUX.2 Klein D-OPSD does not use Qwen3-VL. Its teacher cache is generated with
the same Qwen3 text encoder used by the student path.

## FLUX.2 Klein Details

Supported variants:

```text
--model_version klein-4b
--model_version klein-9b
```

Rejected variants:

```text
dev
klein-base-4b
klein-base-9b
control-image batches
```

The teacher cache script encodes a fixed identity-edit instruction and stacks
the same Qwen3 hidden layers as the student `Qwen3Embedder`:

```text
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
ctx = concat(hidden_states[9], hidden_states[18], hidden_states[27])
```

During training, the teacher batch replaces `ctx_vec` with this cached identity
edit context and injects the clean target latent as `latents_control_0`. The
student batch remains text-only.

The D-OPSD schedule matches inference:

```text
times = flux2_utils.get_schedule(dopsd_num_sampling_steps, image_seq_len, flow_shift)
model_timesteps = times[:-1] * 1000
rollout_sigmas = times
```

The shared parser default `--discrete_flow_shift 1.0` is treated as "not
explicitly provided" for FLUX.2 D-OPSD so that FLUX.2's empirical default shift
is preserved.

Rollout uses the standard flow sign convention:

```text
state = state + (sigmas[i + 1] - sigmas[i]) * student_pred
```

Z-Image keeps its existing sign convention through its model-specific rollout
hook.

The D-OPSD loss uses the same `x0` target as Z-Image, expressed in the
equivalent low-allocation form:

```text
loss = sigma ** 2 * mse(student_velocity, stopgrad(teacher_velocity))
```

## Validation

Fail early when:

- `--dopsd` is set on an unsupported model variant.
- The expected teacher cache key is missing.
- Teacher hidden dimension does not match the student condition dimension.
- FLUX.2 Klein teacher layer stack does not match the student layer stack.
- FLUX.2 Klein control-image batches are used with `--dopsd`.
- A Z-Image Qwen3-VL path is used with a transformers version older than `4.57.6`.

## Risk Register

- FLUX.2 Klein 9B depends on a matching Qwen3-8B student text encoder for both
  the normal student context and the identity-edit teacher context.
- FLUX.2 Klein D-OPSD is a D-OPSD derivative, not the paper's Qwen3-VL teacher
  path. The teacher sees the target latent as a privileged reference, which may
  improve fitting while increasing memorization risk.
- Full-parameter Z-Image D-OPSD uses `auto` EMA placement by default. CPU
  fallback is much slower than adapter D-OPSD and consumes a full extra
  DiT-sized CPU/RAM copy. `--dopsd_full_ema_device gpu` can force faster swaps
  at the cost of roughly one extra trainable DiT copy of VRAM; `cpu` forces the
  lower-VRAM slower path.
- Full-parameter D-OPSD rejects fused backward because the fused optimizer steps
  parameters during backward, while D-OPSD needs one optimizer step after all
  rollout-step backward calls.

# Z-Image D-OPSD Official Alignment Spec

## Linus Review

Answer the three design questions before changing code:

1. Is this a real problem?
   Yes. The local D-OPSD trainer still uses velocity loss, while the current
   official implementation uses `x0` loss. This changes the gradient weighting
   across rollout steps and can plausibly explain slow or poor LoRA fitting.

2. Is there a simpler way?
   Yes. Do not redesign D-OPSD. Align the Z-Image path to the official training
   equations first: official schedule, official `x0` loss, official Qwen3-VL
   processor settings, and the existing EMA teacher semantics.

3. What will this break?
   Existing experimental D-OPSD runs will no longer be numerically comparable to
   new runs. Teacher caches generated before the processor/token cap alignment
   may remain loadable, but they are not official-equivalent and should be
   regenerated for controlled tests.

## Goal

Bring the local Z-Image D-OPSD implementation into behavioral alignment with
the current official repository:

```text
https://github.com/vvvvvjdy/D-OPSD
official commit checked: 4a4cd7c1e2683db0e21addcf3599914ee269889b
```

The first implementation target is Z-Image Turbo adapter training. Full
fine-tuning should keep using the same shared D-OPSD primitives when the change
is mathematically identical. FLUX.2 Klein uses the same `x0` objective with its
own flow sign convention, but still keeps its own schedule because the official
repository has not published FLUX.2 training code.

## Non-Goals

- No Qwen-Image support. Its text encoder family is not Qwen3.
- No raw, unreweighted VLM as the paper-consistent path.
- No forced migration to PEFT dual adapters. The local EMA shadow implementation
  is acceptable if it produces the same teacher weights.
- No online VLM/text encoder execution during training. Local cache-first
  training is a valid implementation optimization.
- No change to normal SFT, normal LoRA, or normal full fine-tuning when
  `--dopsd` is not set.

## Official Behavior To Match

### Teacher context

Official code loads `Qwen/Qwen3-VL-4B-Instruct`, then copies the student
Qwen3 text encoder weights into:

```text
vl_model.model.language_model
```

Local single-file reweighted VLM checkpoints are equivalent if the language
submodule has already been replaced with the same student Qwen3 weights.

The official processor is created with:

```text
min_pixels = 512 * 512
max_pixels = 768 * 768
```

Teacher hidden states use:

```text
hidden_state_layer = -2
max_sequence_length = 1024
use_system_prompt = False
add_generation_prompt = True
```

The local cache script must match this for newly generated teacher caches.

### Student context

Student text embeddings use Qwen3 hidden state `-2`, the normal Z-Image chat
template, and `enable_thinking=True`. The current local student text cache path
already matches this requirement.

### Rollout start

Official training starts each D-OPSD rollout from Gaussian noise with the same
latent shape as the target image. The target latent is only used to determine
shape and to provide the image to the VLM teacher during cache generation.

The current local behavior already starts from Gaussian noise and should remain
that way.

### Teacher input state

The teacher must be evaluated on the current student rollout state, not on a
separate teacher trajectory:

```text
teacher_pred = teacher(state_student, teacher_context)
student_pred = student(state_student, student_context)
```

The local path already follows this invariant.

## Schedule

Official Z-Image D-OPSD supports only 4-step and 8-step training schedules.

For 4 steps, model-time values are:

```text
[0.0, 0.1000000015, 0.25, 0.5]
```

For 8 steps, official code starts from Z-Image Turbo inference timesteps and
converts them to model-time values:

```text
[0.0, 0.0231009, 0.0522353, 0.0901218,
 0.1414013, 0.2147002, 0.3280788, 0.5267797]
```

Local `predict_velocity` receives scheduler-style timesteps and converts them
inside the Z-Image trainer:

```text
model_time = (1000 - timestep) / 1000
```

Therefore the Z-Image D-OPSD schedule helper should return:

```text
timestep = (1 - model_time) * 1000
sigma = 1 - model_time
```

with a final rollout sigma of `0`.

The current shifted linspace schedule happens to approximate the official
4-step schedule when `--discrete_flow_shift=3.0`, but it does not match the
official 8-step schedule. Do not keep an approximate schedule for official
alignment.

## Loss

Official code no longer uses velocity loss. It computes:

```text
x0_teacher = state + sigma * teacher_velocity
x0_student = state + sigma * student_velocity
loss = mse(x0_student, stopgrad(x0_teacher))
```

where:

```text
sigma = 1 - model_time
```

In local Z-Image terms this is:

```text
sigma = sigmas[step_index]
x0_teacher = state + sigma * teacher_pred
x0_student = state + sigma * student_pred
loss = mse(x0_student.float(), x0_teacher.float().detach())
```

This is exactly equivalent to velocity loss weighted by `sigma ** 2`:

```text
loss = sigma ** 2 * mse(student_velocity, stopgrad(teacher_velocity))
```

because the shared `state` term cancels and the sign disappears under MSE. The
implementation may use this weighted-velocity form to avoid materializing two
extra DiT-sized `x0` tensors, as long as validation proves it matches the
explicit `x0` formula.

## Backward Strategy

Official code accumulates all per-step losses and calls backward once after the
rollout loop. Local code calls backward once per step and detaches the rollout
state between steps.

Keep local stepwise backward. Because the state is detached at each step, this
preserves the intended no-BPTT behavior while reducing peak VRAM. The important
alignment point is the loss formula, not whether backward is called once or K
times.

## EMA Teacher

Official LoRA code uses two adapters:

```text
student: trainable
teacher: frozen EMA copy
```

Local code keeps one EMA shadow of trainable parameters and temporarily swaps
those weights into the network for teacher forward. This is acceptable if:

- EMA initializes from the initial trainable weights.
- Teacher forward is run under no-grad/inference mode.
- Live student weights are restored before student forward.
- EMA updates only after the optimizer step.

This avoids adding a second adapter system just to mirror PEFT structure.
Adapter EMA should cache trainable parameter references after initialization and
refresh that cache only if the trainable parameter set changes; the parameter
set is stable during normal training, so rebuilding a name-to-parameter map on
every teacher step is avoidable overhead.

For full-parameter fine-tuning, EMA storage has three tiers:

```text
auto: default; use GPU only when reported free CUDA memory is sufficient, otherwise CPU
cpu:  lowest VRAM, slower teacher swap
gpu:  full EMA copy on the training device, faster but costs roughly one extra trainable DiT copy
```

## Implementation Plan

1. Add a Z-Image official D-OPSD schedule helper.
   - Accept only `4` and `8` for official alignment.
   - Return local scheduler timesteps plus rollout sigmas.
   - Keep non-D-OPSD schedules untouched.

2. Add a model-specific D-OPSD loss hook.
   - Shared runner default may remain velocity loss for unsupported/experimental
     models.
   - Z-Image must override it with official `x0` loss, implemented either
     explicitly or as the equivalent `sigma ** 2` weighted velocity loss.
   - FLUX.2 Klein may use the same objective with its own flow sign convention.

3. Update `run_dopsd_stepwise_backward`.
   - Compute teacher and student predictions as now.
   - Delegate loss computation to the hook.
   - Keep stepwise backward and detached student rollout.

4. Align Qwen3-VL teacher cache generation.
   - Load the official processor with `min_pixels=512*512` and
     `max_pixels=768*768`.
   - Truncate cached teacher hidden states to 1024 tokens.
   - Keep hidden state index `-2`.
   - Keep `add_generation_prompt=True`.
   - Keep `enable_thinking=True` when supported by the processor.

5. Update D-OPSD docs and examples.
   - Recommend `--dopsd_num_sampling_steps=4` for Z-Image Turbo official-style
     training.
   - Explain that 8-step is available only through the official hardcoded
     schedule, not through shifted linspace.

## Compatibility Rules

- Existing normal training paths must produce the same code path when
  `--dopsd` is false.
- Existing D-OPSD cache files may be loadable, but tests intended to compare
  with official behavior should regenerate teacher caches after this change.
- Do not change LoRA target module selection in this alignment patch unless a
  concrete mismatch is proven to affect correctness. Target coverage is a
  separate adapter-quality question, not the cause of the current equation
  mismatch.
- Do not change FLUX.2 D-OPSD loss/schedule to official Z-Image values. There is
  no official FLUX.2 implementation to compare against yet.

## Validation

Minimum validation before implementation is considered complete:

- Unit or script-level check that Z-Image D-OPSD 4-step model-time values equal:

```text
[0.0, 0.1000000015, 0.25, 0.5]
```

- Unit or script-level check that Z-Image D-OPSD 8-step model-time values equal:

```text
[0.0, 0.0231009, 0.0522353, 0.0901218,
 0.1414013, 0.2147002, 0.3280788, 0.5267797]
```

- Tensor check for the loss hook:

```text
loss == mse(state + sigma * student_pred,
            state + sigma * teacher_pred.detach())
loss == sigma ** 2 * mse(student_pred,
                         teacher_pred.detach())
```

- Cache check that newly saved Z-Image teacher embeddings are variable-length
  tensors with:

```text
sequence_length <= 1024
hidden_dim == 2560
cache key == varlen_dopsd_teacher_llm_embed_<dtype>
```

- Smoke test that `--dopsd` on Z-Image LoRA still reaches forward, backward,
  optimizer step, EMA update, and sampling without loading the VLM during
  training.

## Rejected Simpler-Looking Alternatives

- Just lower `--dopsd_num_sampling_steps` to 4.
  This improves speed but leaves the wrong loss objective in place.

- Keep velocity loss because it is in the paper.
  The current official training code explicitly changed to `x0` loss for faster
  convergence. If the target is official-code consistency, the implementation
  must follow the code.

- Switch the whole local implementation to PEFT dual adapters.
  That adds a second adapter framework without fixing the actual mismatch.

- Reuse the Z-Image official schedule for FLUX.2.
  The `x0` objective is algebraic and model-agnostic, but the schedule is
  model-specific. FLUX.2 must keep deriving its schedule from the packed image
  sequence length and flow shift.

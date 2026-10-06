# DLSS-NR 310.8.0 Training Support

## Overview

This is an experimental, standalone supervised-training path for DLSS-NR 310.8.0,
not a diffusion trainer. Native DLL equivalence remains unverified. It provides
canonical weight conversion, checked DLL weight import/export, default FP32
full/LoRA training, still/closed-loop inference, adapter merging and checked
optimizer-state resume. The two dataset examples enable the
project's shared resolution buckets with a 1024 x 1024 area budget and `bucket_no_upscale` enabled.
Set `enable_bucket = false` for the original strict fixed-resolution behavior.
Each input and its target, controls, motion and masks must share the original pixel grid.

Users must provide their own weights, paired RGB data, fixed condition settings or encoded control lanes and,
for non-reset temporal frames, motion and validity masks. No weights, DLLs or game
assets are included. See [source attribution](../src/musubi_tuner/dlssnr/NOTICE.md).

Normal full/LoRA training requires pretrained canonical weights, complete
conversion provenance and a successful source round-trip, but no forward-validation
report. Optional evidence is checked only when explicitly supplied; training
eligibility does not certify native/DLL compatibility. `--development_smoke`
remains a developer-only option for incomplete provenance or random initialization.
The surrogate uses half/E4M3 forward
publications with surrogate gradients and FP32 master weights; it is not ordinary
FP32 SiLU/softmax and is not bit-identical to the DLL. Non-reentrant gradient
checkpointing and single-node DDP retain the default numerical profile.
CUDA FP16/BF16, LoRA-only FP8 frozen storage and alternative attention require
explicit `--numerics_profile train_experimental`. SDPA supports windows and
global attention; optional FlashAttention is global-only, xFormers is capability
checked, and SageAttention is inference-only. Missing or incompatible optional
extensions fail explicitly, without fallback to another named backend.
Outputs precede the DLL's Natural/Cinematic color grading.

Training writes all artifacts under `output_dir/output_name/`. A new run refuses
an existing run directory; use `--resume` for a matching checkpoint. Sample IDs
must be portable filenames, unique without regard to case. The commands and
manifest contract below apply to both full and LoRA training.

Use `--dataset_config` for a **dataset-only TOML** with `[general]` and one or more
`[[datasets]]` entries. Model, optimizer, LoRA, loss, evaluation and output settings
are CLI arguments, using the project's standard names such as `--optimizer_type`,
`--optimizer_args`, `--learning_rate`, `--network_dim` and `--network_alpha`.
The old all-in-one `--config_file` training interface is no longer accepted.
Dataset-relative paths resolve against the TOML directory; CLI paths resolve
against the working directory. The generated `run_config.json` is an effective
configuration snapshot for provenance/resume, not another user configuration file.

Paired supervision uses FP32 master weights and matrix accumulation by default.
Intermediate activations, attention and reductions include explicit half/E4M3
fake quantization and surrogate gradients. Experimental modes can change matrix
precision and attention kernels, but require separate native-equivalence checks.
Output remains in proxy color space, before the DLL's Natural/Cinematic grading.

## Configuration

As with other architectures in this project, datasets use `--dataset_config`
and training hyperparameters use CLI arguments. Single-frame full and LoRA
training share the [single-frame dataset example](../configs/dlssnr_dataset_single.toml).
Temporal training uses the [temporal dataset example](../configs/dlssnr_dataset_temporal.toml).

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "../data/target"
control_directory = "../data/input"
num_repeats = 1
nr_controls_mode = "fixed"
nr_style = 0
nr_tone = 1.0
nr_structure = 1.0
nr_skin = -1.0
nr_auto_mask = true
# validation_manifest = "../data/validation_single.jsonl"
# sequence_manifest = "../data/validation_sequences.jsonl"
```

Each `[[datasets]]` entry can override `resolution`, `batch_size`, `enable_bucket`,
`bucket_no_upscale` and `num_repeats` from `[general]`. Datasets retain their own
batch sizes, repeat counts, dimensions and conditions. Directory mode reuses the
project's image-editing source-matching rules, but each target must match exactly
one input control image. Missing or ambiguous matches and mismatched pixel grids
raise errors; the loader never picks the first match or resizes inputs independently.

The default `resolution` is `[1024, 1024]`; existing explicit dimensions are
unchanged. With bucketing enabled, this is an area budget, not a requirement to
stretch every image into a square.

The five condition channels are generated at load time from `nr_style`, `nr_tone`,
`nr_structure`, `nr_skin` and `nr_auto_mask`, without additional JSONL/NPY files.
Tone and structure range from `0` to `1`; skin accepts `[0,1]` or `-1` to follow
structure. Auto mask is a model condition: it does not generate supervision masks
or add a segmentation model. Style changes the network condition only; output
still excludes the DLL's postprocessing color grading.

Existing data can still use `train_manifest` instead of the two directory fields.
Its default `nr_controls_mode = "files"` reads the existing `controls_path`.
Selecting `fixed` explicitly overrides `controls_path` in that dataset and its
validation manifests with the TOML conditions; overridden NPY files are not read.
Temporal data still requires a manifest declaring frame order, motion and validity
masks. Ordinary directory images are not automatically treated as clips.
`caption_extension` and `cache_directory` may remain in shared dataset settings,
but NR reads neither captions nor diffusion caches.

Legacy TOML training sections such as `[model]`, `[training]`, `[optimizer]`,
`[lora]`, `[loss]` and `[output]` raise errors rather than providing a second source
of hyperparameters. The generated `run_config.json` stores fully resolved values
for provenance and resume validation only.

## Bucketing and Paired Transforms

Bucketing uses the shared `BucketSelector`, and same-bucket batches use the public
sample-selection interface of `BucketBatchManager`. Neither reads diffusion latent
or text-encoder caches. NR has its own architecture ID, `nr`, with a bucket step of
16 and a minimum of 48 pixels on each axis. Internal padded-field geometry and
forward computation are unchanged.

- With `enable_bucket = true`, the product of the `resolution` dimensions is the area budget. Bucket selection follows the original aspect ratio. For example, a `128 x 128` budget maps a `320 x 192` input to a `160 x 96` bucket; portrait inputs use the corresponding portrait bucket.
- `bucket_no_upscale = true` follows the shared project behavior. Images within the area budget are rounded down to multiples of 16 without enlargement. If either resulting axis is below 48, the sample is rejected rather than enlarged or padded. Images over budget still select aspect-ratio candidates and use cover resizing/cropping; extreme aspect ratios can enlarge the shorter axis, so this flag does not prohibit every possible upscale.
- Omitting `enable_bucket` keeps it disabled and retains strict fixed-resolution behavior. All paired tensors must already match `resolution`; mismatches raise errors. The example TOMLs explicitly enable bucketing and no-upscale.
- Paired tensors must align on the original pixel grid. Every frame in a clip must share the original input dimensions, bucket, aspect-preserving cover resize and center-crop position. There are no per-frame random crops or independent target/control resizes to hide bad pairing.
- RGB, encoded controls and continuous loss masks share FP32 bilinear/antialias resizing. Binary history/temporal masks use pixel-center-aligned nearest-exact resizing to remain binary. Motion uses bilinear resizing; x/y displacement is scaled by the actual rounded width/height ratios. Identical current/previous-frame crop offsets cancel out, and the reprojection inside mask still excludes history outside the crop.
- A microbatch contains only samples with matching dimensions and frame counts. Different buckets can participate in one gradient-accumulation window; loss remains normalized by the total valid pixel count. Each bucket's short final batch is retained without dropping or duplicate padding. `consumed_samples` records the actual count.

Indexing reads image headers or NPY mmap shapes to select buckets without keeping
pixels resident. Before training, each sample is decoded to validate pairing,
values and the remaining supervision region after transformation. Training and
validation share transform rules. The `bucket_plan` in `run_config.json` records
the actual buckets, samples per bucket, batches per epoch and a fingerprint of
the deterministic batch order. NR retains fixed bucket/manifest order and
deterministic center crops, with no added random shuffling or augmentation.

Bucketing does not establish native-output equivalence: enhancing an image before
resizing generally differs from enhancing it afterward. To preserve native paired
pixel scale, use no-upscale and source dimensions aligned to 16 within the area
budget. Resized pairs are a training-data transform, not evidence of DLL forward parity.

## Current Scope

Root scripts remain thin entry points. Model code, numerics, data, training losses
and evaluation live in `src/musubi_tuner/dlssnr/`; LoRA lives in
`networks/lora_dlssnr.py`, and the training lifecycle in `training/dlssnr_trainer.py`.
Full and LoRA training share `NRTrainModule`, Accelerate wrapping and the same
update loop. They do not call the diffusion trainer.

- The default is single-process FP32, automatically using CUDA when available and CPU otherwise. `--device` accepts `auto`, `cpu` or `cuda`; explicitly requesting unavailable CUDA raises an error. Mixed precision requires CUDA. DDP supports multiple processes on one machine.
- Single-frame training and one finite temporal segment support full/LoRA gradient accumulation, periodic saves and resume. Accumulation is normalized by the total valid element count; steps count optimizer updates.
- Dataset configuration and CLI arguments are strictly validated. Manifest paths resolve relative to the TOML directory; CLI paths resolve relative to the working directory. Resolved configuration and source fingerprints are saved in `run_config.json`.
- Dataset `validation_manifest` and `sequence_manifest` entries are evaluated with independent student history. Evaluation compares the initial base by default and writes `evaluation/stepNNNNNN.json`, without changing training RNG or dropout state.
- Manifests are decoded on demand rather than keeping the entire training set resident. Inference and evaluation replay frames in order.

The native reference forward pass and compatibility/quality acceptance on real
clips remain incomplete. Layout byte round-trips, floating-point training tests
and native compatibility establish different things.

The first comparison on 2026-09-24 used Steam screenshots and a local GPU-adapted
310.8.0 DLL. It found that **the FP32 forward pass did not match**. Four screenshots
and 19 image/parameter combinations covered style, intensity, tone, structure,
skin and auto mask. Default-parameter MAE against the DLL ranged from 0.0294 to
0.0420. Identical packed weights do not prove forward correctness or compatibility
of trained artifacts with the native DLL. See the
[screenshot forward experiment](dlssnr_forward_experiment_2026-09-24.md).

Later that day, the main default-parameter factors were ruled out and the cubic
activation and key numerical boundaries were corrected. Default-parameter MAE
across the four images fell to 0.00262-0.00520. Nine of 19 combinations met the
original three display-metric thresholds, including one zero-intensity passthrough
case. **Full forward acceptance remained incomplete.** The full suite passed 172
tests, including real-weight CUDA full/LoRA updates and merged-output comparisons.
See the [trainable forward alignment report](dlssnr_forward_alignment_2026-09-24.md).
The numerical implementation version changed; older training states cannot resume
exactly across that boundary.

## Training Inputs and Validation Evidence

Normal full and LoRA training require `--model_dir`, canonical weights, a matching
schema/profile, complete source files and a conversion report with a passing
source round-trip. Actual data and tensors are still validated. **A forward
validation report is no longer a training prerequisite**, so development mode is
unnecessary for that reason. Missing model or provenance files stop the run;
there is no implicit random initialization or provenance bypass.

`--development_smoke` is reserved for explicit developer smoke tests. It allows
incomplete provenance, or random weights when `--model_dir` is omitted. It is not
a normal-training option and cannot bypass an invalid report supplied explicitly.

To bind validation evidence, pass `--forward_validation_report` explicitly. The
report must match `--model_dir`; missing paths, identity mismatches and failed checks
raise errors even in development mode. Normal training does not automatically read
`forward_validation_report.json` from the model directory, so an unrequested stale
report cannot change training or resume behavior. Tool callers can require evidence
with `inspect_canonical(..., require_forward_validation=True)`; only that call uses
the canonical directory's default report when no `validation_report` path is given.

The report contract remains `dlssnr_forward_validation_v1`, containing `profile`,
`numerics_profile`, the current `model_sha256`, `implementation_sha256`, a fixed
`reference_identity`, `float_validated` and `checks`. The checks are `raw_head`,
`neural_preclamp` and `rendered_proxy`, all backed by actual reference validation.
The implementation fingerprint is `json_sha256(implementation_identity())`.
Ordinary training evaluation cannot replace this report. Do not fill in passing
flags manually.

Run metadata records whether a valid base report was checked in
`source_forward_validated`. Without a report, it is `false` and
`experimental_surrogate=true` remains set, but `development_smoke` is not enabled
automatically. Developer smoke artifacts also retain the experimental flag.
Regardless of base evidence, newly trained artifacts set `float_validated`,
`temporal_validated` and `native_export_validated` to `false`: evidence for the base
does not prove compatibility of updated weights.

## Memory and Runtime Modes

When GPU memory is limited, try `--gradient_checkpointing` first. It recomputes
FFN/attention during gradient-enabled training segments, covering boundary blocks,
the encoder, ViT and decoder. Burn-in and evaluation are not recomputed. Full and
LoRA training both support it; frozen inputs do not cut off LoRA gradients, and
dropout RNG state is restored during recomputation. It is disabled by default and
does not require a different numerical profile.

For other options, select experimental mode explicitly:

```text
--numerics_profile train_experimental --mixed_precision bf16 --gradient_checkpointing
```

FP16/BF16 apply only to selected projections and matrix multiplications. Training
parameters, LoRA deltas, explicit half/E4 publications, reductions, loss and history
retain their FP32 boundaries. FP16 uses GradScaler, unscaling before checking or
clipping gradients. On overflow it reduces the scale and replays the same effective
batch and RNG state without advancing update/sample counts. `--max_overflow_retries`
defaults to 16. Nonfinite forward values or exhausted retries stop the run without
publishing a successful checkpoint. BF16 does not use a scaler. Accelerate's
mixed-precision environment settings must match the explicit CLI arguments.

FP8 currently compresses only the frozen projection base for LoRA:

```text
--numerics_profile train_experimental --gradient_checkpointing --fp8_base --fp8_scaled
```

`--fp8_scaled` uses the shared block-64 quantizer, with per-output-channel scaling
when the input width is not divisible by 64. Without it, weights use direct E4M3FN
storage and values outside `[-448,448]` are rejected. The input adapter, RGB/logit
heads, priors and scalars retain their original representation. Computation
materializes an FP32 base before adding the LoRA delta; **this does not execute
FP8 GEMM**. Checkpointing reduces the need to retain temporary materialized matrices
through backward. Artifacts bind the original base, effective quantized base and
scale identities. A quantized-base adapter cannot be added to an original base
that has not undergone the same quantization.

Attention options are mutually exclusive. The default remains the custom NR operator:

| Option | Scope and Restrictions |
| --- | --- |
| `--sdpa` | Experimental window/global attention, retaining window priors. PyTorch chooses its internal kernel; a Flash kernel is not guaranteed. |
| `--xformers` | Probes the required dtype, bias and backward capabilities. Unsupported combinations raise errors; scope can be limited explicitly with `--attention_scope global`. |
| `--flash_attn` | Requires FP16/BF16 and explicit `--attention_scope global`. Windows retain the NR operator and their learned priors. |
| `--sage_attn` | Currently supports only FP16/BF16 global inference. Training is rejected; no substitute backward is supplied. |
| `--attention_backend native` | Explicitly selects the NR operator and can override a saved experimental backend during inference. |

Experimental attention uses standard softmax with `scale=1.0` on preprocessed
Q/K/V, without another `1/sqrt(32)` factor. Zero tokens outside windows still
participate in normalization with priors; artificial alignment padding in global
layers is excluded as real keys. This differs from NR's half/E4 exponentials and
reductions and requires separate output-quality checks. Optional extensions load
on demand. Missing installations or ABI mismatches do not silently select another backend.

### Multi-GPU

For example, launch two processes on one machine as follows. Other training
arguments are the same as for a single GPU:

```bash
torchrun --standalone --nproc_per_node=2 dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_single.toml --model_dir models/canonical_dlssnr \
  --gradient_checkpointing --gradient_accumulation_steps 2 \
  --output_dir output/dlssnr_ddp --output_name full_ddp --save_state
```

A correctly configured `accelerate launch` is also supported. Linux CUDA uses
NCCL; Windows/CPU use Gloo initialization. Each GPU still holds a complete model.
DDP does not shard the model or make a larger model fit on an individual GPU.

Batch positions use `(update * accumulation + micro) * world_size + rank`.
Short final batches are not padded, and the deterministic batch plan continues
into the next epoch. The loss denominator is the total valid pixel count across
the accumulated global batch, rather than an average of per-rank means. The main
process handles shared weights, logs and evaluation; each rank's RNG/scaler is
saved. Resume requires the same world size, data plan and runtime policy. Automatic
resharding across world sizes, FSDP, ZeRO, DeepSpeed and multi-node execution are unsupported.

### Local Measurements

Measurements on 2026-10-01 used an RTX 4090, PyTorch `2.13.0+cu130`, the same real
canonical weights, one `512 x 512` effective frame (internal field `576 x 512`),
batch size 1, seed 42 and AdamW. LoRA used ViT rank 16. Each mode ran in its own
process with one warm-up update followed by two measured updates. Memory values
are PyTorch **peak allocated** GPU memory, excluding other processes. The small
timing sample is only a local comparison.

| Mode | Peak GiB | Median Update Time, Seconds |
| --- | ---: | ---: |
| Full FP32 | 15.92 | 1.96 |
| Full FP32 + checkpoint | 4.52 | 2.90 |
| Full BF16 + checkpoint | 4.46 | 3.17 |
| Full FP16 + checkpoint | 4.46 | 4.65 |
| Full BF16 + checkpoint + SDPA | 3.95 | 3.12 |
| LoRA FP32 | 8.17 | 1.39 |
| LoRA FP32 + checkpoint | 2.96 | 1.94 |
| LoRA BF16 + checkpoint | 2.91 | 2.14 |
| LoRA BF16 + checkpoint + scaled FP8 | 2.52 | 2.35 |

Checkpointing reduced memory by about 72% for full training and 64% for LoRA.
Both FP32 on/off comparisons had identical losses at the measured steps. FP16 had
one overflow retry within the measured interval, included in its timing.
Checkpointing provided the largest memory saving here; mixed precision added a
smaller saving and did not improve speed on this machine. Experimental-mode losses
differed. These synthetic-input tests do not establish learning quality or native parity.

To repeat the measurements, run each mode in a separate process with a distinct output file:

```bash
python tools/benchmark_dlssnr_runtime.py --model_dir models/canonical_dlssnr \
  --output output/benchmark-checkpoint.json --width 512 --height 512 \
  --gradient_checkpointing --warmup 1 --steps 2
```

Local checks covered real-weight CUDA AMP/SDPA updates and FP8 LoRA updates/merges.
Two-process CPU/Gloo tests covered unequal valid pixel counts, short final batches
and resume. This machine has one GPU, so real two-GPU execution was not verified.
The isolated test environment lacked FlashAttention, SageAttention and xFormers;
their actual CUDA kernels remain unverified. Installing an extension does not
establish capability or quality acceptance.

## DLL Unpacking and Repacking

Unpack a user-supplied original DLL directly into the existing canonical training
format, without depending on MLX or changing the training network:

```bash
python dlssnr_unpack_model.py --source_dll /path/to/original/nvngx_dlssnr.dll --output_dir models/canonical_dlssnr
```

The entry point also accepts the conversion scripts' usual `--input` / `--output`
aliases. It currently supports only the audited DLSS-NR 310.8.0 `WEIGHTS_HT`
resource, with SHA256
`836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`.
Other resources are rejected; filenames are not used to infer versions. The source
DLL is neither executed nor modified, and the output model directory must be new.
Unpacking always verifies the canonical byte round-trip and saves the provenance
files required for training. Existing model directories converted from the same
source by `dlssnr_convert_model.py` remain supported.

After full training, or after merging LoRA separately, repack the complete model directory:

```bash
python dlssnr_pack_model.py --template_dll /path/to/original/nvngx_dlssnr.dll --model_dir output/full/run/final --output_dll output/export/nvngx_dlssnr.dll --mix 1.0 --strength 1.0
```

LoRA can also be merged during repacking, without manually creating an intermediate
model directory. In this case, `--model_dir` must identify that adapter's matching
base, not a different fine-tuned model:

```bash
python dlssnr_pack_model.py --template_dll /path/to/original/nvngx_dlssnr.dll --model_dir models/canonical_dlssnr --merge_lora output/lora/run/final/adapter.safetensors --lora_multiplier 0.8 --mix 1.0 --strength 1.0 --output_dll output/export-lora/nvngx_dlssnr.dll
```

`--merge_lora` takes an adapter file path; `--lora_weight` is an alias. Only this
project's DLSS-NR LoRA format is accepted, one adapter per invocation. Existing base
identity and FP8 quantized-base checks remain in place; arbitrary safetensors files
are not treated as compatible adapters. The temporary merge directory is removed
automatically, and neither the original base nor the adapter is modified.

The three controls are independent. They apply in this order: LoRA merge, mix,
strength, then native quantization.

| Option | Default | Meaning |
| --- | --- | --- |
| `--lora_multiplier` | `1.0` | Scales only the LoRA delta. A nondefault value requires `--merge_lora`. |
| `--mix` | `1.0` | Interpolates between the original DLL weights and the full trained/merged result, within `[0,1]`. |
| `--strength` | `1.0` | Multiplies all decoded numeric weights after mixing. Alias: `--multiplier`. |

```text
W_effective = W_trained                              # full checkpoint
W_effective = W_lora_base + lora_multiplier * delta   # adapter merge
W_export = quantize_native(strength * (W_original + mix * (W_effective - W_original)))
```

For FP8 LoRA, `W_lora_base` is the effective quantized base bound to the adapter.
`--mix 0 --strength 1` restores the template DLL's original weights.
`--strength 0` zeros decoded numeric weights; it does not restore the original
model. Strength scales matrices, priors, scales and other decoded floating-point
values, rather than controlling a visual effect in an inference UI. All three
controls must be finite. Strength and the LoRA multiplier may be negative or exceed
1, but final values must fit the finite range of their native storage types.
Opaque bytes and padding are not scaled.

Repacking rounds FP32 directly to the E4M3FN, FP16 or FP32 storage required by each
original resource region, using nearest/ties-to-even. It adds no scale and never
silently clips overflow. NaN/Inf, out-of-range values, missing/extra tensors,
shape/dtype mismatches, nonzero input lane 15, modified opaque data and source
mismatches are rejected. Quantization never overwrites the original FP32 training artifact.

Both the output DLL and `<output_dll>.report.json` must be new paths. Only same-sized
weight payloads are replaced; PE structure, resource metadata and all other content
are preserved. The tool verifies decoded quantized values per tensor, then extracts
the final DLL again to check its payloads. The report records input/output hashes,
all three controls, and per-tensor counts of trained changes, changes after
mix/strength, changes actually written, updates lost to rounding and quantization
errors. The CLI warns if every requested update rounds back to the original weights.

These checks do not certify native deployment. The tool does not repair the PE
checksum, re-sign the DLL or bypass load checks; a loader may reject a modified
DLL. Reports always retain `native_export_validated=false`. In-game loading and
image quality require separate acceptance checks. Do not use an exported DLL as
the next repacking template. Keep the original DLL, and continue training from
canonical model directories.

The standalone merge script supports the same LoRA multiplier:

```bash
python dlssnr_merge_lora.py --base_model_dir models/canonical_dlssnr --adapter output/lora/run/final/adapter.safetensors --lora_multiplier 0.8 --output_dir output/merged
```

## Commands

Convert a user-supplied OpenDLSS-NR `models/nr` directory into canonical weights:

```bash
python dlssnr_convert_model.py --source_dir ../OpenDLSS-NR/models/nr --output_dir models/canonical_dlssnr --verify_roundtrip
```

Single-frame full training. Multiline commands below use Bash continuation;
in PowerShell, use one line or replace trailing backslashes with backticks:

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-5 \
  --optimizer_args weight_decay=0.0 \
  --output_dir output/dlssnr_full_single --output_name dlssnr_310_8_0_full \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

Temporal full training:

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_temporal.toml \
  --model_dir models/canonical_dlssnr \
  --training_mode temporal --sequence_length 4 --burn_in 2 --tbptt_length 2 \
  --gradient_checkpointing \
  --loss_temporal 0.10 --optimizer_type AdamW --learning_rate 1e-5 \
  --output_dir output/dlssnr_full_temporal --output_name dlssnr_310_8_0_temporal \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

ViT LoRA, selecting rank, alpha and optimizer through CLI arguments and using the
same single-frame dataset TOML:

```bash
python dlssnr_train_network.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --network_dim 16 --network_alpha 16 --network_dropout 0.0 \
  --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-4 \
  --optimizer_args weight_decay=0.0 betas=0.9,0.999 \
  --output_dir output/dlssnr_lora --output_name dlssnr_310_8_0_lora_vit \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

Common arguments:

| Option | Purpose |
| --- | --- |
| `--optimizer_type` / `--optimizer_args` | Uses the shared optimizer factory, for example `AdamW`, `SGD`, `AdamW8bit` or a fully qualified class path. Install any extra dependencies separately. |
| `--learning_rate` | Defaults to `1e-5` for full training and `1e-4` for LoRA. AdamW weight decay defaults to `0.0`. |
| `--network_dim` / `--network_alpha` | ViT LoRA rank defaults to `16`; omitted alpha follows rank. |
| `--network_dropout` / `--network_args` | Dropout defaults to `0`. Use `--network_args profile=multiscale` for multiscale LoRA, optionally with `rank_by_width` / `alpha_by_width` dictionaries. Do not combine this with a single dim/alpha. |
| `--max_train_steps` / `--gradient_accumulation_steps` / `--seed` | Update count, gradient accumulation and random seed. |
| `--max_grad_norm` | Clips gradients at the accumulated update boundary. Default `0` disables clipping. |
| `--gradient_checkpointing` | Recomputes activations to reduce memory, preserving the default numerical profile. Disabled by default. |
| `--numerics_profile train_experimental` / `--mixed_precision` | Explicit experimental mode. CUDA precision accepts `no`, `fp16` or `bf16`; master weights remain FP32. |
| `--fp8_base` / `--fp8_scaled` | LoRA-only frozen-base storage quantization. Scaled requires base, and both require experimental mode. |
| `--sdpa` / `--xformers` / `--flash_attn` / `--attention_scope` | Explicit experimental attention and its scope; restrictions are listed above. |
| `--loss_pre` / `--loss_out` / `--loss_edge` / `--loss_temporal` | Loss weights, defaulting to `1`, `1`, `0.05` and `0`, respectively. |
| `--prior_lr_multiplier` / `--scale_lr_multiplier` / `--temporal_blend_lr_multiplier` | Full-training parameter-group multipliers, default `0.1`. The LoRA entry point rejects these arguments. |
| `--sample_every_n_steps` / `--min_sequence_frames` | Validation interval and minimum temporal evaluation frame count. A nonzero interval requires validation manifests in the dataset TOML. |
| `--no-compare_baseline` | Disables the initial-base comparison, which is enabled by default. |
| `--save_state` / `--save_every_n_steps` / `--resume` | Training-state saves, save interval and resume checkpoint. Without `--save_state`, only weights are saved. |
| `--forward_validation_report` | Optional base-validation evidence, checked only when explicitly supplied. It is not a training prerequisite. |
| `--development_smoke` | Developer tests only: explicitly permits incomplete provenance or random initialization. Normal training does not need it. |

For example, `--optimizer_type SGD --optimizer_args momentum=0.9 weight_decay=0.0`
constructs an actual SGD optimizer rather than changing only a configuration label.
Adafactor requires explicit `--optimizer_args relative_step=False warmup_init=False`.
Learning-rate schedules use the shared factory, supporting constant,
constant_with_warmup, linear, cosine, cosine_with_restarts, cosine_with_min_lr,
polynomial, inverse_sqrt and warmup_stable_decay. Warmup/decay, cycles, power,
timescale and min_lr_ratio follow the shared CLI. The scheduler advances once per
successful global optimizer update, not per accumulation microstep, DDP rank or
overflow retry. Scheduler state is saved and restored with training state;
changing its configuration prevents exact resume. Schedule-free weight switching
and optimizers requiring a closure remain unsupported.

To resume at an optimizer-update boundary, append the resume argument to the
**original complete training command**, retaining its effective parameters and
dataset configuration. For example:

```text
--resume output/dlssnr_full_single/dlssnr_310_8_0_full/state-step000100
```

The corresponding LoRA resume directory is
`output/dlssnr_lora/dlssnr_310_8_0_lora_vit/state-stepNNNNNN`.
Loading weights without optimizer state starts a new run rather than resuming one.

Still-image and closed-loop clip inference. Inference manifests may omit targets;
non-reset clip frames must provide motion and a history mask:

```bash
python dlssnr_generate_image.py --model_dir models/canonical_dlssnr --sample_manifest data/inference_single.jsonl --bucket_width 512 --bucket_height 512 --output_dir output/preview
python dlssnr_generate_video.py --model_dir models/canonical_dlssnr --sequence_manifest data/inference_sequence.jsonl --bucket_width 512 --bucket_height 512 --output_dir output/video
python dlssnr_merge_lora.py --base_model_dir models/canonical_dlssnr --adapter output/dlssnr_lora/dlssnr_310_8_0_lora_vit/final/adapter.safetensors --output_dir output/dlssnr_merged
```

Inference inherits the artifact's runtime policy unless overridden. Legacy models
without a policy use FP32/native defaults. Explicit overrides are recorded in
`inference_metadata.json` in the output directory, or the clip subdirectory for
videos. Returning to baseline requires compatible options together, for example
`--numerics_profile train_surrogate --mixed_precision no --attention_backend native`.
Disable FP8 with `--no-fp8_base` / `--no-fp8_scaled`. Sage global inference requires
explicit `--numerics_profile train_experimental --mixed_precision bf16 --sage_attn --attention_scope global`
and an available local extension.

## Unsupported Features

Full FP8 optimizer training, SageAttention training, CPU activation offload, block
swapping, FSDP/ZeRO/DeepSpeed, multi-node execution and TF32 are unsupported. Unknown
CLI arguments still raise errors. Experimental modes target `float_runtime` only;
`native_roundtrip` cannot be selected as that runtime mode.

History reprojection uses bilinear sampling rather than the original five-tap
Catmull-Rom. History is stored as FP32 rather than truncated toward zero to float16.

Pixel caching and multi-worker prefetching are not implemented, and dataset TOML
does not accept placeholder cache fields. FP32 entry points temporarily disable
TF32 and restore its previous global settings afterward.

## Data Contract

Each JSONL record requires a unique, filename-safe `sample_id`, a `sequence_id`,
`schema = dlssnr_pairs_v1`, `source_encoding = srgb_proxy` and
`controls_encoding = dlssnr_lanes_10_14_v1`. Training and paired evaluation also
require a target and `target_encoding = srgb_proxy`. The first frame must reset;
`frame_index` must be nonnegative and strictly increasing. Training and validation
cannot share sequence IDs.

`sample_id` uniqueness is checked case-insensitively on every platform: `Frame`
and `frame` cannot occur in the same manifest. Names cannot contain path separators,
Windows-invalid characters, control characters, trailing periods/spaces or Windows
reserved names such as `NUL` and `CON`. Errors are raised during manifest indexing,
before partial inference outputs are written. Original IDs are never rewritten automatically.

- RGB accepts 8-bit images or numeric `[3,H,W]` NPY arrays, with proxy values in `[0,1]`. High-bit-depth images are not silently reduced to 8-bit.
- Controls are finite-valued `[5,H,W]` NPY arrays. Each record explicitly declares motion layout with `motion_layout = chw|hwc`; values are current-to-previous pixel displacements.
- Masks accept single-channel PNGs or `[1,H,W]` NPY arrays. History/temporal masks must be binary; optional `loss_mask_path` supports weights in `[0,1]`.
- With nonzero temporal loss, non-reset frames require a temporal mask and `tbptt_length >= 2`. Loss uses adjacent frames within the differentiable segment only, never across the burn-in boundary.
- `--loss_temporal 0` allows temporal masks to be omitted, but non-reset frames still require motion and history masks.

## Saving and Resuming

Each run uses its own `output_dir/output_name/` directory for `final/`, periodic
weights, training state, logs and evaluation reports. `--output_name` must follow
the portable-name rules above and cannot contain a path. A new run refuses an
existing directory with the same name. Continue with explicit `--resume`, or
choose another name/output directory to start over.

Existing artifacts are not moved automatically. Resolved configuration still uses
schema version 2. Training state now uses `dlssnr_train_state_v3`, including scaler,
per-rank state and runtime identity. Old v1/v2 states and implementation fingerprints
cannot resume exactly across versions. Older full weights can serve as a new run's
base; older LoRA can be merged into a canonical base first, but this does not restore
the original optimizer or adapter parameterization. Do not edit checkpoint identity
markers manually.

Full-training `final/` contains FP32 weights, canonical configuration, the source
manifest, original opaque records, numerics/preprocessing information and training
metadata. Runtime policy is bound to both required model config and safetensors
headers. Missing or inconsistent bindings are rejected; reproducing the policy does
not depend on an optional training sidecar. LoRA v2 stores the adapter,
rank/alpha/targets, complete base identity and required runtime/quantization metadata.
Conforming v1 FP32 adapters remain readable. FP8 adapter merging reconstructs and
verifies the quantized base, then exports ordinary FP32 base-plus-delta weights marked
as materialized. Default loading does not quantize them again. Merging preserves
canonical provenance and opaque data.

LoRA forward computation uses `W + alpha/r * B@A` on the weight side, following
the same projection computation as merged weights. Real-weight tests found that
FP32 error from two separate GEMMs followed by addition is amplified in ViT; small
linear-layer tests alone cannot establish whole-network merge equivalence. Dropout
is added as a rank-space correction and disabled for evaluation. This choice favors
consistency at the cost of temporary matrices and backward work. Artifacts record
`canonical_weight_plus_delta_v1`.

`--save_state` creates `state-stepNNNNNN/` at periodic and final update boundaries.
State includes the optimizer, actual global consumed-sample count, each rank's
CPU/current-CUDA-device/Python/NumPy RNG, FP16 scaler, resolved configuration and
data/batch-plan/base/implementation fingerprints. The checksum manifest is published
only after all state is complete. Successful update count, gradient accumulation
and world size locate the global resume batch, including short final batches and
epoch transitions. Without this option, `stepNNNNNN/` weights are still saved, but
those directories cannot provide exact resume.

Changing only dataset TOML comments or CLI argument order does not prevent resume.
Changing learning rate, optimizer, rank, bucketing, precision, attention, FP8 policy,
world size, referenced data, base or training implementation rejects exact resume.
Explicit report paths and contents are also part of resume identity; unrequested
reports in the model directory are not. Nondeterministic CUDA backward operators
still require tolerance-based acceptance. Bitwise equivalence across all hardware
is not promised.

Legacy adapters missing complete base identity, rank/alpha or forward-mode metadata
are also rejected. Keep the old files and regenerate validated artifacts.

## Validation History and Regression Tests

On 2026-09-23, the local full suite passed 159 tests with no skips, and Ruff passed.
Tests covered original 310.8.0 weights on an RTX 4090: single FP32 full/LoRA updates,
gradients for every ViT LoRA target, and raw-head comparisons after merging. The CUDA
smoke tests used synthetic paired `48 x 48` inputs; they do not establish production-
resolution performance, real-data training quality or native parity.

If the project package is not installed in a development environment, set
`PYTHONPATH` at the repository root before testing. Normal use should still run in
the project's dependency environment. Real-weight CUDA tests require local source
weights, defaulting to the sibling `../OpenDLSS-NR/models/nr` directory, or a path
set through `DLSSNR_SOURCE_DIR`. These external-weight tests skip when weights are
unavailable; a skip is not real-weight acceptance.

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:DLSSNR_RUN_CUDA_SMOKE = "1"
python -m pytest -q --tb=short
```

DLL import/export integration tests are separately opt-in. Set `DLSSNR_DLL_PATH`
to a user-supplied original DLL containing the audited resource:

```powershell
$env:DLSSNR_DLL_PATH = "D:/path/to/original/nvngx_dlssnr.dll"
python -m pytest tests/test_dlssnr_dll_io.py -q
```

These tests cover original-byte reconstruction, changed-weight re-extraction,
resetting to the original weights with `--mix 0`, and LoRA merging with its own
multiplier. They write only temporary artifacts and never execute the DLL.

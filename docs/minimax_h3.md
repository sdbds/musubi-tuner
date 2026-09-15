# MiniMax-H3

This is the user guide: what to download, which training recipe fits your goal, and the commands for caching, training, sampling, and generation. Two companion documents cover the rest:

- `docs/minimax_h3_1f.md` — one-frame (image) generation and training: time indices, editing/inbetween datasets, reference-conditioned images.
- `docs/minimax_h3_advanced.md` — how the pieces work: timestep and loss internals, the guidance loss and teacher matching in depth, cache and quantization internals, generation internals, implementation provenance.

## Overview

Musubi Tuner supports MiniMax-H3 text-to-video-with-audio (T2VA), first/last-frame-to-video-with-audio (FL2VA), and reference-to-video-with-audio (Ref2VA) LoRA training and standalone generation, plus an experimental one-frame (image) mode for both.

The implementation follows the released MiniMax-H3 packing, Qwen3-VL conditioning, dual video/audio flow schedules, and two VAE layouts. It supports the published full and pruned BF16 transformers, the full and pruned ConvRot INT8 transformers, and the ConvRot INT8 and NVFP4+AWQ Qwen3-VL text encoders.

Read and accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or using the weights.

## Model Files

Download the following files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| Component | Supported file |
| --- | --- |
| FL2VA and T2VA transformer | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` |
| FL2VA and T2VA pruned transformer | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` |
| FL2VA and T2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` |
| FL2VA and T2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Ref2VA transformer | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` |
| Ref2VA pruned transformer | `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` |
| Ref2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors` |
| Ref2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| Qwen3-VL-32B ConvRot INT8 text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| Qwen3-VL-32B NVFP4+AWQ text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` |

Which base for which task: T2VA and FL2VA (and every one-frame image recipe except reference-conditioned images) use the FL2VA transformer; Ref2VA uses the Ref2VA transformer. The pruned, ConvRot INT8, and NVFP4+AWQ files are drop-in replacements for their BF16 counterparts and are detected automatically from their tensor structure — pass them to `--dit` / `--text_encoder` and nothing else changes. What each one saves is summarized in [Memory and speed options](#memory-and-speed-options). FP8 files and NVFP4 transformers are rejected.

The Qwen3-VL processor and config are downloaded by Transformers from the official [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) repository (`processor` and `text_encoder` subfolders, a few config and tokenizer files only, no weights). The upstream `Qwen/Qwen3-VL-32B-Instruct` files are not interchangeable: the H3 tokenizer adds `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, and `<|caption_end|>` as special tokens, and the released prompt format writes dialogue and lyrics as `<d>[Language] ...</d>`.

## Geometry And Media Contract

- Target video is 24 fps. Source videos are normalized to 24 fps from their frame timestamps, so `source_fps` is not needed and is ignored if set.
- Width and height must be positive multiples of 32.
- Frame count must be `17*n+5`. The released duration range is 5 to 15 seconds: at 24 fps, frame counts from 124 through 345 in steps of 17. `--allow_experimental_duration` bypasses only this duration check.
- Target audio is optional. When present it is decoded as stereo 32000 Hz audio; when absent, the cache stores a silence placeholder that is never used as a supervision target (see [Audio policy](#audio-policy)).
- Ref2VA references are ordered per record: from the JSONL `references` list for video datasets (the shared control-video fields are not used), and for image datasets also from control images (`control_directory` / `control_path`, one image reference per control). At most 12 references per record, of which at most 9 images, 3 videos, and 3 audio-bearing; at least one image or video; reference videos 2 to 15 seconds.
- Expanded Qwen conditioning is limited to 32768 rows. A BF16 text cache at the limit is approximately 320 MiB for one sample.

## Choosing A Training Recipe

The released H3 checkpoints are CFG-distilled: they predict in an amplified "guided" space, and a LoRA trained on the plain flow-matching target pulls the model out of it. Video training then washes out and loses prompt adherence as it progresses; image training breaks structurally within about 50 steps. Plain flow training is therefore not offered as a recipe. Pick one of the three loss methods in Table A, then find your goal in Table B for the matching dataset shape, base, and flags.

### Table A: loss methods

| Method | What it does | Extra flags | Cost | Constraints |
| --- | --- | --- | --- | --- |
| Training adapter (de-distillation LoRA) | Merges a third-party (or Musubi-provided) adapter into the base at load time and trains with the plain flow loss on the de-distilled model. At inference the trained LoRA runs on the plain base, without the adapter. | `--base_weights adapter.safetensors` | none | BF16 source (with or without `--convrot_int8`); pre-quantized INT8 files cannot be merged into. Combining with the guidance loss or teacher matching is allowed but warned (see below). Training-time samples show the de-distilled model, not the plain base + LoRA. |
| Guidance loss | Re-anchors the flow target in the guided space using the model's own no-grad unconditional prediction. | text cache: `--uncond_output uncond.safetensors`; training: `--h3_guidance_loss_scale 4.0 --h3_guidance_loss_sigma_min 0.15 --h3_guidance_loss_uncond_cache uncond.safetensors` | +1 no-grad forward on ~85% of steps | any base, including INT8; not with teacher matching |
| Teacher matching | Trains a text-only student against the frozen base's prediction under privileged conditions (the clip's endpoints, the clip itself, or other pictures of the subject). | `--h3_teacher_matching --h3_teacher_conditions first,last` / `ref` / `subject_ref` plus the per-teacher settings in [Training](#training) | +1 no-grad forward per step | student is always `--task t2va`; image targets: `subject_ref` only; not with the guidance loss |

Choosing between them: the adapter is the cheapest (no extra forward) and the least to configure; the guidance loss costs about 1.5x per step but needs no third-party file and works on a pre-quantized INT8 base; teacher matching is the recipe for identity training when the appearance is kept out of the captions (the teacher sees it, the student has to learn it). `--base_weights` also has ordinary uses (a style LoRA under a character LoRA), so combining it with the other two only logs a warning: the adapter authors advise against the guidance loss on top of it, and under teacher matching the merged base becomes the teacher.

### Table B: recipes by goal

| Goal | Dataset shape | Base | Latent cache | Text cache | Training | Loss method |
| --- | --- | --- | --- | --- | --- | --- |
| Video: style, motion, general concept | `video_directory` or video JSONL | FL2VA | `--task t2va` | `--task t2va` (+`--uncond_output` for GL) | `--task t2va` | Adapter or guidance loss |
| Video: character identity, appearance kept out of captions | same | FL2VA | `--task fl2va` (endpoint teacher) or `--task t2va` (reference teacher) | `--task t2va --teacher_conditions first,last` or `ref` | `--task t2va` + [endpoint or reference teacher](#teacher-matching) | Teacher matching (`ref`: identity + voice; `first,last`: identity, base audio kept). Alternative: adapter or GL with a trigger word |
| Video: FL2VA (first/last-frame conditioned) | `video_directory` | FL2VA | `--task fl2va` | `--task fl2va` | `--task fl2va` | Adapter or GL |
| Video: Ref2VA (reference conditioned) | video JSONL with `references` | Ref2VA | `--task ref2va` | `--task ref2va` | `--task ref2va` | Adapter or GL |
| Image: plain image LoRA | `image_directory` or image JSONL | FL2VA | `--task t2va --one_frame` | `--task t2va --one_frame` | `--task t2va --one_frame --video_only` | Adapter or GL |
| Image: character identity, text-only at inference | image JSONL `references` or `control_directory` without `fp_1f_clean_indices` | FL2VA | `--task ref2va --one_frame` | `--task t2va --one_frame --teacher_conditions subject_ref` | `--task t2va --one_frame --video_only` + [subject-reference teacher](#teacher-matching) | Teacher matching (`subject_ref`). Alternative: plain image row with a trigger word |
| Image: editing / inbetween | image + timed controls (`fp_1f_clean_indices`, `fp_1f_target_index`) | FL2VA | `--task fl2va --one_frame` | `--task fl2va --one_frame` | `--task fl2va --one_frame --video_only` | Adapter or GL |
| Image: reference-conditioned at inference | image + untimed references | Ref2VA (FL2VA also works) | `--task ref2va --one_frame` | `--task ref2va --one_frame` | `--task ref2va --one_frame --video_only` | Adapter or GL |

Use the same `--task` for latent caching, text caching, and training unless the row says otherwise (the teacher-matching rows deliberately cache with a richer task than the student trains with). The image rows are described in detail in `docs/minimax_h3_1f.md`, including how the time indices and the timed-versus-untimed control distinction work. Mixed image+video training in one run is expected to work but is untested.

### Training adapters

Third-party de-distillation adapters that have been used with `--base_weights` on Musubi Tuner (all three load and train; output quality has not been evaluated here):

| Adapter | Target base | Rank | Notes |
| --- | --- | --- | --- |
| [circlestone-labs/MiniMax-H3-Image-Training-Adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter) | FL2VA, image-first (video and mixed "seem to work" per its README) | 64 | plain flow on 10k images / 10k steps; the README advises against the guidance loss on top of it |
| [ostris/minimax_h3_training_adapter](https://huggingface.co/ostris/minimax_h3_training_adapter) v2 | FL2VA | 32 | ai-toolkit 0.13.4 |
| [ostris/minimax_h3_training_adapter](https://huggingface.co/ostris/minimax_h3_training_adapter) `minimax_h3_ref2va_training_adapter_v1` | Ref2VA | 16 | ai-toolkit 0.12.23 |

All three are in the Diffusers key format (`diffusion_model.blocks.N....lora_A/lora_B.weight`, no alpha, so alpha = rank), which `--base_weights` and every generation `--lora_weight` route accept alongside Musubi's own format. A LoRA loaded from weights is applied to every module the file contains, including token refiner modules that Musubi's own training default leaves out.

## Dataset Configuration

Dataset configuration uses the common TOML schema (`docs/dataset_config.md`). H3-specific rules: `batch_size` must be 1 in every H3 dataset (use gradient accumulation for a larger effective batch), and image and video datasets may share one TOML but must not share a `cache_directory`.

### Video directory (T2VA, FL2VA)

```toml
[general]
resolution = [768, 1344]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "/data/h3/videos"
cache_directory = "/data/h3/cache"
caption_extension = ".txt"
target_frames = [124]
frame_extraction = "head"
```

For a directory item such as `clip.mp4`, put the caption in `clip.txt`. FL2VA derives its first and last conditions from each selected target crop. Target audio is resolved in this order: exactly one same-stem audio sidecar such as `clip.wav`, then the video's embedded audio stream, then the silence placeholder.

### Video JSONL with references (Ref2VA)

```toml
[[datasets]]
video_jsonl_file = "/data/h3/ref2va.jsonl"
cache_directory = "/data/h3/cache-ref2va"
target_frames = [124]
frame_extraction = "head"
```

Each line holds the target plus its ordered references; relative paths resolve from the JSONL directory. A JSONL `audio_path` on the target takes precedence over sidecar and embedded audio.

```json
{"video_path":"targets/clip.mp4","audio_path":"targets/clip.wav","caption":"A singer performs under stage lights.","references":[{"type":"image","path":"refs/style.png"},{"type":"video","path":"refs/motion.mp4","audio_path":"refs/motion.wav"},{"type":"audio","path":"refs/voice.wav"}]}
```

A `video` reference uses its explicit `audio_path`, else its embedded track; `"audio_path": null` makes it a visual-only reference (motion or composition) even when the file has audio. A reference video without an audio track is likewise visual-only. `audio_path` is valid only on `video` references. The reference limits are listed under [Geometry And Media Contract](#geometry-and-media-contract). The same JSONL (with `references` on image targets) also feeds the subject-reference teacher for video identity training.

### Image directory or image JSONL (plain image LoRA)

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/data/h3/images"
cache_directory = "/data/h3/cache-images"
caption_extension = ".txt"
```

`image_jsonl_file` works as usual. Buckets snap to the 32-pixel grid. Since every image item carries silent audio rows, state the absence of sound in the caption (a `sound:`-style field describing a silent still) so the text stays consistent with what the model sees.

### Image with timed controls (editing / inbetween)

```toml
[[datasets]]
image_directory = "/data/h3/edit/targets"
control_directory = "/data/h3/edit/sources"
cache_directory = "/data/h3/cache-edit"
caption_extension = ".txt"
fp_1f_clean_indices = [0]     # control image positions (24 fps pixel-frame indices)
fp_1f_target_index = 24       # target position, required when controls are present
```

`fp_1f_clean_indices` is what makes the controls timed FL2VA anchors. Index choice matters a lot (a control at the target's own index trains against verbatim copying); see `docs/minimax_h3_1f.md`.

### Image with untimed references (reference-conditioned images, subject-reference teacher)

Either per-record `references` in `image_jsonl_file` (same schema as the video JSONL, images and videos), or `control_directory` / `control_path` **without** `fp_1f_clean_indices`, in which case each control image becomes one image reference in index order:

```json
{"image_path": "/data/h3/char/targets/pose_01.png", "caption": "...", "references": [{"type": "image", "path": "refs/front.png"}]}
```

For the subject-reference teacher, the references should be *other* pictures of the same subject, and `caption` is the student's plain caption (a trigger word, appearance left out); an optional `teacher_caption` overrides the teacher's automatically wrapped caption.

## Caching

Run latent caching and text-encoder caching once per dataset with the `--task` (and `--one_frame`) columns of Table B:

```bash
python minimax_h3_cache_latents.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --cache_seed 42 \
  --skip_existing

python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --text_cache_dtype bf16 \
  --skip_existing
```

- `--audio_vae` is always required: H3 always includes audio rows, even for silent items.
- `--skip_existing` rebuilds any cache whose stored metadata (task, cache seed, crop, format version, media and VAE fingerprints) no longer matches, so it is safe to leave on. Fingerprints are size + mtime, so a re-copied file triggers a one-time re-cache.
- The latent cache script prints the supervised-audio fraction at the end; a warning means no item had real audio.
- Text caching accepts the ConvRot INT8 and NVFP4+AWQ text encoders as well. On VRAM-limited GPUs add `--text_encoder_blocks_to_swap 50`, and `--text_encoder_attn_mode flash_attention_2` for long Ref2VA presentations.

Per-recipe additions to the text-caching command:

- **Guidance loss:** `--uncond_output /data/h3/uncond.safetensors` writes the tiny unconditional probe embedding (about 10 KB, one extra forward). `--uncond_text` overrides the probe text (default: a single space, which was selected as the true distillation uncond; see the advanced document).
- **Teacher matching:** `--teacher_conditions first,last`, `ref`, or `subject_ref` (always with `--task t2va`) stores the teacher's presentation next to the plain caption rows. The caption is shared; the teacher rows add the pictures or the reference declaration. The trainer hard-fails when the cache's teacher kind and `--h3_teacher_conditions` disagree, so re-cache text when switching teachers.

## Training

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 minimax_h3_train_network.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --network_dim 16 \
  --network_alpha 16 \
  --sdpa \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --blocks_to_swap 48 \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --output_dir /data/h3/output \
  --output_name h3-lora \
  <loss method flags>
```

`--network_module` defaults to `networks.lora_minimax_h3`, whose default targets are `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2` in the 50 main DiT blocks. `--timestep_sampling uniform`, `--weighting_scheme none`, and `--discrete_flow_shift 1.0` are the H3 defaults and the only accepted values: H3 draws one base time per item and derives the video and audio sigmas from it with its own two shifts (12 and 3). `--min_timestep` / `--max_timestep` clip that base time (1000 = pure noise) before the shifts; the conversion to per-stream sigmas is tabulated in the advanced document.

Add exactly one of the following.

### Training adapter

```text
--base_weights /models/minimax_h3_training_adapter.safetensors
```

Works with a BF16 `--dit`, with or without `--convrot_int8` (the adapter is merged into the BF16 weights during the streaming load and quantized with them). Pre-quantized INT8 files are rejected. Use the trained LoRA on the plain base at inference; do not judge it by the training-time samples, which show the de-distilled model.

### Guidance loss

```text
--h3_guidance_loss_scale 4.0 \
--h3_guidance_loss_sigma_min 0.15 \
--h3_guidance_loss_uncond_cache /data/h3/uncond.safetensors
```

Scale 3-4 (4 is more reliable for longer runs). `--h3_guidance_loss_sigma_min 0.15` skips the extra forward on the lowest-noise ~15% of steps, where the correction is mostly amplified noise; `0` applies it always. `--h3_guidance_loss_scale_audio` sets a separate audio scale. Each step logs `guidance/applied` and the gap magnitudes `guidance/video_gap_rms` / `guidance/audio_gap_rms`.

### Teacher matching

The student is always `--task t2va`. Three teachers, each with its validated starting recipe:

**Endpoint teacher** (`first,last`; latent cache `--task fl2va`; identity from video, base audio behavior preserved):

```text
--h3_teacher_matching --h3_teacher_conditions first,last \
--h3_teacher_condition_sigma_max 0.75 --h3_teacher_loss_dc_weight 0.3 --h3_timestep_focus_prob 0.5
```

**Reference teacher** (`ref`; latent cache `--task t2va` or `fl2va`; identity and voice from video — the teacher copies each clip's actual audio, so the audio must be worth learning, otherwise lower `--audio_loss_weight` or pass `--video_only`):

```text
--h3_teacher_matching --h3_teacher_conditions ref \
--h3_teacher_condition_sigma_max 0.75 --h3_teacher_loss_dc_weight 0.3 --h3_timestep_focus_prob 0.5
```

**Subject-reference teacher** (`subject_ref`; latent cache `--task ref2va`; identity from other pictures of the subject; the only teacher for image targets, also usable for video):

```text
--h3_teacher_matching --h3_teacher_conditions subject_ref \
--h3_teacher_condition_sigma_min 0.15 --h3_teacher_loss_mag_weight 0.5 --h3_teacher_loss_dc_weight 0.3 \
--learning_rate 3e-4 --lr_warmup_steps 50 --max_train_steps 500
```

What the knobs mean, briefly: `--h3_teacher_condition_sigma_max` turns the highest-noise band into a base-preservation anchor so composition decisions are not overwritten (`0.75` for the endpoint and reference teachers; the subject-reference teacher keeps the default `1.0` because identity is decided at the top of the range, and the trainer warns when the value does not match the teacher's recipe); `--h3_teacher_condition_sigma_min` is the mirror gate at the low end; `--h3_teacher_loss_dc_weight` below 1 stops the dataset's palette from being learned as a style shift (keep `1.0` for style LoRAs); `--h3_teacher_loss_mag_weight` below 1 prioritizes direction over magnitude (a candidate for the reference teacher too, where the remaining distillation wedge is mostly a magnitude effect); `--h3_timestep_focus_prob P` lands a fraction P of the draws in `[--h3_timestep_focus_min, --h3_timestep_focus_max)` (default 0.4-0.8 in base units, where content is decided), which roughly doubles the band's convergence speed at 0.5; `--h3_teacher_preservation_weight` (default 1.0) strengthens the anchor for long runs. The loss does not converge to zero (the teacher knows things the text cannot), the teaching-band residual can plateau after a few hundred steps, and the strongest checkpoints tend to sit at or just after the plateau — save and evaluate intermediate checkpoints. The mechanism, the sigma-binned logs (`teacher/*`), how to read them, and the metadata keys are in the advanced document.

### Audio policy

Every sample contributes the video loss. A sample cached with real audio additionally contributes `--audio_loss_weight` (default 1.0) times the audio loss; items without real audio never contribute audio loss. `--video_only` disables audio supervision entirely (the model still attends to the audio latents as context). Because H3 is single-stream, a video-only LoRA modifies the weights the audio path uses too: treat audio from a fully video-only LoRA as unconstrained output. Image datasets should always pass `--video_only` (their audio rows are silence placeholders and would contribute nothing anyway).

## Memory And Speed Options

All options combine with each other and with every recipe. Sizes are transformer weight sizes unless noted.

| Option | Effect | Notes |
| --- | --- | --- |
| Pruned transformer (`*_pruned_*` files, or `--prune_adaln` on a full BF16 file) | ~66 → ~40 GB BF16, ~34 → ~21 GB INT8; each swapped block ~40% smaller, block-swap steps faster by the same fraction | detected automatically; `--prune_adaln` prunes at load time with slightly better reconstruction than the published files and combines with `--convrot_int8` |
| ConvRot INT8 transformer (`*_int8_convrot` files, or `--convrot_int8` on a BF16 file) | ~66 → ~34 GB; block-swap step time roughly halved (transfer-bound) | bit-identical to the published INT8 files; `--base_weights` needs the BF16 file + `--convrot_int8`; requires triton for the fused kernels (`triton-windows` on Windows), otherwise a slower dequantizing fallback with the same memory saving |
| `--blocks_to_swap N` (up to 48 of 50) | streams N blocks from CPU | `--gradient_checkpointing` recommended |
| `--block_swap_h2d_only` | faster block swap for frozen-base LoRA training (no device-to-host copies) | requires `--gradient_checkpointing`; see `docs/block_swap.md` |
| ConvRot INT8 text encoder | text encoder ~48 → ~25 GB | wherever `--text_encoder` is accepted |
| NVFP4+AWQ text encoder | text encoder ~48 → ~15 GB | inference-only artifact (the text encoder is always frozen); `--nvfp4_scaled_mm` opts into faster W4A4 matmuls on Blackwell GPUs with PyTorch 2.10+ |
| `--text_encoder_blocks_to_swap N` (up to 50) | streams N of the 50 Qwen3-VL layers from CPU; at 50 only embedding, vision tower, norms, and two one-layer buffers stay resident | requires CUDA; combines with the quantized encoders; add `--text_encoder_attn_mode flash_attention_2` for long Ref2VA presentations, where SDPA can fall back to an O(L^2) FP32 kernel |
| `--compile` (training and generation) | torch.compile on the 50 DiT blocks | with block swap or an INT8 base the Linears stay eager; each new latent shape recompiles (`--compile_dynamic true` for varying shapes) |

The reference teacher's forward carries the full reference video and audio tokens, so its teacher step is slower and needs more memory than the endpoint teacher's.

## Training-Time Samples

Add the sampling assets and the normal sampling schedule flags to the training command:

```text
--sample_prompts /data/h3/sample_prompts.json \
--sample_every_n_epochs 1 \
--video_vae /models/minimax_h3_video_vae_fp16.safetensors \
--audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
--text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors
```

Samples are written as muxed MP4s under `OUTPUT_DIR/sample` (one-frame samples as PNG). The text encoder is loaded on the accelerator before the transformer to prepare every prompt once, so `--sample_prompts` needs room for it at that point: about 50 GB for the BF16 artifact, ~25 GB INT8, ~15 GB NVFP4; `--text_encoder_blocks_to_swap 50` removes most of that.

All entries use the training `--task`. A `.txt` prompt file holds one prompt per line with the same line options as generation ([Batch and interactive modes](#batch-and-interactive-modes)); lines starting with `#` are skipped:

```text
# T2VA
A singer performs under stage lights. --w 768 --h 1344 --f 124 --s 30 --d 42
# FL2VA: first and last frame (--i / --ei), or an ordered --ci list for one-frame samples
Official-format FL2VA caption... --w 768 --h 1344 --f 124 --s 30 --d 42 --i first.png --ei last.png
# Ref2VA: inline references (--ref, repeatable) or a record of a JSONL file (--rj)
A cat sings. --w 768 --h 1344 --f 124 --s 30 --d 42 --ref refs/cat.png --ref refs/dance.mp4;audio=refs/song.wav
```

Relative `--ref` and `--rj` paths resolve from the prompt file's directory. A `.json` prompt file takes the same requests as objects (`prompt`, `width`, `height`, `frame_count`, `sample_steps`, `seed`; `first_frame` / `last_frame`, `reference_jsonl` + optional `reference_index`, or a `ref` list). `--n`/`--l`/`--g` are rejected (no negative prompt or CFG). Geometry must be 32-pixel aligned; frame counts are rounded down to `17*n+5`, and `--h3_allow_experimental_sample_duration` permits samples shorter than 5 seconds.

Two caveats: samples under a merged `--base_weights` adapter show the de-distilled model, not the plain base + LoRA. And a LoRA trained toward a small equilibrium (teacher matching in particular) can look weaker in generation than in samples when it is merged into the BF16 base, because the merge rounds deltas below a BF16 mantissa step away; pass `--lora_runtime_attach` to generation to reproduce the training-time forward.

## Generation

T2VA with the FL2VA base:

```bash
python minimax_h3_generate_video.py \
  --task t2va \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --prompt "A singer performs under stage lights." \
  --video_size 1344 768 \
  --video_length 124 \
  --infer_steps 30 \
  --seed 42 \
  --blocks_to_swap 48 \
  --save_path output.mp4
```

`--video_size HEIGHT WIDTH` (multiples of 32), `--video_length` (pixel frames, `17*n+5`; `1` selects the one-frame image mode of `docs/minimax_h3_1f.md`), `--infer_steps N` (N model evaluations; the official 50-step default corresponds to `--infer_steps 49`). `--seed` is optional and logged when drawn. `--save_path` takes a file name or a directory (auto-named `<timestamp>_<seed>`); an existing file is never overwritten. `--output_type` can save the latents instead of or next to the video, or the frames as PNGs plus `audio.wav`.

Task inputs:

- **FL2VA:** `--task fl2va --first_frame first.png --last_frame last.png` with the FL2VA base. Either picture alone is also valid (I2VA / L2VA; use the matching official instruction line in the prompt). Condition images are scaled to cover the canvas and center-cropped, exactly as training fits controls to the bucket.
- **Ref2VA:** `--task ref2va` with the Ref2VA base and either `--reference_jsonl file.jsonl --reference_index 0` (the training JSONL schema; the target media only identifies the record) or inline references: `--ref refs/cat.png --ref "refs/dance.mp4;audio=refs/song.wav" --ref refs/bgm.mp3`. `--ref PATH[;type=image|video|audio][;audio=AUDIO_PATH]` is repeatable in reference order; the type is inferred from the extension when omitted. `--prompt` supplies (or overrides) the caption.
- **Text cache instead of the text encoder:** T2VA and Ref2VA accept `--text_cache` (a dataset text cache whose fingerprint matches the prompt and media); FL2VA does not.

Add a trained LoRA with:

```text
--lora_weight /data/h3/output/h3-lora.safetensors --lora_multiplier 1.0
```

Every route accepts Musubi's format and the Diffusers format written by ai-toolkit and diffusion-pipe. With a BF16 base the LoRA is merged once after loading (with `--convrot_int8`, merged before quantization); with a pre-quantized INT8 base it is attached as a runtime branch, so LoRA generation does not need the BF16 file. `--lora_runtime_attach` forces the runtime branch on any base (see the caveat under [Training-Time Samples](#training-time-samples)).

Model loading dominates single-shot latency (and `--convrot_int8` requantizes at every start), so repeated generation should use the batch or interactive mode.

### Batch and interactive modes

Both read prompt lines in the shared sample-prompt vocabulary; unspecified options inherit the command line, and a line starting with `--` re-runs the command-line prompt with new options:

```text
A singer performs under stage lights. --w 768 --h 1344 --f 124 --d 42 --s 30
```

| Line option | Maps to |
| --- | --- |
| `--w`, `--h` | `--video_size` (`--w` is the width, `--h` the height) |
| `--f` | `--video_length` (`--f 1` selects one-frame mode) |
| `--d` | `--seed` |
| `--s` | `--infer_steps` |
| `--fs`, `--fsa` | `--h3_shift_video`, `--h3_shift_audio` |
| `--ofps`, `--skb` | `--output_fps`, `--stretch_keep_bands` (temporal stretch, see the advanced document) |
| `--i`, `--ei` | `--first_frame`, `--last_frame` (end image) |
| `--ci` | `--condition_image` (one-frame FL2VA; repeatable, ordered; replaces the session-level list) |
| `--ref` | `--ref` (repeatable; replaces the session-level list) |
| `--of` | `--one_frame_inference` |
| `--o` | output filename inside the output directory |

`--from_file prompts.txt` runs every line in four phases (condition encoding, text encoding, sampling, decoding), loading each model family once; peak VRAM matches single-shot generation, and each sampled latent is saved before decoding so a crash never loses finished work. `--interactive` keeps the text encoder and transformer resident for a console session; on 24 GB and below combine a quantized transformer, a generous `--blocks_to_swap`, and `--text_encoder_blocks_to_swap 50`, and budget host RAM for both artifacts. `--latent_path FILE...` decodes saved latents with only the VAEs loaded. Details of the phases, naming, and text-conditioning cache are in the advanced document.

## Limitations

- Released BF16 and ConvRot INT8 (each full or pruned) FL2VA/Ref2VA transformer bases only.
- BF16, ConvRot INT8, or NVFP4+AWQ Qwen3-VL text encoder only.
- No FP8 artifact loading, and no NVFP4 transformer loading.
- No CFG or negative prompt.
- Video datasets take Ref2VA references from JSONL only (no reference-directory convention); control images as references are an image-dataset feature.
- Dataset `batch_size` is fixed to 1; use gradient accumulation for larger effective batches.
- No padded multi-sample packed layouts.
- Plain flow-matching training without one of the three loss methods is not a supported recipe (it runs, but de-distills the model).

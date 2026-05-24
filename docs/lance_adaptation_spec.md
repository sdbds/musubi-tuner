# Lance Adaptation Spec

Status: draft
Date: 2026-05-19
Branch: `codex/lance-adaptation-spec`

This spec defines the first Musubi Tuner adaptation pass for
[bytedance/Lance](https://github.com/bytedance/Lance). It is intentionally
implementation-oriented: the goal is to identify the smallest correct training
surface before adding code.

## Source Facts

- Lance is a 3B native unified multimodal model for image and video
  understanding, generation, and editing.
- The public inference tasks are `t2i`, `t2v`, `image_edit`, `video_edit`,
  `x2t_image`, and `x2t_video`.
- The public model package is hosted at
  [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)
  and currently exposes `Lance_3B`, `Lance_3B_Video`,
  `Qwen2.5-VL-ViT`, and `Wan2.2_VAE.pth` assets.
- Official inference requires Python 3.10+, CUDA 12.4+, and at least 40 GB
  VRAM.
- The official code wraps a modified Qwen2 causal language model, optional
  Qwen2.5-VL ViT tokens for understanding, and Wan2.2 VAE latents for visual
  generation.
- The Lance Wan2.2 VAE path is not interchangeable with Musubi's existing Wan
  VAE wrapper: Lance's public VAE contract uses 48 latent channels, while the
  existing Wan training path is built around the Wan 2.x 16-channel latent
  layout.
- Official generation uses a flow-matching objective over VAE latent tokens.
  In code, clean latents and sampled noise are mixed by shifted timesteps, and
  the MSE target is `noise - clean_latent`.

## Problem

Musubi Tuner already has the right outer shape for model families:

- cache latents;
- cache text encoder outputs;
- train LoRA or full finetune through a model-specific `NetworkTrainer`;
- sample during training;
- save metadata and architecture identifiers.

Lance does not fit the common DiT-only assumption. It is a packed, interleaved
sequence model where text, ViT tokens, clean VAE latents, and noisy VAE latents
share one causal attention stream. The adaptation must preserve that sequence
contract instead of pretending Lance is just another diffusion transformer.

## Non-Goals

- Do not implement full pretraining from scratch.
- Do not support understanding-only CE finetuning in the first pass.
- Do not implement every official task before the first trainable loop works.
- Do not require users to restructure the official Hugging Face checkpoint
  tree.
- Do not merge Lance code into existing Qwen-Image, Wan, or Z-Image modules.

## First Supported Surface

The first usable milestone is generation LoRA training with sample generation:

| Task | Phase 1 | Notes |
| --- | --- | --- |
| `t2i` | yes | Treat as one-frame VAE latent generation. |
| `t2v` | yes | Primary validation target, because Lance video weights are public. |
| `image_edit` | limited | Use source visual context plus target latent loss after T2I works. |
| `video_edit` | limited | Use source visual context plus target latent loss after T2V works. |
| `x2t_image` | no | Requires CE loss path and answer formatting. |
| `x2t_video` | no | Requires CE loss path and video-understanding evaluation. |

## Linus Pass: Invariants

1. Lance must stay a Lance model.
   The implementation must carry over the official packed-sequence contract:
   `sample_lens`, `split_lens`, `attn_modes`, text indexes, ViT indexes, VAE
   indexes, latent position ids, and loss indexes must all refer to the same
   packed sequence.

2. Generation loss must be isolated first.
   The first trainer computes only flow MSE on target VAE tokens. CE loss for
   understanding can be added after generation loss and sampling are correct.

3. Cache formats must be explicit.
   Latent caches need architecture metadata for Lance, VAE downsample
   `(4, 16, 16)`, `z_channels=48`, `latent_patch_size`, and whether the cached
   item is target or conditioning context.

4. Official checkpoint layout is the user contract.
   `--model_path` should point at a Lance checkpoint directory. The loader is
   responsible for finding `llm_config.json`, tokenizer files, model shards,
   optional `Qwen2.5-VL-ViT`, and `Wan2.2_VAE.pth`.

5. Lance VAE must be explicit.
   Do not reuse the existing Musubi Wan VAE implementation unless its latent
   contract is proven to match Lance's 48-channel Wan2.2 VAE.

6. LoRA targets must be conservative.
   Start with the generation expert path and bridge layers. Do not LoRA the VAE,
   tokenizer embeddings, or ViT in the first pass.

7. Sample generation must use the same packer as training.
   A separate inference-only prompt packer will drift. The validation path must
   reuse the same sequence builder with loss indexes disabled.

## Proposed Files

Add these files:

- `src/musubi_tuner/lance/__init__.py`
- `src/musubi_tuner/lance/lance_vae.py`
- `src/musubi_tuner/lance/lance_model.py`
- `src/musubi_tuner/lance/lance_utils.py`
- `src/musubi_tuner/lance/lance_dataset.py`
- `src/musubi_tuner/lance_cache_latents.py`
- `src/musubi_tuner/lance_cache_text_encoder_outputs.py`
- `src/musubi_tuner/lance_train_network.py`
- `src/musubi_tuner/lance_generate.py`
- `src/musubi_tuner/networks/lora_lance.py`
- `docs/lance.md`

Modify these files:

- `src/musubi_tuner/dataset/image_video_dataset.py`
- `src/musubi_tuner/networks/network_arch.py`
- `src/musubi_tuner/utils/sai_model_spec.py`
- `docs/dataset_config.md`
- `README.md`

## Architecture IDs

Add one short architecture ID first:

```python
ARCHITECTURE_LANCE = "la"
ARCHITECTURE_LANCE_FULL = "lance"
```

If image and video checkpoints need incompatible cache or LoRA metadata later,
split into `lance_image` and `lance_video`. Do not split until there is an
actual incompatibility.

## Loader Contract

`lance_utils.load_lance_model(args, device, dtype)` should:

1. Load `Qwen2Config` from `<model_path>/llm_config.json`.
2. Apply Lance-specific config fields:
   `layer_module=Qwen2MoTDecoderLayer`, q/k norm flags, `freeze_und`, and
   Qwen2.5-VL position embedding options.
3. Construct `Qwen2ForCausalLM`.
4. Load optional `Qwen2.5-VL-ViT` only when the task requires visual context or
   understanding tokens.
5. Load `WanVideoVAE` from `--vae` or from the official model tree.
   This must be the Lance-compatible 48-channel VAE path, not the current Wan
   16-channel wrapper.
6. Construct `LanceConfig` with public-weight defaults:
   `visual_gen=true`, `visual_und=true`, `max_num_frames=121`,
   `max_latent_size=64`, and `latent_patch_size=(1, 1, 1)`.
7. Load Lance checkpoint shards with `safetensors`.
8. Add Lance special tokens and resize embeddings only when required by the
   official tokenizer state.

The loader may copy official Apache-2.0 code into `src/musubi_tuner/lance/*`,
but imports must be made package-local and must not depend on running from the
official repository root.

## Data Contract

Musubi dataset config remains the user-facing input. Lance-specific sequence
packing is internal.

For T2I/T2V:

```toml
[[datasets]]
image_directory = "path/to/images"
caption_extension = ".txt"
resolution = [768, 768]
batch_size = 1
```

or:

```toml
[[datasets]]
video_directory = "path/to/videos"
caption_extension = ".txt"
target_frames = 49
frame_extraction = "head"
batch_size = 1
```

For edit tasks, reuse existing `control_directory` or JSONL `control_path`
fields. The Lance packer maps:

- caption text to text tokens;
- source/control media to clean VAE or ViT context tokens;
- target media to noisy VAE tokens with MSE loss indexes.

The packer must produce one canonical batch dictionary:

- `sequence_length`
- `packed_text_ids`
- `packed_text_indexes`
- `sample_lens`
- `sample_type`
- `sample_N_target`
- `packed_position_ids`
- `split_lens`
- `attn_modes`
- `packed_label_ids`
- `ce_loss_indexes`
- `padded_latent`
- `patchified_vae_latent_shapes`
- `packed_latent_position_ids`
- `packed_vae_token_indexes`
- `packed_timesteps`
- `mse_loss_indexes`
- `packed_vit_tokens`
- `packed_vit_token_indexes`
- `packed_vit_position_ids`
- `vit_token_seqlens`
- `vit_video_grid_thw`
- `vae_video_grid_thw`
- `video_grid_thw`
- `vit_data_mode`
- `key_frame_mask`
- `sample_task`
- `sample_modality`

## Cache Contract

`lance_cache_latents.py`:

- uses Wan2.2 VAE;
- stores latents as `[t, h, w, 48]`;
- records original media size and target frame count;
- records `vae_downsample=(4,16,16)`;
- records `latent_patch_size=(1,1,1)` by default;
- supports target and control latents.

`lance_cache_text_encoder_outputs.py`:

- initially caches tokenized prompt packs, not only final embeddings;
- records tokenizer special token ids;
- stores enough metadata to rebuild the packed sequence without re-tokenizing;
- does not cache ViT embeddings in phase 1 unless profiling proves it is needed.

Reason: Lance text, ViT, and VAE token positions are coupled. A cache that only
stores final text embeddings is too weak for reliable edit and understanding
tasks.

## Training Contract

`LanceNetworkTrainer` should subclass `NetworkTrainer` and implement:

- `architecture`
- `architecture_full_name`
- `handle_model_specific_args`
- `load_vae`
- `load_transformer`
- `scale_shift_latents` as identity unless official normalization requires more
- `call_dit` or the Lance equivalent that calls `Lance.forward`
- `do_inference`
- `process_sample_prompts`

The first pass uses:

- `--mixed_precision bf16`;
- `--network_module networks.lora_lance`;
- `--timestep_sampling shift` or a Lance-specific alias that reproduces
  logit-normal plus timestep shift;
- default validation settings from official inference:
  `validation_num_timesteps=30`, `validation_timestep_shift=3.5`,
  `cfg_text_scale=4.0`.

The trainer reduces `outputs["mse"]` exactly like other flow-matching trainers.
`outputs["ce"]` is ignored or asserted absent in phase 1.

## LoRA Contract

`networks/lora_lance.py` should target:

- `language_model.model.layers.*.self_attn.q_proj_moe_gen`
- `language_model.model.layers.*.self_attn.k_proj_moe_gen`
- `language_model.model.layers.*.self_attn.v_proj_moe_gen`
- `language_model.model.layers.*.self_attn.o_proj_moe_gen`
- `language_model.model.layers.*.mlp_moe_gen.*`
- `vae2llm`
- `llm2vae`

Optional, behind explicit include patterns only:

- shared `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- shared `mlp`;
- `time_embedder`.

Default excludes:

- `vit_model.*`
- `language_model.model.embed_tokens`
- `language_model.lm_head`
- `latent_pos_embed`
- VAE modules

This keeps phase 1 focused on visual generation behavior and avoids corrupting
understanding or tokenizer semantics before CE training is supported.

## Sampling Contract

`lance_generate.py` and training samples must:

- support `t2i` and `t2v` first;
- support `image_edit` and `video_edit` after source-context packing is proven;
- use the same sequence packer as training;
- decode latents through Wan2.2 VAE;
- expose `--use_kvcache`, but default it off until parity tests pass;
- accept `--lora_weight`, `--lora_multiplier`, `--include_patterns`, and
  `--exclude_patterns`.

## Verification Plan

1. Unit-test tokenizer setup and special token ids against a local Lance
   checkpoint.
2. Unit-test VAE encode/decode shapes:
   `[3,T,H,W] -> [t,h,w,48] -> [3,T,H,W]`.
3. Unit-test packed sequence index consistency:
   every index tensor must be in range and disjoint where required.
4. Run `lance_generate.py --task t2i` on one prompt without LoRA.
5. Run `lance_generate.py --task t2v` on one prompt without LoRA.
6. Run a one-step LoRA train on one image or one short video.
7. Resume from the produced LoRA and verify sampling changes with multiplier
   `0.0` versus `1.0`.
8. Verify saved metadata contains `modelspec.architecture=lance`.

## Open Questions

- Does the public image checkpoint and video checkpoint require different
  `latent_patch_size` or only different weights?
- Should edit training use clean source VAE tokens, ViT source tokens, or both
  by default?
- Is Lance LoRA useful when only generation expert modules are trainable, or do
  shared attention and MLP modules need opt-in support earlier?
- Can `torch.nn.attention.flex_attention` be made optional for Windows users, or
  is it a hard runtime dependency?
- Do official prompt templates need to be exposed in dataset config, or can they
  be fixed per task?

## Milestones

1. Port loader and generation-only inference.
2. Add latent cache and no-op text/token cache.
3. Add sequence packer for `t2i` and `t2v`.
4. Add `lora_lance` and one-step LoRA train.
5. Add sample generation during training.
6. Add edit task packing.
7. Add CE/understanding training only after generation is stable.

## Rejection Criteria

Reject the implementation if any of these happen:

- Lance is loaded through a DiT-shaped wrapper that hides packed sequence
  indexes.
- Dataset cache metadata cannot detect Lance versus Qwen-Image or Wan caches.
- Sample generation has a separate prompt path from training.
- LoRA defaults touch tokenizer embeddings, VAE, or ViT.
- The first PR attempts to support all six Lance tasks before a one-step
  generation train passes.

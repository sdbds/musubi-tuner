# Z-Image D-OPSD LoRA Spec

## Goal

Implement the low-VRAM variant of D-OPSD for Z-Image network training:

- train only adapter parameters
- cache both student text embeddings and teacher multimodal embeddings before training
- keep one frozen base transformer in memory
- keep one EMA copy of adapter parameters, not a second transformer
- run distillation loss timestep by timestep and call backward after each timestep

The baseline LoRA/SFT path must remain byte-for-byte behaviorally unchanged unless `--dopsd` is set.

## Non-Goals

- Full-parameter D-OPSD.
- Online VLM/text encoder execution during training.
- Reproducing the paper's Qwen3-VL LLM reweighting inside the trainer. The cache script can apply this offline.
- Control/Omni Z-Image D-OPSD.
- Combining SFT loss with D-OPSD loss by default.

## Cache Contract

Normal Z-Image text cache keeps:

```text
varlen_llm_embed_<dtype>
```

D-OPSD adds one optional key to the same `_te.safetensors` file:

```text
varlen_dopsd_teacher_llm_embed_<dtype>
```

The trainer reads this as batch key:

```text
dopsd_teacher_llm_embed
```

The teacher embedding last dimension must match Z-Image `cap_feat_dim` (`2560`). This is deliberate. Training should fail early if someone caches teacher outputs from a VLM with the wrong language hidden size.

## Training Algorithm

For each batch:

1. Load target latents and cached student/teacher embeddings.
2. Initialize rollout state from Gaussian noise with the same shape as target latents.
3. For each Z-Image inference timestep:
   - swap live adapter weights to EMA weights
   - run teacher prediction with cached multimodal teacher embedding under `torch.no_grad()`
   - restore live student adapter weights
   - run student prediction with cached text embedding
   - minimize `MSE(student_velocity, stopgrad(teacher_velocity)) / K`
   - call `accelerator.backward(...)` immediately
   - update rollout state with detached student prediction and Z-Image's inference sign convention
4. After the optimizer step, update EMA adapter weights.

This preserves the main memory property: at no point do we retain all K student graphs until a final backward.

## CLI

Training:

```bash
python src/musubi_tuner/zimage_train_network.py \
  ...existing LoRA args... \
  --dopsd \
  --dopsd_num_sampling_steps 8 \
  --dopsd_ema_decay 0.9999
```

Teacher cache:

```bash
python src/musubi_tuner/zimage_cache_text_encoder_outputs.py \
  --dataset_config path/to/dataset.toml \
  --text_encoder path/to/qwen3-4b \
  --dopsd_cache_teacher_outputs \
  --dopsd_teacher_text_encoder path/to/qwen3-vl-compatible-model \
  --dopsd_teacher_llm_reweight_source path/to/qwen3-4b
```

If the teacher VLM checkpoint was already prepared with Qwen3-4B language weights, pass
`--dopsd_teacher_already_reweighted` instead of `--dopsd_teacher_llm_reweight_source`.
`--dopsd_teacher_allow_raw_vlm` is only for ablations and is not paper-consistent for Z-Image.

The teacher cache path rejects text-only models and processors that do not produce image tensors. This keeps
the teacher condition aligned with `f_mm(y, x0)` instead of silently falling back to text-only features.

## LyCORIS Extension

The EMA and swap logic operates on `network.named_parameters()` rather than LoRA-specific names. That makes the training path compatible with LoHa/LoKr-style LyCORIS modules as long as:

- the network module injects into the same Z-Image transformer forward path
- trainable adapter tensors are exposed as parameters
- `network.apply_to(...)`, `prepare_optimizer_params(...)`, and `save_weights(...)` follow the existing LoRA network contract

The current `networks.loha` and `networks.lokr` modules satisfy that contract through the shared `LoRANetwork` wrapper and architecture auto-detection. This should work mechanically, but it still needs image-quality validation because LoHa/LoKr update geometry is not identical to LoRA.

## Failure Modes

- Missing `dopsd_teacher_llm_embed`: recache with teacher outputs.
- Teacher hidden size not `2560`: use a compatible Qwen3-VL teacher or precompute compatible embeddings externally.
- Teacher LLM reweight failure: use a Qwen3 text model with the same language hidden architecture as the Qwen3-VL teacher, or provide an externally reweighted VLM.
- Poor teacher multimodal behavior: D-OPSD will faithfully distill a bad teacher; this is a data/model capability failure, not a trainer bug.
- Higher compute: expected. This path trades FLOPs for lower peak VRAM.

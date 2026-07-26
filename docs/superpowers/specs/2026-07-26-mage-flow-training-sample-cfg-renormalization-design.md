# Mage-Flow Training-Sample CFG Renormalization

## Goal

Enable CFG renormalization by default for Mage-Flow preview images generated
during training. This addresses the highlight clipping and overexposure observed
with CFG sampling after applying a Mage-Flow LoRA.

## Scope

- Change only `MageFlowNetworkTrainer.do_inference`.
- Always pass `renormalize_cfg=True` to Mage-Flow's existing `sample_latents`
  call.
- Do not add a training CLI option or a per-prompt sampling option.
- Do not change the shared sampling-prompt parser or any other architecture.
- Keep the standalone `mage_flow_generate_image.py --renormalize_cfg` interface
  unchanged for CLI compatibility.
- Keep the reusable `sample_latents` default unchanged so direct callers retain
  their current behavior.

## Data Flow

The training sampler already constructs conditional and blank-negative
conditioning for CFG. `MageFlowNetworkTrainer.do_inference` will select the
existing Mage-Flow per-token velocity-norm renormalization when it calls
`sample_latents`. The underlying CFG formula and Euler schedule are unchanged.

## Compatibility

Training, loss calculation, saved LoRA weights, cache formats, and inference for
other architectures remain unchanged. Only Mage-Flow preview images emitted by
`--sample_at_first`, `--sample_every_n_steps`, or
`--sample_every_n_epochs` change.

## Verification

Add a focused test that replaces `sample_latents` with a capture function and
asserts that Mage-Flow training inference passes `renormalize_cfg=True`.
Existing Mage-Flow sampling and training tests must continue to pass.

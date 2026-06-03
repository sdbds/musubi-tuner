# Colored Noise Sampling Spec

## Goal

Implement Colored Noise Sampling (CNS) as an opt-in sampler extension for Musubi inference and training-preview sampling.

The reference implementation is Hadar Davidson et al.'s colored-noise-sampling repository. Its core idea is not to change the denoising model or timestep count, but to replace each SDE step's isotropic white noise with spectrally shaped noise. The shaping is driven by a precomputed gamma matrix over radial frequency bands and sampling progress, then globally renormalized so the injected noise keeps unit variance.

Musubi uses many deterministic flow/Euler/UniPC paths that do not have an SDE random term. For those paths, CNS applies to the initial latent noise. For SDE-capable paths, CNS also applies to the scheduler's per-step noise injection.

## Integration Scope

Supported entry points:

- Inference scripts: HunyuanVideo, HunyuanVideo 1.5, Wan, FramePack, Qwen-Image, Flux Kontext, Flux.2, Z-Image, and Kandinsky5.
- Training preview sampling: architecture-specific `sample_images` paths for the same model families.
- Wan DPM++ with `--dpm_algorithm_type sde-dpmsolver++` additionally colors the SDE step noise inside `FlowDPMSolverMultistepScheduler`.

Layout handling:

- BCHW image latents and BCTHW video latents shape the last two spatial axes.
- Packed 2x2 latent sequences are unpacked to their latent spatial grid, shaped, then packed back with the original layout.
- Kandinsky5 NHWC visual latents shape `(height, width)` instead of the trailing channel axis.

## Non-Goals

- Do not change default sampling behavior.
- Do not add or vendor the full reference repository.
- Do not generate gamma matrices inside Musubi in this change.
- Do not add bundled gamma matrix artifacts.
- Do not change the training forward noising distribution used for loss calculation.
- Do not claim deterministic ODE/Euler/UniPC paths reproduce the reference SDE CNS exactly; they receive colored initial latent noise only.

## User Interface

All supported inference and training-preview sampling entry points gain CNS options:

- `--cns`; enables colored noise shaping.
- `--cns_gamma_matrix_path PATH`; required when `--cns` is enabled.
- `--cns_gamma_matrix_divider FLOAT`; default `1.0`.
- `--cns_sqrt_gamma`; use square root of residual gamma energy.
- `--cns_power_gamma FLOAT`; default `1.0`.
- `--cns_alpha_tilting [FLOAT [FLOAT]]`; zero, one, or two alpha values. Two values interpolate over sampling progress.
- `--cns_alpha_tilting_inside_exp`.
- `--cns_alpha_tilting_use_fnorm`.
- `--cns_alpha_exponential_interpolation`.
- `--cns_alpha_exponential_interpolation_sharpness FLOAT`; default `4.0`.
- `--cns_energy_scale FLOAT`; default `1.0`.

Wan inference also exposes explicit DPM options:

- `--dpm_algorithm_type {dpmsolver++,sde-dpmsolver++}`; default remains `dpmsolver++`.
- `--dpm_solver_order INT`; default `2`.
- `--dpm_solver_type {midpoint,heun}`; default `midpoint`.

Validation:

- `--cns` requires a readable gamma matrix path.
- Numeric CNS parameters must be positive where required.

## Algorithm

For initial latent noise:

1. Generate the sampler's normal Gaussian initial noise as before.
2. Select the first gamma row, or map to sampling progress if a caller supplies a specific step.
3. Shape spatial frequencies in FFT space.
4. Renormalize to unit standard deviation.
5. Continue with the existing sampler unchanged.

For each SDE step:

1. Generate the scheduler's normal Gaussian noise as before.
2. Select the gamma row for the current step. If gamma row count differs from inference step count, map by relative progress.
3. Compute residual energy per radial frequency bin:
   - `base_residual = 1 - gamma_row / gamma_matrix_divider`
   - optional alpha tilting follows the reference implementation.
   - final per-band scale is residual, sqrt residual, or residual power.
4. Build a radial frequency index grid over the latent spatial axes.
5. Apply the per-frequency scale in FFT space over the last two dimensions.
6. Convert back with inverse FFT and renormalize to unit standard deviation.
7. Pass the shaped noise into the existing SDE-DPM update formulas.

For 5D video latents, CNS shapes spatial frequency over `(height, width)` and broadcasts across batch, channel, and frame dimensions. Temporal frequency shaping is left out of scope for this pass.

## Testing

Unit tests cover:

- 4D and 5D colored noise shaping preserves shape and dtype.
- non-trailing spatial axes for NHWC-style latents.
- packed 2x2 latent unpack/shape/repack layout preservation.
- shared CLI helpers validate and cache gamma matrices.
- shaped noise is finite and approximately unit standard deviation.
- DPM SDE scheduler can run one colored-noise step on a 5D latent.
- invalid CNS configuration raises clear errors.

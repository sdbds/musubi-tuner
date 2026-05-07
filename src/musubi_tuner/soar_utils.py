from __future__ import annotations

from typing import Optional

import torch


def per_point_aux_scale(lambda_aux: float, trajectory_length: int) -> float:
    if trajectory_length < 1:
        raise ValueError("trajectory_length must be at least 1")
    return float(lambda_aux) / float(trajectory_length)


def _coerce_batch_scalar(name: str, value: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    scalar = torch.as_tensor(value, dtype=torch.float32, device=device)
    if scalar.ndim == 0:
        scalar = scalar.expand(batch_size)
    elif scalar.ndim != 1 or scalar.shape[0] != batch_size:
        raise ValueError(f"{name} must be a scalar or a tensor with shape [{batch_size}]")
    return scalar


def _expand_weights(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    view_shape = (weight.shape[0],) + (1,) * (target.ndim - 1)
    return weight.view(view_shape)


def t_to_sigma_timestep(t: torch.Tensor, noise_scheduler) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.as_tensor(t, dtype=torch.float32, device=t.device)
    num_steps = len(noise_scheduler.timesteps)
    indices = ((1.0 - t) * num_steps).long().clamp(0, num_steps - 1)
    sigmas = noise_scheduler.sigmas.to(device=t.device, dtype=torch.float32)[indices]
    timesteps = noise_scheduler.timesteps.to(device=t.device, dtype=torch.float32)[indices]
    return sigmas, timesteps


def sigma_to_t(sigma: torch.Tensor, noise_scheduler) -> torch.Tensor:
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=sigma.device)
    num_steps = len(noise_scheduler.timesteps)
    sched_sigmas = noise_scheduler.sigmas[:num_steps].to(device=sigma.device, dtype=torch.float32)
    indices = torch.searchsorted(-sched_sigmas, -sigma).clamp(0, num_steps - 1)
    return 1.0 - indices.float() / num_steps


def sigma_to_training_timestep(sigma: torch.Tensor, noise_scheduler, timestep_offset: float = 1.0) -> torch.Tensor:
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=sigma.device)
    scheduler_config = getattr(noise_scheduler, "config", None)
    num_train_timesteps = getattr(scheduler_config, "num_train_timesteps", len(noise_scheduler.timesteps))
    return sigma * float(num_train_timesteps) + float(timestep_offset)


def build_single_step_ode_aux_points(
    *,
    start_state: torch.Tensor,
    end_state: torch.Tensor,
    sigma_start: torch.Tensor,
    sigma_end: torch.Tensor,
    trajectory_length: int,
    generator: Optional[torch.Generator] = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if trajectory_length < 1:
        raise ValueError("trajectory_length must be at least 1")
    if start_state.shape != end_state.shape:
        raise ValueError("start_state and end_state must have the same shape")
    if start_state.ndim < 2:
        raise ValueError("start_state must be batched")

    batch_size = start_state.shape[0]
    sigma_start = _coerce_batch_scalar("sigma_start", sigma_start, batch_size, start_state.device)
    sigma_end = _coerce_batch_scalar("sigma_end", sigma_end, batch_size, start_state.device)

    points: list[tuple[torch.Tensor, torch.Tensor]] = []
    sigma_span = sigma_end - sigma_start
    safe_denominator = torch.where(sigma_span == 0, torch.ones_like(sigma_span), sigma_span)
    start_state_fp32 = start_state.float()
    end_state_fp32 = end_state.float()

    for _ in range(trajectory_length):
        random_weight = torch.rand(batch_size, device=start_state.device, generator=generator, dtype=torch.float32)
        aux_sigma = sigma_start + random_weight * sigma_span
        interpolation_weight = (aux_sigma - sigma_start) / safe_denominator
        aux_state = torch.lerp(start_state_fp32, end_state_fp32, _expand_weights(interpolation_weight, start_state_fp32))
        aux_state = aux_state.to(dtype=start_state.dtype)
        points.append((aux_state, aux_sigma))

    return points


@torch.no_grad()
def single_step_aux_points(
    *,
    z_t0: torch.Tensor,
    sigma_t0: torch.Tensor,
    v_standard: torch.Tensor,
    z_noise: torch.Tensor,
    points_per_path: int,
    noise_scheduler,
    num_sampling_steps: int,
    sigma_upper_ratio: float = 1.5,
    sigma_upper: Optional[torch.Tensor] = None,
) -> list[dict[str, torch.Tensor]]:
    if points_per_path < 1:
        return []
    if num_sampling_steps < 1:
        raise ValueError("num_sampling_steps must be at least 1")

    sigma_t0_1d = _coerce_batch_scalar("sigma_t0", sigma_t0, z_t0.shape[0], z_t0.device).clamp(0.0, 1.0)
    sigma_t1_1d = (sigma_t0_1d.detach() - 1.0 / float(num_sampling_steps)).clamp_min(0.0)

    if sigma_upper is None:
        sigma_upper = (sigma_t0_1d * float(sigma_upper_ratio)).clamp(max=1.0)

    sigma_t0 = sigma_t0_1d.view(-1, *([1] * (z_t0.ndim - 1)))
    sigma_t1 = sigma_t1_1d.view(-1, *([1] * (z_t0.ndim - 1)))
    z_t1_prime = (z_t0.float() + v_standard.float() * (sigma_t1 - sigma_t0)).to(dtype=z_t0.dtype).detach()

    aux_pairs = build_single_step_ode_aux_points(
        start_state=z_t1_prime,
        end_state=z_noise,
        sigma_start=sigma_t1_1d,
        sigma_end=sigma_upper,
        trajectory_length=points_per_path,
    )

    points: list[dict[str, torch.Tensor]] = []
    for z_t_prime, sigma_t_prime_1d in aux_pairs:
        timesteps_t_prime = sigma_to_training_timestep(sigma_t_prime_1d, noise_scheduler)
        points.append(
            {
                "latents": z_t_prime.detach(),
                "sigmas": sigma_t_prime_1d.detach(),
                "timesteps": timesteps_t_prime.detach(),
            }
        )
    return points

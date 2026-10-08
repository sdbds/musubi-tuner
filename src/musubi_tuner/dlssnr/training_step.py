"""Differentiable NR clip losses, independent of the optimizer and Accelerate."""

from __future__ import annotations

import torch

from musubi_tuner.dlssnr.losses import _rho, masked_gaussian_lowpass
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.temporal import warp_bilinear


def _supervised_mask(batch, burn_in):
    source = batch["source"]
    mask = batch.get("loss_mask", torch.ones_like(source[..., :1, :, :]))
    return mask if source.ndim == 4 else torch.cat(list(mask[:, burn_in:].unbind(1)), dim=0)


def _temporal_mask(batch, index):
    motion = batch["motion"][:, index]
    height, width = motion.shape[-2:]
    x = torch.arange(width, device=motion.device).view(1, 1, width) + motion[:, 0]
    y = torch.arange(height, device=motion.device).view(1, height, 1) + motion[:, 1]
    inside = ((x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1))[:, None]
    active = (~batch["reset"][:, index]).view(-1, 1, 1, 1)
    return batch["temporal_valid"][:, index].float() * inside * active


def _frequency_temporal_masks(batch, index):
    masks = batch.get("loss_mask")
    if masks is None:
        masks = torch.ones_like(batch["source"][:, :, :1])
    previous = masks[:, index - 1]
    warped, _ = warp_bilinear(previous, batch["motion"][:, index])
    valid = _temporal_mask(batch, index) * masks[:, index] * warped
    return valid, previous, warped


def loss_denominators(batch, burn_in, *, loss_profile=None, include_dino=False, include_base_anchor=False):
    mask = _supervised_mask(batch, burn_in)
    count = 3 * mask.sum()
    edge = 3 * ((mask[..., 1:] * mask[..., :-1]).sum() + (mask[..., 1:, :] * mask[..., :-1, :]).sum())
    temporal = mask.new_zeros(())
    if batch["source"].ndim == 5:
        for index in range(burn_in + 1, batch["source"].shape[1]):
            valid = _temporal_mask(batch, index) if loss_profile is None else _frequency_temporal_masks(batch, index)[0]
            temporal = temporal + 3 * valid.sum()
    result = {"pre": float(count), "out": float(count), "edge": float(edge), "temporal": float(temporal)}
    if include_dino:
        result["dino"] = float(count)
    if include_base_anchor:
        result["base_anchor"] = float(count)
    return result


def _clip_outputs(model, batch, seeds, burn_in):
    history = None
    outputs = []
    frame_count = batch["source"].shape[1]
    if burn_in < 1 or burn_in >= frame_count:
        raise ValueError("burn_in must leave at least one differentiable frame")
    for index in range(frame_count):
        with torch.set_grad_enabled(torch.is_grad_enabled() and index >= burn_in):
            frame = forward_frame(
                model,
                batch["source"][:, index],
                batch["controls"][:, index],
                [item[index] for item in seeds],
                history=history,
                motion=batch["motion"][:, index] if history is not None else None,
                history_valid=batch["history_valid"][:, index] if history is not None else None,
                reset=batch["reset"][:, index],
            )
        history = frame["next_history"].detach() if index < burn_in else frame["next_history"]
        if index >= burn_in:
            outputs.append(frame)
    return outputs


def supervised_outputs(model, batch, seeds, burn_in):
    if batch["source"].ndim == 5:
        return _clip_outputs(model, batch, seeds, burn_in)
    return [forward_frame(model, batch["source"], batch["controls"], seeds)]


def training_loss(
    model, batch, seeds, weights, burn_in=0, normalizers=None, *, loss_profile=None, dino_loss=None, base_reference=None
):
    if loss_profile is not None and loss_profile.get("name") != "frequency_split":
        raise ValueError("unsupported loss_profile")
    use_dino = weights.get("dino", 0) > 0
    if use_dino and dino_loss is None:
        raise ValueError("positive DINO loss weight requires a frozen feature backend")
    use_base_anchor = weights.get("base_anchor", 0) > 0
    if use_base_anchor and base_reference is None:
        raise ValueError("positive base anchor weight requires a frozen reference")
    is_clip = batch["source"].ndim == 5
    outputs = supervised_outputs(model, batch, seeds, burn_in)
    preclamp = torch.cat([frame["neural_preclamp"] for frame in outputs], dim=0)
    rendered = torch.cat([frame["rendered_proxy"] for frame in outputs], dim=0)
    target = torch.cat(list(batch["target"][:, burn_in:].unbind(1)), dim=0) if is_clip else batch["target"]
    mask = _supervised_mask(batch, burn_in)
    pre_error, out_error = preclamp - target, rendered - target
    edge_target = target
    metric_names = {}
    if loss_profile is not None:
        sigma = loss_profile["lowpass_sigma"]
        pre_error = masked_gaussian_lowpass(pre_error, mask, sigma)
        out_error = masked_gaussian_lowpass(out_error, mask, sigma)
        edge_target = torch.cat(list(batch["source"][:, burn_in:].unbind(1)), dim=0) if is_clip else batch["source"]
        metric_names = {"pre": "lowpass_pre", "out": "lowpass_out", "edge": "input_edge", "temporal": "lowpass_temporal"}
    terms = {"pre": (_rho(pre_error) * mask).sum(), "out": (_rho(out_error) * mask).sum()}
    edge = rendered.new_zeros(())
    for dimension in (-1, -2):
        valid = mask[..., 1:] * mask[..., :-1] if dimension == -1 else mask[..., 1:, :] * mask[..., :-1, :]
        edge = edge + (_rho(rendered.diff(dim=dimension) - edge_target.diff(dim=dimension)) * valid).sum()
    terms["edge"] = edge
    terms["temporal"] = rendered.new_zeros(())
    if is_clip and weights.get("temporal", 0):
        for offset in range(1, len(outputs)):
            index = burn_in + offset
            if loss_profile is None:
                valid = _temporal_mask(batch, index)
                previous, _ = warp_bilinear(outputs[offset - 1]["rendered_proxy"], batch["motion"][:, index])
                previous_target, _ = warp_bilinear(batch["target"][:, index - 1], batch["motion"][:, index])
                error = (outputs[offset]["rendered_proxy"] - previous) - (batch["target"][:, index] - previous_target)
            else:
                valid, previous_mask, warped_mask = _frequency_temporal_masks(batch, index)
                previous_error = outputs[offset - 1]["rendered_proxy"] - batch["target"][:, index - 1]
                # Normalize the masked warp before blurring; invalid bilinear neighbors must not leak into valid pixels.
                previous_error = torch.where(previous_mask > 0, previous_error, 0) * previous_mask
                previous, _ = warp_bilinear(previous_error, batch["motion"][:, index])
                previous = previous / torch.where(warped_mask > 0, warped_mask, 1)
                error = outputs[offset]["rendered_proxy"] - batch["target"][:, index] - previous
                error = masked_gaussian_lowpass(error, valid, sigma)
            terms["temporal"] = terms["temporal"] + (_rho(error) * valid).sum()
    if use_dino:
        # Each image's statistic is weighted by its original valid RGB mass, not
        # a rank-local mean or the number of resized ViT tokens.
        per_image = dino_loss(rendered, target, mask)
        terms["dino"] = (per_image * (3 * mask.sum(dim=(1, 2, 3)))).sum()
    if use_base_anchor:
        if base_reference.shape != rendered.shape:
            raise ValueError("base anchor reference shape must match supervised rendered outputs")
        terms["base_anchor"] = (_rho(rendered - base_reference.detach()) * mask).sum()
    denominators = (
        normalizers
        if normalizers is not None
        else loss_denominators(
            batch, burn_in, loss_profile=loss_profile, include_dino=use_dino, include_base_anchor=use_base_anchor
        )
    )
    if denominators["pre"] <= 0:
        raise ValueError("loss mask has no supervised pixels")
    losses = {name: value / denominators[name] if denominators[name] > 0 else value * 0 for name, value in terms.items()}
    total = sum(weights.get(name, 0.0) * value for name, value in losses.items())
    if not torch.isfinite(total):
        raise RuntimeError("non-finite NR loss")
    metrics = {
        f"loss/{metric_names.get(name, name)}": float(value.detach())
        for name, value in losses.items()
        if name != "temporal" or weights.get(name, 0)
    }
    metrics["loss"] = float(total.detach())
    if use_dino:
        metrics["loss/dino_weighted"] = float((weights["dino"] * losses["dino"]).detach())
    if use_base_anchor:
        metrics["loss/base_anchor_weighted"] = float((weights["base_anchor"] * losses["base_anchor"]).detach())
    metrics["blend_max"] = max(float(frame["blend_weight"].detach().max()) for frame in outputs)
    return total, metrics

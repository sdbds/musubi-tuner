"""Export-aligned weight publication. Activation publication has different rounding."""

from __future__ import annotations

import torch

from musubi_tuner.dlssnr.profiles import build_records

STORAGE_DTYPES = {
    "e4": torch.float8_e4m3fn,
    "f16": torch.float16,
    "f16frag": torch.float16,
    "prior": torch.float16,
    "f32": torch.float32,
}


def native_storage_kinds() -> dict[str, str]:
    return {view.name: region.kind for record in build_records() for region in record.regions for view in region.views}


class _NativeWeightRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, kind):
        if weight.dtype != torch.float32:
            raise ValueError("native weight publication requires FP32 effective weights")
        dtype = STORAGE_DTYPES[kind]
        if not (torch.isfinite(weight) & (weight.abs() <= torch.finfo(dtype).max)).all():
            raise ValueError(f"native {kind} weights must be finite and within the storage range")
        return weight.to(dtype).float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def native_weight_ste(weight: torch.Tensor, kind: str) -> torch.Tensor:
    """Direct FP32 -> storage -> FP32, with identity gradients and no silent clipping."""
    if kind not in STORAGE_DTYPES:
        raise ValueError(f"unsupported native weight storage {kind}")
    return _NativeWeightRound.apply(weight, kind)


def _effective_weights(model, network=None):
    from musubi_tuner.dlssnr.fp8 import iter_canonical_tensors

    adapters = {adapter.target: adapter for adapter in network.adapters} if network is not None else {}
    attached = {module._dlssnr_lora_target for module in model.modules() if hasattr(module, "_dlssnr_lora_target")}
    if attached != set(adapters):
        raise ValueError("native diagnostics require the model's complete LoRA network")
    for name, value in iter_canonical_tensors(model):
        yield name, value + adapters[name].delta_weight() if name in adapters else value


@torch.no_grad()
def capture_native_reference(model, network=None) -> dict[str, torch.Tensor]:
    """Keep a detached CPU snapshot in native storage, not another FP32 model."""
    kinds = native_storage_kinds()
    return {
        name: native_weight_ste(value.detach().cpu(), kinds[name]).to(STORAGE_DTYPES[kinds[name]]).clone()
        for name, value in _effective_weights(model, network)
    }


def _summarize(rows):
    fields = (
        "values",
        "trained_changed_values",
        "blended_changed_values",
        "exported_changed_values",
        "rounded_values",
        "lost_update_values",
    )
    total = {field: sum(row[field] for row in rows) for field in fields}
    count = total["values"]
    total.update(
        flip_fraction=total["exported_changed_values"] / count,
        max_abs_quantization_error=max(row["max_abs_quantization_error"] for row in rows),
        mean_abs_quantization_error=sum(row["mean_abs_quantization_error"] * row["values"] for row in rows) / count,
        rmse=(sum(row["rmse"] ** 2 * row["values"] for row in rows) / count) ** 0.5,
    )
    return total


@torch.no_grad()
def native_quantization_report(model, reference, network=None) -> dict:
    from musubi_tuner.dlssnr.native import quantization_statistics, quantize_tensor

    kinds = native_storage_kinds()
    rows = []
    for name, value in _effective_weights(model, network):
        value = value.detach().cpu().contiguous().numpy()
        source = reference[name].float().numpy()
        if value.shape != source.shape:
            raise ValueError(f"{name}: native reference shape mismatch")
        quantized = quantize_tensor(value, kinds[name])
        rows.append(quantization_statistics(name, kinds[name], source, value, value, quantized))
    if not rows or {row["name"] for row in rows} != set(reference):
        raise ValueError("native reference parameter map does not match the model")
    return {
        "schema": "dlssnr_native_quantization_v1",
        "reference": "initial_native_weights",
        "mix": 1.0,
        "strength": 1.0,
        "lora_multiplier": 1.0,
        "native_equivalent": False,
        "totals": _summarize(rows),
        "by_storage": {
            kind: _summarize([row for row in rows if row["storage"] == kind]) for kind in sorted({row["storage"] for row in rows})
        },
        "tensors": rows,
    }

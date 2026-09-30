"""Single-process runtime services for NR, without diffusion trainer imports."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from accelerate import Accelerator


def reject_unsupported_runtime():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("multi-GPU NR training is not supported")
    if os.environ.get("ACCELERATE_MIXED_PRECISION", "no") not in ("no", "none", ""):
        raise RuntimeError("Accelerate mixed precision conflicts with FP32 NR training; use --mixed_precision no")


def create_accelerator(training):
    reject_unsupported_runtime()
    if training["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    accelerator = Accelerator(
        cpu=training["device"] == "cpu", mixed_precision="no", gradient_accumulation_steps=training["gradient_accumulation_steps"]
    )
    if accelerator.num_processes != 1 or accelerator.mixed_precision != "no":
        raise RuntimeError("NR supports one FP32 process only")
    if accelerator.device.type not in ("cpu", "cuda"):
        raise RuntimeError(f"unsupported NR device {accelerator.device}")
    if training["device"] != "auto" and accelerator.device.type != training["device"]:
        raise RuntimeError(f"requested {training['device']}, but Accelerate selected {accelerator.device}")
    return accelerator


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def assert_finite_gradients(module):
    parameters = module.parameters() if isinstance(module, torch.nn.Module) else module
    gradients = [parameter.grad for parameter in parameters if parameter.requires_grad and parameter.grad is not None]
    if not gradients:
        raise RuntimeError("no trainable gradients were produced")
    if not torch.stack([torch.isfinite(value).all() for value in gradients]).all():
        raise RuntimeError("non-finite gradient; optimizer update aborted")


def assert_finite_parameters(module):
    values = [torch.isfinite(parameter).all() for parameter in module.parameters() if parameter.requires_grad]
    if values and not torch.stack(values).all():
        raise RuntimeError("non-finite parameter update; checkpoint aborted")


def capture_rng():
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG devices do not match")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))


@contextmanager
def evaluation_mode(module):
    modes = [(child, child.training) for child in module.modules()]
    rng = capture_rng()
    try:
        module.eval()
        with torch.no_grad():
            yield
    finally:
        for child, mode in modes:
            child.training = mode
        restore_rng(rng)

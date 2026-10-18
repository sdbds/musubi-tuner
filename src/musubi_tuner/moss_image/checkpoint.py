"""Canonical full-model exports and resumable MOSS training progress."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from musubi_tuner.utils.safetensors_utils import mem_eff_save_file

PROGRESS_FILE = "moss_progress.json"


def model_config(model):
    return {
        name: getattr(model, name)
        for name in ("dim", "ffn_dim", "num_heads", "num_layers", "text_dim", "freq_dim", "latent_channels")
    } | {"rope_base": model.rope.base}


class _ExportTensors:
    """Cast one tensor at a time for the existing streaming safetensors writer."""

    def __init__(self, model, dtype):
        self.state = model.state_dict()
        self.dtype = dtype

    def items(self):
        for name, value in self.state.items():
            name = name.replace("_orig_mod.", "")
            dtype = torch.float32 if name == "rope.bands" else self.dtype if value.is_floating_point() else value.dtype
            yield name, value.detach().to(device="cpu", dtype=dtype or value.dtype).contiguous()


def save_full_model(model, path, dtype, metadata):
    model.wait_for_pending_block_moves()
    metadata = dict(metadata)
    metadata["moss_image_config"] = json.dumps(model_config(model), sort_keys=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        mem_eff_save_file(_ExportTensors(model, dtype), str(temporary), metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def dataset_fingerprint(group):
    digest = hashlib.sha256()
    for dataset in group.datasets:
        digest.update(str(dataset.batch_size).encode())
        for bucket, items in sorted(dataset.batch_manager.buckets.items()):
            digest.update(str(bucket).encode())
            for item in items:
                for path in (item.latent_cache_path, item.text_encoder_output_cache_path):
                    stat = os.stat(path)
                    digest.update(f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


@dataclass
class TrainingProgress:
    seed: int
    training_type: str
    gradient_accumulation_steps: int
    dataset_fingerprint: str
    num_processes: int = 1
    global_step: int = 0
    epoch: int = 0
    next_batch: int = 0
    version: int = 2
    num_timestep_buckets: int | None = None
    preserve_distribution_shape: bool = False
    timestep_range_pool: list = field(default_factory=list)

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        for key in (
            "version",
            "training_type",
            "gradient_accumulation_steps",
            "dataset_fingerprint",
            "num_processes",
            "seed",
            "num_timestep_buckets",
            "preserve_distribution_shape",
        ):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"MOSS resume {key} mismatch")
        for key in ("global_step", "epoch", "next_batch"):
            value = state.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid MOSS resume {key}")
            setattr(self, key, value)
        pool = state.get("timestep_range_pool")
        if not isinstance(pool, list) or any(
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bounds)
            or not 0 <= bounds[0] < bounds[1] <= 1
            for bounds in pool
        ):
            raise ValueError("Invalid MOSS resume timestep_range_pool")
        self.timestep_range_pool = [tuple(bounds) for bounds in pool]

    def write(self, directory, rank=None):
        path = Path(directory) / (PROGRESS_FILE if rank is None else f"moss_progress_rank{rank}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def read_progress(directory, rank=None):
    filename = PROGRESS_FILE if rank is None else f"moss_progress_rank{rank}.json"
    path = Path(directory) / filename
    if not path.is_file():
        raise ValueError(f"Not a MOSS training state (missing {filename}): {directory}")
    return json.loads(path.read_text(encoding="utf-8"))

"""Trainer progress not covered by Accelerate's checkpoint state."""

from dataclasses import asdict, dataclass, field
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

from accelerate.utils import load, RNG_STATE_NAME
import numpy as np
import torch


STATE_FILENAME = "trainer_state.json"
_PLAN_OPTIONS = (
    "gradient_accumulation_steps",
    "max_train_steps",
    "num_timestep_buckets",
    "timestep_sampling",
    "min_timestep",
    "max_timestep",
    "preserve_distribution_shape",
    "discrete_flow_shift",
    "optimizer_type",
    "optimizer_args",
    "mixed_precision",
    "learning_rate",
    "lr_scheduler",
    "lr_scheduler_type",
    "lr_scheduler_args",
    "lr_warmup_steps",
    "lr_decay_steps",
    "lr_scheduler_num_cycles",
    "lr_scheduler_power",
    "lr_scheduler_timescale",
    "lr_scheduler_min_lr_ratio",
)


@dataclass
class LocalTrainingState:
    seed: int
    plan: dict
    global_step: int
    epoch: int
    batch_index: int
    timestep_range_pool: list[tuple[float, float]] = field(default_factory=list)
    rng_state_sha256: str | None = None
    version: int = 1

    @classmethod
    def start(cls, args, batches_per_epoch: int, datasets, device):
        plan = {name: getattr(args, name) for name in _PLAN_OPTIONS}
        plan["batches_per_epoch"] = batches_per_epoch
        plan["dataset_batches"] = [[len(dataset), dataset.batch_size] for dataset in datasets]
        plan["device"] = str(device)
        return cls(args.seed, plan, global_step=0, epoch=0, batch_index=0)

    @classmethod
    def read(cls, directory):
        path = Path(directory) / STATE_FILENAME
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("incomplete"):
                raise ValueError("checkpoint save is incomplete; choose a different state directory")
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported format version")
            if "timestep_range_pool" not in data:
                raise ValueError("missing timestep_range_pool")
            state = cls(**data)
            if not isinstance(state.rng_state_sha256, str) or len(state.rng_state_sha256) != 64:
                raise ValueError("invalid RNG state digest")
            state._validate_cursor()
            return state
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid local training state at {path}: {exc}") from exc

    def _validate_cursor(self):
        values = (self.seed, self.global_step, self.epoch, self.batch_index)
        if any(type(value) is not int or value < 0 for value in values) or self.seed >= 2**32:
            raise ValueError("invalid seed or training cursor")
        if not isinstance(self.plan, dict) or self.plan.keys() != {
            *_PLAN_OPTIONS,
            "batches_per_epoch",
            "dataset_batches",
            "device",
        }:
            raise ValueError("invalid training plan")
        batches, accum, target = (self.plan[k] for k in ("batches_per_epoch", "gradient_accumulation_steps", "max_train_steps"))
        if any(type(value) is not int or value < 1 for value in (batches, accum, target)):
            raise ValueError("invalid batch, accumulation or step count")
        updates_per_epoch = (batches + accum - 1) // accum
        if (
            self.batch_index >= batches
            or self.batch_index % accum != 0
            or self.global_step != self.epoch * updates_per_epoch + self.batch_index // accum
            or self.global_step > target
        ):
            raise ValueError("training cursor is not on an optimizer-step boundary")
        if not isinstance(self.timestep_range_pool, list) or any(
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(type(value) not in (int, float) for value in bounds)
            or not 0 <= bounds[0] < bounds[1] <= 1 + 1e-12
            for bounds in self.timestep_range_pool
        ):
            raise ValueError("invalid timestep_range_pool")

    def validate_plan(self, current):
        for name, saved in self.plan.items():
            actual = current.plan[name]
            if type(saved) is not type(actual) or saved != actual:
                raise ValueError(f"Local resume requires the same {name}: checkpoint has {saved!r}, current run has {actual!r}")

    def advance(self, global_step: int, epoch: int, next_batch: int):
        self.global_step = global_step
        if next_batch == self.plan["batches_per_epoch"]:
            self.epoch, self.batch_index = epoch + 1, 0
        else:
            self.epoch, self.batch_index = epoch, next_batch


def _write_metadata(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_local_training_state(accelerator, directory, progress: LocalTrainingState | None):
    path = Path(directory) / STATE_FILENAME
    if progress is None and not path.exists():
        accelerator.save_state(directory)
        return

    # State directories can be reused. Invalidate the old cursor before any payload is overwritten.
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_metadata(path, {"incomplete": True})
    accelerator.save_state(directory)
    if progress is None:
        path.unlink()
    else:
        progress.rng_state_sha256 = hashlib.sha256(_read_rng_file(accelerator, directory)).hexdigest()
        _write_metadata(path, asdict(progress))


def _read_rng_file(accelerator, directory):
    path = Path(directory) / f"{RNG_STATE_NAME}_{accelerator.process_index}.pkl"
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read the RNG state required for local resume: {path}") from exc


def load_resume_rng(accelerator, directory, progress: LocalTrainingState):
    data = _read_rng_file(accelerator, directory)
    if hashlib.sha256(data).hexdigest() != progress.rng_state_sha256:
        raise ValueError(f"The RNG state does not match the trainer progress in {directory}")
    try:
        return load(BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not decode the RNG state required for local resume in {directory}") from exc


def verify_restored_rng(accelerator, saved):
    # Accelerate otherwise treats RNG restoration as best effort.
    np_state = np.random.get_state()
    saved_np = saved["numpy_random_seed"]
    restored = (
        accelerator.step == saved["step"]
        and random.getstate() == saved["random_state"]
        and np_state[0] == saved_np[0]
        and np.array_equal(np_state[1], saved_np[1])
        and np_state[2:] == saved_np[2:]
        and torch.equal(torch.get_rng_state(), saved["torch_manual_seed"])
    )
    if restored and accelerator.device.type == "cuda":
        index = accelerator.device.index
        index = torch.cuda.current_device() if index is None else index
        saved_cuda = saved.get("torch_cuda_manual_seed", [])
        restored = index < len(saved_cuda) and torch.equal(torch.cuda.get_rng_state(index), saved_cuda[index])
    if not restored:
        raise ValueError("Accelerate did not restore the RNG state required for local resume")

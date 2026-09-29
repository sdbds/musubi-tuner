"""Lazy, fixed-bucket paired clips with explicit encodings and motion semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from musubi_tuner.dlssnr.filenames import validate_filename
from musubi_tuner.dlssnr.identity import file_sha256, json_sha256


def _numeric(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf":
        raise ValueError(f"{path} must be a real numeric npy array")
    tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path} contains non-finite values")
    return tensor


def _proxy_image(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        tensor = _numeric(path)
    else:
        with Image.open(path) as image:
            if image.mode not in ("RGB", "RGBA", "L"):
                raise ValueError(f"{path}: unsupported image precision/mode {image.mode}; use FP32 CHW npy")
            array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if tensor.ndim != 3 or tensor.shape[0] != 3 or not ((tensor >= 0) & (tensor <= 1)).all():
        raise ValueError(f"{path} must be [3,H,W] srgb_proxy in [0,1]")
    return tensor


def _controls(path: Path) -> torch.Tensor:
    tensor = _numeric(path)
    if tensor.ndim != 3 or tensor.shape[0] != 5:
        raise ValueError(f"{path} must be [5,H,W], got {tuple(tensor.shape)}")
    return tensor


def _mask(path: Path, height: int, width: int, *, binary: bool) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        tensor = _numeric(path)
    else:
        with Image.open(path) as image:
            if image.mode not in ("1", "L"):
                raise ValueError(f"{path} must be a single-channel mask")
            tensor = torch.from_numpy(np.array(image.convert("L"), dtype=np.float32) / 255.0)[None]
    if tuple(tensor.shape) != (1, height, width):
        raise ValueError(f"{path}: expected mask [1,{height},{width}], got {tuple(tensor.shape)}")
    valid = (tensor == 0) | (tensor == 1) if binary else (tensor >= 0) & (tensor <= 1)
    if not valid.all():
        raise ValueError(f"{path}: mask must be {'binary' if binary else 'in [0,1]'}")
    return tensor.bool() if binary else tensor


class NRDataset(Dataset):
    def __init__(
        self, path, width, height, sequence_length, *, single_frame=False, require_target=True, require_temporal_mask=True
    ):
        self.path = Path(path).resolve()
        self.width, self.height = width, height
        self.single_frame = single_frame
        self.require_target = require_target
        self.require_temporal_mask = require_temporal_mask
        self.rows = []
        self.sequence_ids = set()
        sample_ids = set()
        self.files = {self.path}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            context = f"{self.path}:{line_number}"
            if row.get("schema") != "dlssnr_pairs_v1":
                raise ValueError(f"{context}: schema must be dlssnr_pairs_v1")
            for key, value in (("source_encoding", "srgb_proxy"), ("controls_encoding", "dlssnr_lanes_10_14_v1")):
                if row.get(key) != value:
                    raise ValueError(f"{context}: {key} must be {value}")
            if require_target and row.get("target_encoding") != "srgb_proxy":
                raise ValueError(f"{context}: target_encoding must be srgb_proxy")
            sample_id = row.get("sample_id")
            validate_filename(sample_id, f"{context}: sample_id")
            # Enforce case-insensitive output uniqueness even on case-sensitive hosts.
            sample_key = sample_id.upper().casefold()
            if sample_key in sample_ids:
                raise ValueError(f"{context}: sample_id {sample_id!r} collides with another filename (case-insensitive)")
            sample_ids.add(sample_key)
            sequence_id = row.get("sequence_id")
            if not isinstance(sequence_id, str) or not sequence_id:
                raise ValueError(f"{context}: sequence_id is required")
            self.sequence_ids.add(sequence_id)
            crop_id = row.get("crop_id", 0)
            if type(crop_id) is not int or crop_id < 0:
                raise ValueError(f"{context}: crop_id must be a nonnegative integer")
            frames = row.get("frames", [])
            if not isinstance(frames, list) or not frames or (sequence_length is not None and len(frames) != sequence_length):
                raise ValueError(f"{context}: sequence_length requires {sequence_length} frames")
            previous_index = -1
            for index, frame in enumerate(frames):
                frame_index = frame.get("frame_index")
                if type(frame_index) is not int or frame_index <= previous_index:
                    raise ValueError(f"{context}: frame_index must be nonnegative and strictly increasing")
                previous_index = frame_index
                reset = frame.get("reset")
                if type(reset) is not bool or (index == 0 and not reset):
                    raise ValueError(f"{context}: reset must be boolean and the first frame must reset")
                required = {"input_path", "controls_path"}
                if require_target:
                    required.add("target_path")
                if not reset:
                    required.update(("motion_path", "history_valid_path"))
                    if row.get("motion_layout") not in ("chw", "hwc"):
                        raise ValueError(f"{context}: motion_layout must be chw or hwc")
                    if require_temporal_mask:
                        required.add("temporal_valid_path")
                if missing := required - frame.keys():
                    raise ValueError(f"{context}: missing {sorted(missing)}")
                for key in (
                    "input_path",
                    "target_path",
                    "controls_path",
                    "motion_path",
                    "history_valid_path",
                    "temporal_valid_path",
                    "loss_mask_path",
                ):
                    if key in frame:
                        if not isinstance(frame[key], str) or not frame[key]:
                            raise ValueError(f"{context}: {key} must be a path")
                        file = (self.path.parent / frame[key]).resolve()
                        if not file.is_file():
                            raise FileNotFoundError(f"{context}: missing {file}")
                        frame[key] = str(file)
                        self.files.add(file)
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"{self.path} has no samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames = [self._load_frame(frame, row.get("motion_layout")) for frame in row["frames"]]
        metadata = {"sample_id": row["sample_id"], "sequence_id": row["sequence_id"], "crop_id": row.get("crop_id", 0)}
        return {**frames[0], **metadata} if self.single_frame else {"frames": frames, **metadata}

    def _load_frame(self, frame, motion_layout):
        source = _proxy_image(Path(frame["input_path"]))
        target = _proxy_image(Path(frame["target_path"])) if "target_path" in frame else None
        controls = _controls(Path(frame["controls_path"]))
        shape = (self.height, self.width)
        if source.shape[-2:] != shape or controls.shape[-2:] != shape or (target is not None and target.shape != source.shape):
            raise ValueError(f"frame {frame['frame_index']} does not match bucket {self.width}x{self.height}")
        motion = torch.zeros(2, *shape)
        history_valid = torch.zeros(1, *shape, dtype=torch.bool)
        if not frame["reset"]:
            motion = _numeric(Path(frame["motion_path"]))
            expected = (2, *shape) if motion_layout == "chw" else (*shape, 2)
            if tuple(motion.shape) != expected:
                raise ValueError(f"motion must have declared {motion_layout} shape {expected}")
            if motion_layout == "hwc":
                motion = motion.permute(2, 0, 1).contiguous()
            history_valid = _mask(Path(frame["history_valid_path"]), *shape, binary=True)
        temporal_valid = torch.zeros(1, *shape, dtype=torch.bool)
        if not frame["reset"] and "temporal_valid_path" in frame:
            temporal_valid = _mask(Path(frame["temporal_valid_path"]), *shape, binary=True)
        loss_mask = (
            _mask(Path(frame["loss_mask_path"]), *shape, binary=False) if "loss_mask_path" in frame else torch.ones(1, *shape)
        )
        if self.require_target and not loss_mask.any():
            raise ValueError(f"frame {frame['frame_index']}: loss mask has no supervised pixels")
        return {
            "source": source,
            "target": target,
            "controls": controls,
            "motion": motion,
            "history_valid": history_valid,
            "temporal_valid": temporal_valid,
            "loss_mask": loss_mask,
            "reset": frame["reset"],
            "frame_index": frame["frame_index"],
        }

    def validate(self):
        for index in range(len(self)):
            for _ in self.iter_frames(index):
                pass

    def iter_frames(self, index):
        row = self.rows[index]
        for frame in row["frames"]:
            yield self._load_frame(frame, row.get("motion_layout"))

    def fingerprint(self):
        return json_sha256(
            {"bucket": [self.width, self.height], "files": {str(path): file_sha256(path) for path in sorted(self.files)}}
        )


def load_single_frame_manifest(path, bucket_width, bucket_height, *, require_target=True):
    return NRDataset(
        path, bucket_width, bucket_height, 1, single_frame=True, require_target=require_target, require_temporal_mask=False
    )


def load_temporal_manifest(path, bucket_width, bucket_height, sequence_length, *, require_target=True, require_temporal_mask=None):
    if require_temporal_mask is None:
        require_temporal_mask = require_target
    return NRDataset(
        path,
        bucket_width,
        bucket_height,
        sequence_length,
        require_target=require_target,
        require_temporal_mask=require_temporal_mask,
    )


def collate_single_frames(samples):
    batch = {name: torch.stack([sample[name] for sample in samples]) for name in ("source", "target", "controls")}
    batch["loss_mask"] = torch.stack([sample.get("loss_mask", torch.ones_like(sample["source"][:1])) for sample in samples])
    return batch


def collate_clips(clips):
    fields = ("source", "target", "controls", "motion", "history_valid", "temporal_valid", "loss_mask")
    batch = {name: torch.stack([torch.stack([frame[name] for frame in clip["frames"]]) for clip in clips]) for name in fields}
    batch["reset"] = torch.tensor([[frame["reset"] for frame in clip["frames"]] for clip in clips], dtype=torch.bool)
    batch["sample_id"] = [clip["sample_id"] for clip in clips]
    return batch

"""Strict, expanded configuration for the NR supervised runner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import toml

from musubi_tuner.dlssnr.filenames import validate_filename
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.profiles import PROFILE_ID


DEFAULTS = {
    "model": {"profile": PROFILE_ID, "numerics_profile": "train_surrogate", "deployment_target": "float_runtime"},
    "data": {"require_cache": False},
    "training": {
        "mode": "single_frame",
        "seed": 42,
        "batch_size": 1,
        "sequence_length": 1,
        "burn_in": 0,
        "tbptt_length": 1,
        "gradient_accumulation_steps": 1,
        "max_train_steps": 1000,
        "gradient_checkpointing": False,
        "development_smoke": False,
        "device": "auto",
    },
    "optimizer": {"type": "AdamW", "learning_rate": 1e-5, "weight_decay": 0.0, "lr_scheduler": "constant"},
    "parameter_groups": {"prior_lr_multiplier": 0.1, "scale_lr_multiplier": 0.1, "temporal_blend_lr_multiplier": 0.1},
    "precision": {"mixed_precision": "no", "master_dtype": "float32"},
    "loss": {"pre": 1.0, "out": 1.0, "edge": 0.05, "temporal": 0.0},
    "evaluation": {"sample_every_n_steps": 0, "min_sequence_frames": 64, "compare_baseline": True},
    "output": {"output_dir": "../output/dlssnr", "output_name": "dlssnr", "save_every_n_steps": 100, "save_state": True},
}
EXTRA_FIELDS = {
    "model": {"model_dir", "forward_validation_report"},
    "data": {
        "train_manifest",
        "validation_manifest",
        "cache_directory",
        "source_encoding",
        "target_encoding",
        "controls_encoding",
        "bucket_size",
    },
    "evaluation": {"sequence_manifest"},
    "lora": {"profile", "rank", "alpha", "dropout", "qkv_mode", "rank_by_width", "alpha_by_width"},
}
PATH_FIELDS = {
    "model": ("model_dir", "forward_validation_report"),
    "data": ("train_manifest", "validation_manifest", "cache_directory"),
    "evaluation": ("sequence_manifest",),
    "output": ("output_dir",),
}


def _integer(value, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name: str, *, positive: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def validate_lora(table: dict) -> None:
    if table.get("qkv_mode", "fused_head_major") != "fused_head_major":
        raise ValueError("only qkv_mode = fused_head_major is implemented")
    dropout = table.get("dropout", 0.0)
    _number(dropout, "lora.dropout")
    if dropout >= 1:
        raise ValueError("lora.dropout must be < 1")
    profile = table.get("profile")
    if profile == "vit_only":
        if "rank_by_width" in table or "alpha_by_width" in table:
            raise ValueError("vit_only cannot also set a width rank map")
        _integer(table.get("rank"), "lora.rank")
        if table["rank"] > 1024:
            raise ValueError("vit_only rank cannot exceed 1024")
        _number(table.get("alpha"), "lora.alpha", positive=True)
    elif profile == "multiscale":
        if "rank" in table or "alpha" in table:
            raise ValueError("multiscale uses the width table, not a single rank/alpha")
        defaults = {"32": 2, "64": 4, "128": 8, "256": 8, "512": 16, "1024": 16}
        table.setdefault("rank_by_width", defaults.copy())
        table.setdefault("alpha_by_width", table["rank_by_width"].copy())
        for field in ("rank_by_width", "alpha_by_width"):
            values = table[field]
            if not isinstance(values, dict) or set(values) != set(defaults):
                raise ValueError(f"lora.{field} requires exactly widths {list(defaults)}")
            for width, value in values.items():
                if field == "rank_by_width":
                    _integer(value, f"lora.{field}.{width}")
                    if value > {"512": 64, "1024": 1024}.get(width, 32):
                        raise ValueError(f"rank exceeds a target's input/output width at C={width}")
                else:
                    _number(value, f"lora.{field}.{width}", positive=True)
    else:
        raise ValueError(f"unknown LoRA profile {profile}")
    table.setdefault("dropout", 0.0)
    table.setdefault("qkv_mode", "fused_head_major")


def load_train_config(path: str | Path, *, lora: bool = False, development_smoke: bool = False) -> dict:
    config_path = Path(path).resolve()
    raw = toml.load(config_path)
    if lora and "parameter_groups" in raw:
        raise ValueError("LoRA training rejects [parameter_groups]")
    if lora and "lora" not in raw:
        raise ValueError("LoRA training requires [lora]")
    if not lora and "lora" in raw:
        raise ValueError("full training rejects [lora]; use dlssnr_train_network.py")
    allowed = (set(DEFAULTS) - {"parameter_groups"} | {"lora"}) if lora else set(DEFAULTS)
    unknown = set(raw) - allowed - {"schema_version"}
    if unknown:
        raise ValueError(f"unknown config tables: {sorted(unknown)}")
    for section in allowed:
        table = raw.get(section, {})
        if not isinstance(table, dict):
            raise ValueError(f"{section} must be a table")
        unknown = set(table) - set(DEFAULTS.get(section, {})) - EXTRA_FIELDS.get(section, set())
        if unknown:
            raise ValueError(f"unsupported {section} fields: {sorted(unknown)}")
    # A temporal clip has no meaningful implicit lengths.
    if raw.get("training", {}).get("mode") == "temporal":
        if not {"sequence_length", "burn_in", "tbptt_length"} <= set(raw["training"]):
            raise ValueError("temporal mode requires sequence_length, burn_in, and tbptt_length")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    config = {"schema_version": 1}
    for section in sorted(allowed):
        config[section] = {**copy.deepcopy(DEFAULTS.get(section, {})), **raw.get(section, {})}
    training = config["training"]
    if development_smoke:
        training["development_smoke"] = True
    for section, field in (
        ("training", "development_smoke"),
        ("training", "gradient_checkpointing"),
        ("data", "require_cache"),
        ("evaluation", "compare_baseline"),
        ("output", "save_state"),
    ):
        if type(config[section][field]) is not bool:
            raise ValueError(f"{section}.{field} must be a boolean")
    for field in ("batch_size", "sequence_length", "tbptt_length", "gradient_accumulation_steps", "max_train_steps"):
        _integer(training[field], f"training.{field}")
    _integer(training["burn_in"], "training.burn_in", 0)
    _integer(training["seed"], "training.seed", 0)
    mode = training["mode"]
    if mode == "single_frame":
        if (training["sequence_length"], training["burn_in"], training["tbptt_length"]) != (1, 0, 1):
            raise ValueError("single-frame mode requires sequence_length=1, burn_in=0, tbptt_length=1")
        if config["loss"]["temporal"] != 0:
            raise ValueError("single-frame mode requires loss.temporal=0")
    elif mode == "temporal":
        if training["sequence_length"] != training["burn_in"] + training["tbptt_length"] or training["burn_in"] < 1:
            raise ValueError("sequence_length must equal burn_in + tbptt_length, with both at least 1")
        if config["loss"]["temporal"] and training["tbptt_length"] < 2:
            raise ValueError("temporal loss requires at least two supervised frames (tbptt_length >= 2)")
    else:
        raise ValueError("training.mode must be single_frame or temporal")
    if training["gradient_checkpointing"]:
        raise ValueError("gradient checkpointing is not supported")
    if training["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("training.device must be auto, cpu or cuda")
    model = config["model"]
    if model["profile"] != PROFILE_ID:
        raise ValueError(f"model.profile must be {PROFILE_ID}")
    if model["numerics_profile"] != "train_surrogate":
        raise ValueError("only train_surrogate is implemented")
    if model["deployment_target"] not in ("float_runtime", "native_roundtrip"):
        raise ValueError("unknown deployment_target")
    if not model.get("model_dir") and not training["development_smoke"]:
        raise ValueError("model.model_dir is required; random initialization is only allowed in development_smoke")
    data = config["data"]
    if data["require_cache"]:
        raise ValueError("pixel cache is not implemented; set data.require_cache=false")
    for field, expected in (
        ("source_encoding", "srgb_proxy"),
        ("target_encoding", "srgb_proxy"),
        ("controls_encoding", "dlssnr_lanes_10_14_v1"),
    ):
        if data.get(field) != expected:
            raise ValueError(f"data.{field} must be {expected}")
    if not data.get("train_manifest"):
        raise ValueError("data.train_manifest is required")
    bucket = data.get("bucket_size")
    if not isinstance(bucket, list) or len(bucket) != 2 or any(type(value) is not int for value in bucket):
        raise ValueError("data.bucket_size must contain integer width and height")
    resolve_geometry(*bucket)
    if config["precision"] != {"mixed_precision": "no", "master_dtype": "float32"}:
        raise ValueError("NR training is FP32 only")
    optimizer = config["optimizer"]
    if optimizer["type"] != "AdamW" or optimizer["lr_scheduler"] != "constant":
        raise ValueError("only AdamW with a constant learning rate is supported")
    if lora and "learning_rate" not in raw.get("optimizer", {}):
        optimizer["learning_rate"] = 1e-4
    _number(optimizer["learning_rate"], "optimizer.learning_rate", positive=True)
    _number(optimizer["weight_decay"], "optimizer.weight_decay")
    for section in ("loss", "parameter_groups"):
        for field, value in config.get(section, {}).items():
            _number(value, f"{section}.{field}")
    if not any(config["loss"].values()):
        raise ValueError("at least one loss weight must be positive")
    evaluation = config["evaluation"]
    _integer(evaluation["sample_every_n_steps"], "evaluation.sample_every_n_steps", 0)
    _integer(evaluation["min_sequence_frames"], "evaluation.min_sequence_frames")
    if evaluation["sample_every_n_steps"] and not (data.get("validation_manifest") or evaluation.get("sequence_manifest")):
        raise ValueError("evaluation requires a validation_manifest or sequence_manifest")
    _integer(config["output"]["save_every_n_steps"], "output.save_every_n_steps", 0)
    validate_filename(config["output"]["output_name"], "output.output_name")
    if lora:
        validate_lora(config["lora"])
    for section, fields in PATH_FIELDS.items():
        for field in fields:
            if field in config[section]:
                value = config[section][field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{section}.{field} must be a nonempty path")
                config[section][field] = str((config_path.parent / value).resolve())
    return config


def config_sha256(config: dict) -> str:
    """Hash effective values, not TOML formatting or comments."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

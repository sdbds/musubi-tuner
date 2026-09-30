"""Resolve dataset-only TOML and CLI arguments into a checked run snapshot."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import toml

from musubi_tuner.dlssnr.filenames import validate_filename
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.profiles import PROFILE_ID


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


def load_dataset_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = toml.load(path)
    if unknown := set(raw) - {"general", "datasets"}:
        raise ValueError(
            f"dataset TOML accepts only [general] and [[datasets]], not {sorted(unknown)}; "
            "pass model, optimizer and training settings as command-line arguments"
        )
    general = raw.get("general", {})
    datasets = raw.get("datasets")
    settings = {"resolution", "batch_size", "enable_bucket", "bucket_no_upscale"}
    if not isinstance(general, dict) or (unknown := set(general) - settings):
        raise ValueError(f"dataset [general] supports only {sorted(settings)}")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], dict):
        raise ValueError("DLSS-NR currently requires exactly one [[datasets]] manifest entry")
    dataset = datasets[0]
    paths = {"train_manifest", "validation_manifest", "sequence_manifest"}
    if unknown := set(dataset) - paths - settings:
        raise ValueError(f"unsupported dataset fields: {sorted(unknown)}; training settings belong on the command line")
    effective = {**general, **dataset}
    resolution = effective.get("resolution", [512, 512])
    if type(resolution) is int:
        resolution = [resolution, resolution]
    if not isinstance(resolution, list) or len(resolution) != 2 or any(type(value) is not int for value in resolution):
        raise ValueError("dataset resolution must be an integer or [width, height]")
    resolve_geometry(*resolution)
    batch_size = effective.get("batch_size", 1)
    _integer(batch_size, "dataset.batch_size")
    if not effective.get("train_manifest"):
        raise ValueError("dataset.train_manifest is required")
    data = {"bucket_size": resolution, "batch_size": batch_size}
    for name in ("enable_bucket", "bucket_no_upscale"):
        value = effective.get(name, False)
        if type(value) is not bool:
            raise ValueError(f"dataset.{name} must be a boolean")
        data[name] = value
    if not data["enable_bucket"]:
        data["bucket_no_upscale"] = False
    for name in paths:
        if name in effective:
            value = effective[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"dataset.{name} must be a nonempty path")
            data[name] = str((path.parent / value).resolve())
    return data


def _key_value_args(values, name, *, allow_strings=False):
    result = {}
    for item in values or []:
        key, separator, text = item.partition("=")
        if not separator or not key.isidentifier() or key in result or not text:
            raise ValueError(f"{name} requires unique key=value arguments, got {item!r}")
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if not allow_strings:
                raise ValueError(f"{name}: {key} must be a Python literal (for example betas=0.9,0.999)") from None
            value = text
        result[key] = value
    return result


def build_train_config(args, *, lora=False) -> dict:
    data = load_dataset_config(args.dataset_config)
    mode = args.training_mode
    lengths = (args.sequence_length, args.burn_in, args.tbptt_length)
    if mode == "temporal" and any(value is None for value in lengths):
        raise ValueError("temporal mode requires --sequence_length, --burn_in and --tbptt_length")
    lengths = [value if value is not None else default for value, default in zip(lengths, (1, 0, 1))]
    training = {
        "mode": mode,
        "seed": args.seed,
        "device": args.device,
        "development_smoke": args.development_smoke,
        "batch_size": data.pop("batch_size"),
        "sequence_length": lengths[0],
        "burn_in": lengths[1],
        "tbptt_length": lengths[2],
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": args.max_train_steps,
        "gradient_checkpointing": False,
    }
    for field in ("batch_size", "sequence_length", "tbptt_length", "gradient_accumulation_steps", "max_train_steps"):
        _integer(training[field], f"--{field}")
    _integer(training["burn_in"], "--burn_in", 0)
    _integer(training["seed"], "--seed", 0)
    if mode == "single_frame":
        if lengths != [1, 0, 1]:
            raise ValueError("single-frame mode requires sequence_length=1, burn_in=0, tbptt_length=1")
        if args.loss_temporal != 0:
            raise ValueError("single-frame mode requires --loss_temporal=0")
    elif mode == "temporal":
        if lengths[0] != lengths[1] + lengths[2] or lengths[1] < 1:
            raise ValueError("sequence_length must equal burn_in + tbptt_length, with both at least 1")
        if args.loss_temporal and lengths[2] < 2:
            raise ValueError("temporal loss requires at least two supervised frames (tbptt_length >= 2)")
    else:
        raise ValueError("--training_mode must be single_frame or temporal")
    if args.device not in ("auto", "cpu", "cuda") or args.mixed_precision != "no":
        raise ValueError("NR training is single-process FP32 on auto, cpu or cuda")
    if args.profile != PROFILE_ID or args.numerics_profile != "train_surrogate":
        raise ValueError("unsupported DLSS-NR profile or numerics")
    if args.deployment_target not in ("float_runtime", "native_roundtrip"):
        raise ValueError("unknown deployment_target")
    model = {"profile": args.profile, "numerics_profile": args.numerics_profile, "deployment_target": args.deployment_target}
    for name in ("model_dir", "forward_validation_report"):
        value = getattr(args, name)
        if value is not None:
            model[name] = str(Path(value).resolve())
    if not model.get("model_dir") and not args.development_smoke:
        raise ValueError("--model_dir is required; random initialization is only allowed in --development_smoke")

    optimizer_type = args.optimizer_type or "AdamW"
    optimizer_args = _key_value_args(args.optimizer_args, "--optimizer_args")
    if {"lr", "params"} & optimizer_args.keys():
        raise ValueError("use --learning_rate for lr; optimizer params are managed by the trainer")
    if optimizer_type.lower() in ("adamw", "torch.optim.adamw", "adamw8bit", "bitsandbytes.optim.adamw8bit"):
        optimizer_args.setdefault("weight_decay", 0.0)
    if "weight_decay" in optimizer_args:
        _number(optimizer_args["weight_decay"], "optimizer weight_decay")
    if optimizer_type.lower() in ("adafactor", "transformers.optimization.adafactor"):
        if optimizer_args.get("relative_step", True) or optimizer_args.get("warmup_init", False):
            raise ValueError("NR Adafactor requires --optimizer_args relative_step=False warmup_init=False (constant LR only)")
    if args.lr_scheduler != "constant":
        raise ValueError("only --lr_scheduler constant is implemented for NR")
    _number(args.learning_rate, "--learning_rate", positive=True)
    _number(args.max_grad_norm, "--max_grad_norm")
    optimizer = {
        "type": optimizer_type,
        "args": [f"{key}={value!r}" for key, value in sorted(optimizer_args.items())],
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "max_grad_norm": args.max_grad_norm,
    }
    loss = {name: getattr(args, f"loss_{name}") for name in ("pre", "out", "edge", "temporal")}
    for name, value in loss.items():
        _number(value, f"--loss_{name}")
    if not any(loss.values()):
        raise ValueError("at least one loss weight must be positive")
    evaluation = {
        "sample_every_n_steps": args.sample_every_n_steps,
        "min_sequence_frames": args.min_sequence_frames,
        "compare_baseline": args.compare_baseline,
    }
    if "sequence_manifest" in data:
        evaluation["sequence_manifest"] = data.pop("sequence_manifest")
    _integer(args.sample_every_n_steps, "--sample_every_n_steps", 0)
    _integer(args.min_sequence_frames, "--min_sequence_frames")
    if args.sample_every_n_steps and not (data.get("validation_manifest") or evaluation.get("sequence_manifest")):
        raise ValueError("evaluation requires a validation_manifest or sequence_manifest in the dataset TOML")
    validate_filename(args.output_name, "--output_name")
    _integer(args.save_every_n_steps, "--save_every_n_steps", 0)
    output = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "output_name": args.output_name,
        "save_every_n_steps": args.save_every_n_steps,
        "save_state": args.save_state,
    }
    config = {
        "schema_version": 2,
        "model": model,
        "data": data,
        "training": training,
        "optimizer": optimizer,
        "precision": {"mixed_precision": "no", "master_dtype": "float32"},
        "loss": loss,
        "evaluation": evaluation,
        "output": output,
    }
    if lora:
        network = _key_value_args(args.network_args, "--network_args", allow_strings=True)
        if unknown := set(network) - {"profile", "qkv_mode", "rank_by_width", "alpha_by_width"}:
            raise ValueError(f"unsupported --network_args: {sorted(unknown)}")
        network.setdefault("profile", "vit_only")
        if network["profile"] == "vit_only":
            network["rank"] = args.network_dim if args.network_dim is not None else 16
            network["alpha"] = args.network_alpha if args.network_alpha is not None else network["rank"]
        elif args.network_dim is not None or args.network_alpha is not None:
            raise ValueError("multiscale uses rank_by_width/alpha_by_width, not --network_dim/--network_alpha")
        network["dropout"] = args.network_dropout
        validate_lora(network)
        config["lora"] = network
    else:
        groups = {
            name: getattr(args, name) for name in ("prior_lr_multiplier", "scale_lr_multiplier", "temporal_blend_lr_multiplier")
        }
        for name, value in groups.items():
            _number(value, f"--{name}")
        config["parameter_groups"] = groups
    return config


def config_sha256(config: dict) -> str:
    """Hash effective values, not CLI argument order or dataset TOML comments."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

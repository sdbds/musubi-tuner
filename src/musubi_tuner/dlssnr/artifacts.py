"""Canonical directories and evidence checks. Training state is a separate artifact."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.dlssnr.identity import file_sha256, implementation_identity, json_sha256
from musubi_tuner.dlssnr.numerics import SURROGATE_FLAGS
from musubi_tuner.dlssnr.profiles import PROFILE_ID


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_tensors(path, tensors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".safetensors") as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_canonical(folder, *, development_smoke=False, validation_report=None):
    folder = Path(folder).resolve()
    if not (folder / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing canonical weights in {folder}")
    config_file = folder / "model_config.json"
    if not config_file.is_file():
        raise ValueError("canonical provenance is missing model_config.json")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("schema") != "dlssnr_canonical_v1" or config.get("profile") != PROFILE_ID:
        raise ValueError("canonical schema/profile does not match DLSS-NR 310.8.0")
    identity = {"model_sha256": file_sha256(folder / "model.safetensors")}
    for name in (
        "model_config.json",
        "source_manifest.json",
        "conversion_report.json",
        "opaque_records.safetensors",
        "numerics.json",
        "preprocessing.json",
    ):
        if (folder / name).is_file():
            identity[name] = file_sha256(folder / name)
    if not development_smoke:
        required = {
            "source_manifest.json",
            "conversion_report.json",
            "opaque_records.safetensors",
            "numerics.json",
            "preprocessing.json",
        }
        if missing := required - identity.keys():
            raise ValueError(
                f"canonical conversion/provenance is missing {sorted(missing)}; development_smoke is experimental only"
            )
        conversion = json.loads((folder / "conversion_report.json").read_text(encoding="utf-8"))
        if conversion.get("roundtrip") != "byte_identical" or conversion.get("profile") != PROFILE_ID:
            raise ValueError("canonical conversion has not passed source round-trip validation")
        report_file = Path(validation_report) if validation_report else folder / "forward_validation_report.json"
        if not report_file.is_file():
            raise ValueError("baseline forward validation is required; use explicit development_smoke for unvalidated experiments")
        report = json.loads(report_file.read_text(encoding="utf-8"))
        required_checks = ("raw_head", "neural_preclamp", "rendered_proxy")
        if (
            report.get("schema") != "dlssnr_forward_validation_v1"
            or report.get("profile") != PROFILE_ID
            or report.get("numerics_profile") != "train_surrogate"
            or report.get("float_validated") is not True
            or report.get("model_sha256") != identity["model_sha256"]
            or report.get("implementation_sha256") != json_sha256(implementation_identity())
            or not report.get("reference_identity")
            or any(report.get("checks", {}).get(name) is not True for name in required_checks)
        ):
            raise ValueError("baseline forward validation does not match the weights, implementation or required checks")
        identity["forward_validation_report"] = file_sha256(report_file)
    return identity


def save_canonical(model, folder, *, source_dir=None, metadata=None):
    from musubi_tuner.dlssnr.convert import canonical_config

    folder = Path(folder).resolve()
    source = Path(source_dir).resolve() if source_dir else None
    if source == folder:
        raise ValueError("refusing to overwrite the source canonical directory")
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous() for name, parameter in model.named_parameters()
    }
    if source:
        with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as handle:
            for block in range(31, 39):
                key = f"blocks.{block}.opaque.layer3"
                if key in handle.keys():
                    tensors[key] = handle.get_tensor(key)
        for name in ("model_config.json", "source_manifest.json", "conversion_report.json", "opaque_records.safetensors"):
            if (source / name).is_file():
                shutil.copyfile(source / name, folder / name)
    else:
        write_json(folder / "model_config.json", canonical_config())
    save_tensors(folder / "model.safetensors", tensors)
    write_json(
        folder / "numerics.json",
        {
            "schema": "dlssnr_numerics_v1",
            "profiles": {"train_surrogate": SURROGATE_FLAGS},
            "implementation": implementation_identity(),
        },
    )
    write_json(
        folder / "preprocessing.json",
        {
            "schema": "dlssnr_preprocessing_v1",
            "controls_encoding": "dlssnr_lanes_10_14_v1",
            "source_encoding": "srgb_proxy",
            "implementation": implementation_identity(),
        },
    )
    report = dict(metadata or {})
    report.update(
        schema="dlssnr_training_artifact_v1",
        profile=PROFILE_ID,
        model_sha256=file_sha256(folder / "model.safetensors"),
        float_validated=False,
        temporal_validated=False,
        native_export_validated=False,
        opaque_available=(folder / "opaque_records.safetensors").is_file(),
    )
    write_json(folder / "training_metadata.json", report)

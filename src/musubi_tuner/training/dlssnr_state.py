"""Checked optimizer-boundary checkpoints for both full and adapter training."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from musubi_tuner.dlssnr.artifacts import write_json
from musubi_tuner.dlssnr.identity import file_sha256
from musubi_tuner.training.dlssnr_services import capture_rng, restore_rng


def save_state(folder, optimizer, identity, update, cursor):
    folder = Path(folder)
    payload = {
        "schema": "dlssnr_train_state_v2",
        "identity": identity,
        "global_update": update,
        "consumed_samples": cursor,
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(),
    }
    with tempfile.NamedTemporaryFile(dir=folder, delete=False, suffix=".pt") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, folder / "trainer_state.pt")
    finally:
        temporary.unlink(missing_ok=True)
    files = {
        path.name: file_sha256(path) for path in folder.iterdir() if path.is_file() and path.name != "checkpoint_manifest.json"
    }
    write_json(folder / "checkpoint_manifest.json", {"schema": "dlssnr_checkpoint_v2", "files": files})


def read_state(folder, identity):
    folder = Path(folder)
    manifest_file = folder / "checkpoint_manifest.json"
    if not manifest_file.is_file():
        raise ValueError("not a complete v2 training state; old weights can only be used as a warm start")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if manifest.get("schema") != "dlssnr_checkpoint_v2" or "trainer_state.pt" not in files:
        raise ValueError("invalid training state manifest")
    if not ({"adapter.safetensors", "model.safetensors"} & files.keys()):
        raise ValueError("training state contains no model/adapter")
    for name, digest in files.items():
        if Path(name).name != name or ":" in name or not (folder / name).is_file() or file_sha256(folder / name) != digest:
            raise ValueError(f"incomplete or modified training state file: {name}")
    payload = torch.load(folder / "trainer_state.pt", map_location="cpu", weights_only=True)
    if payload.get("schema") != "dlssnr_train_state_v2" or payload.get("identity") != identity:
        raise ValueError("resume identity mismatch: effective config, data, base, implementation or runtime changed")
    for name in ("global_update", "consumed_samples"):
        if type(payload.get(name)) is not int or payload[name] < 0:
            raise ValueError(f"invalid resume counter {name}")
    return payload


def restore_state(payload, optimizer):
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])

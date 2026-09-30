"""Surrogate inference. Outputs are proxy-space RGB, before any DLL color grade."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from musubi_tuner.dlssnr.dataset import load_single_frame_manifest, load_temporal_manifest
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.temporal import stable_frame_seed


def require_surrogate(numerics_profile: str) -> None:
    if numerics_profile != "train_surrogate":
        raise ValueError(f"only train_surrogate inference is implemented, got {numerics_profile}")


def load_model(model_dir: str | Path, device: str = "auto") -> NRModel:
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    inspect_canonical(model_dir, development_smoke=True)
    model = NRModel().to(dtype=torch.float32)
    model.load_canonical(str(Path(model_dir) / "model.safetensors"))
    model.to(device).eval()
    return model


def save_png(image: torch.Tensor, path: Path) -> None:
    array = image.detach().clamp(0, 1).mul(255).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@fp32_execution()
def generate_stills(
    model, manifest: str | Path, bucket_width: int, bucket_height: int, output_dir: str | Path, seed: int = 0
) -> list[Path]:
    samples = load_single_frame_manifest(manifest, bucket_width, bucket_height, require_target=False)
    output = Path(output_dir)
    written = []
    device = next(model.parameters()).device
    for sample in samples:
        frame_seed = stable_frame_seed(seed, 0, sample["sample_id"], sample["frame_index"], sample["crop_id"])
        with torch.no_grad():
            rendered = forward_frame(
                model, sample["source"].unsqueeze(0).to(device), sample["controls"].unsqueeze(0).to(device), frame_seed
            )["rendered_proxy"][0]
        path = output / f"{sample['sample_id']}.png"
        save_png(rendered, path)
        written.append(path)
    return written


@fp32_execution()
def generate_sequence(
    model, manifest: str | Path, bucket_width: int, bucket_height: int, output_dir: str | Path, seed: int = 0
) -> list[Path]:
    clips = load_temporal_manifest(manifest, bucket_width, bucket_height, None, require_target=False)
    if len(clips) != 1:
        raise ValueError("video inference takes one clip; split the manifest")
    clip = clips.rows[0]
    device = next(model.parameters()).device
    output = Path(output_dir) / clip["sample_id"]
    written = []
    history = None
    with torch.no_grad():
        for frame in clips.iter_frames(0):
            if frame["reset"]:
                history = None
            tensors = {name: value.unsqueeze(0).to(device) for name, value in frame.items() if isinstance(value, torch.Tensor)}
            result = forward_frame(
                model,
                tensors["source"],
                tensors["controls"],
                stable_frame_seed(seed, 0, clip["sample_id"], frame["frame_index"], clip.get("crop_id", 0)),
                history=history,
                motion=tensors["motion"] if history is not None else None,
                history_valid=tensors["history_valid"] if history is not None else None,
            )
            path = output / f"frame_{frame['frame_index']:04d}.png"
            save_png(result["rendered_proxy"][0], path)
            written.append(path)
            history = result["next_history"].detach()
    return written

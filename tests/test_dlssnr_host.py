import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytest
from PIL import Image

from musubi_tuner.dataset.architectures import ARCHITECTURE_WAN
from musubi_tuner.dlssnr.infer import generate_sequence, generate_stills, require_surrogate
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.networks.lora_dlssnr import inject, merge_adapter
from musubi_tuner.training.dlssnr_trainer import load_train_config, train_from_config


class TinyHistory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        block = nn.Module()
        block.blend_scale = nn.Parameter(torch.tensor([0.5]))
        self.blocks = nn.ModuleDict({"70": block})

    def forward(self, features, geometry):
        height, width = geometry.valid_height, geometry.valid_width
        output = features.new_zeros(features.shape[0], 4, features.shape[-2], features.shape[-1])
        output[:, 0:3, :height, :width] = features[:, 4:7, :height, :width] * self.gain
        output[:, 3, :height, :width] = 0
        return output


def test_old_architecture_ids_are_unchanged():
    assert ARCHITECTURE_WAN == "wan"
    root = Path(__file__).resolve().parents[1]
    text = (root / "hv_train.py").read_text(encoding="utf-8")
    assert text.startswith("from musubi_tuner.hv_train import main\n")


def test_unsupported_runtime_switches_are_rejected(tmp_path: Path, monkeypatch):
    config = _single_config(tmp_path, tmp_path / "out")
    text = config.read_text(encoding="utf-8").replace("gradient_checkpointing = false", "gradient_checkpointing = true")
    config.write_text(text, encoding="utf-8")
    try:
        load_train_config(config)
    except ValueError as exc:
        assert "checkpoint" in str(exc)
    else:
        raise AssertionError("checkpointing was accepted")
    monkeypatch.setenv("WORLD_SIZE", "2")
    try:
        train_from_config(config)
    except RuntimeError as exc:
        assert "multi-GPU" in str(exc)
    else:
        raise AssertionError("multi-GPU was accepted")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    try:
        train_from_config(config)
    except RuntimeError as exc:
        assert "mixed precision" in str(exc)
    else:
        raise AssertionError("bf16 Accelerate launch was accepted")
    try:
        require_surrogate("native_reference")
    except ValueError as exc:
        assert "train_surrogate" in str(exc)
    else:
        raise AssertionError("native inference was accepted")


def test_inference_writes_proxy_pngs_without_a_target(tmp_path: Path):
    _write_frame(tmp_path)
    manifest = tmp_path / "still.jsonl"
    row = json.loads(_still_row(False))
    row["sample_id"] = "Still Frame"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    stills = generate_stills(TinyHistory(), manifest, 48, 48, tmp_path / "preview", seed=3)
    assert stills[0].is_file()
    assert stills[0].name == "Still Frame.png"
    sequence = tmp_path / "clip.jsonl"
    sequence.write_text(_clip_row(tmp_path) + "\n", encoding="utf-8")
    frames = generate_sequence(TinyHistory(), sequence, 48, 48, tmp_path / "video", seed=3)
    assert len(frames) == 2
    assert all(path.is_file() for path in frames)


def test_inference_rejects_case_collisions_before_writing_any_outputs(tmp_path):
    _write_frame(tmp_path)
    first = json.loads(_still_row(False))
    second = {**first, "sample_id": "STILL"}
    manifest = tmp_path / "collision.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in (first, second)), encoding="utf-8")
    output = tmp_path / "preview"
    with pytest.raises(ValueError, match="sample_id"):
        generate_stills(TinyHistory(), manifest, 48, 48, output)
    assert not output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_inference_moves_inputs_to_the_model_device(tmp_path):
    from test_dlssnr_training import SmallNR

    _write_frame(tmp_path)
    manifest = tmp_path / "still.jsonl"
    manifest.write_text(_still_row(False), encoding="utf-8")
    written = generate_stills(SmallNR().cuda(), manifest, 48, 48, tmp_path / "out")
    assert written[0].is_file()


def test_merge_adds_the_adapter_and_leaves_other_weights(tmp_path: Path):
    torch.manual_seed(0)
    model = NRModel()
    network = inject(model, {"profile": "vit_only", "rank": 16, "alpha": 16, "dropout": 0.0})
    adapter = network.adapters[0]
    with torch.no_grad():
        adapter.lora_down.fill_(0.01)
        adapter.lora_up.fill_(0.02)
    original = model.state_dict()[adapter.target].detach().clone()
    untouched = model.blocks["0"].ffn.fc1.weight.detach().clone()
    merged = merge_adapter({key: value.detach() for key, value in model.state_dict().items()}, network)
    assert torch.allclose(merged[adapter.target], original + adapter.delta_weight())
    assert torch.equal(merged["blocks.0.ffn.fc1.weight"], untouched)


def test_resume_matches_an_uninterrupted_run(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    _write_frame(data)
    (data / "train.jsonl").write_text(_still_row(True) + "\n", encoding="utf-8")
    out = tmp_path / "run"
    config = _single_config(data, out)
    train_from_config(config, max_steps=2)
    out = out / "run"
    first = out / "final" / "model.safetensors"
    saved = tmp_path / "first.safetensors"
    shutil.copyfile(first, saved)
    train_from_config(config, max_steps=2, resume=out / "state-step000001")
    from safetensors.torch import load_file

    left = load_file(saved)
    right = load_file(first)
    assert left.keys() == right.keys()
    worst = max((left[key] - right[key]).abs().max().item() for key in left)
    assert worst < 1e-6, worst
    changed = config.read_text(encoding="utf-8").replace("learning_rate = 1e-3", "learning_rate = 2e-3")
    other = tmp_path / "other.toml"
    other.write_text(changed, encoding="utf-8")
    try:
        train_from_config(other, max_steps=2, resume=out / "state-step000001")
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("a changed config was resumed")


def _write_frame(directory: Path) -> None:
    Image.fromarray(np.full((48, 48, 3), 120, dtype=np.uint8)).save(directory / "frame.png")
    Image.fromarray(np.full((48, 48, 3), 40, dtype=np.uint8)).save(directory / "target.png")
    np.save(directory / "controls.npy", np.zeros((5, 48, 48), np.float32))
    np.save(directory / "motion.npy", np.zeros((2, 48, 48), np.float32))
    np.save(directory / "valid.npy", np.ones((1, 48, 48), np.float32))


def _still_row(with_target: bool) -> str:
    frame = {"frame_index": 0, "input_path": "frame.png", "controls_path": "controls.npy", "reset": True}
    if with_target:
        frame["target_path"] = "target.png"
    return json.dumps(
        {
            "schema": "dlssnr_pairs_v1",
            "sample_id": "still",
            "sequence_id": "scene",
            "source_encoding": "srgb_proxy",
            "target_encoding": "srgb_proxy",
            "controls_encoding": "dlssnr_lanes_10_14_v1",
            "frames": [frame],
        }
    )


def _clip_row(directory: Path) -> str:
    first = {"frame_index": 0, "input_path": "frame.png", "controls_path": "controls.npy", "reset": True}
    second = {
        "frame_index": 1,
        "input_path": "frame.png",
        "controls_path": "controls.npy",
        "motion_path": "motion.npy",
        "history_valid_path": "valid.npy",
        "reset": False,
    }
    return json.dumps(
        {
            "schema": "dlssnr_pairs_v1",
            "sample_id": "clip",
            "sequence_id": "scene",
            "source_encoding": "srgb_proxy",
            "target_encoding": "srgb_proxy",
            "controls_encoding": "dlssnr_lanes_10_14_v1",
            "motion_layout": "chw",
            "frames": [first, second],
        }
    )


def _single_config(data: Path, output: Path) -> Path:
    path = data / "train.toml"
    path.write_text(
        f"""
schema_version = 1
[data]
train_manifest = "{(data / "train.jsonl").as_posix()}"
source_encoding = "srgb_proxy"
target_encoding = "srgb_proxy"
controls_encoding = "dlssnr_lanes_10_14_v1"
bucket_size = [48, 48]
require_cache = false
[training]
mode = "single_frame"
development_smoke = true
seed = 4
batch_size = 1
sequence_length = 1
burn_in = 0
tbptt_length = 1
gradient_accumulation_steps = 1
max_train_steps = 2
gradient_checkpointing = false
[optimizer]
type = "AdamW"
learning_rate = 1e-3
weight_decay = 0.0
lr_scheduler = "constant"
[precision]
mixed_precision = "no"
master_dtype = "float32"
[loss]
pre = 1.0
out = 1.0
edge = 0.0
temporal = 0.0
[output]
output_dir = "{output.as_posix()}"
output_name = "run"
save_every_n_steps = 1
save_state = true
""".strip(),
        encoding="utf-8",
    )
    return path

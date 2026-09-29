import json

import numpy as np
import pytest
import toml
import torch
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.dlssnr.model import ChannelLinear, NRModel
from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA
from musubi_tuner.training import dlssnr_trainer as trainer


class SmallNR(torch.nn.Module):
    """Replace expensive NR math, not the real optimizer, state, loader or runner."""

    def __init__(self):
        super().__init__()
        first, last, head = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
        first.input_adapter = ChannelLinear(32, 16)
        head.rgb, head.logit = ChannelLinear(3, 32), ChannelLinear(1, 32)
        last.head = head
        last.blend_scale = torch.nn.Parameter(torch.tensor([0.5]))
        self.blocks = torch.nn.ModuleDict({"0": first, "70": last})
        for name, parameter in self.named_parameters():
            if name.endswith("weight"):
                torch.nn.init.normal_(parameter, std=0.05)
        self.enforce_lane15()

    enforce_lane15 = NRModel.enforce_lane15
    freeze_single_frame = NRModel.freeze_single_frame
    load_canonical = NRModel.load_canonical

    def forward(self, features, geometry):
        features = self.blocks["0"].input_adapter(features)
        return torch.cat((self.blocks["70"].head.rgb(features), self.blocks["70"].head.logit(features)), dim=1)


def small_inject(model, table):
    network = DLSSNRLoRA()
    network.profile = "manual"
    target = "blocks.70.head.rgb.weight"
    network.add(target, model.blocks["70"].head.rgb, 2, 2, table.get("dropout", 0.0))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    network.report = {"profile": "manual", "targets": [{"name": target, "rank": 2, "alpha": 2, "in": 32, "out": 3}]}
    return network


@pytest.fixture
def small_math(monkeypatch):
    from accelerate.state import AcceleratorState
    from musubi_tuner.networks import lora_dlssnr

    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setattr(trainer, "NRModel", SmallNR)
    monkeypatch.setattr(lora_dlssnr, "inject", small_inject)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def make_config(tmp_path, *, lora=False, accum=2, batch=1, mode="single_frame", evaluate=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((48, 48, 3), 140, np.uint8)).save(tmp_path / "source.png")
    Image.fromarray(np.full((48, 48, 3), 60, np.uint8)).save(tmp_path / "target.png")
    np.save(tmp_path / "controls.npy", np.ones((5, 48, 48), np.float32))
    np.save(tmp_path / "motion.npy", np.zeros((2, 48, 48), np.float32))
    np.save(tmp_path / "valid.npy", np.ones((1, 48, 48), np.float32))
    rows = []
    length = 1 if mode == "single_frame" else 3
    for sample in range(2):
        loss_mask = np.ones((1, 48, 48), np.float32)
        if sample:
            loss_mask[:, :32] = 0
        np.save(tmp_path / f"loss{sample}.npy", loss_mask)
        frames = []
        for index in range(length):
            frames.append(
                {
                    "frame_index": index + 10,
                    "input_path": "source.png",
                    "target_path": "target.png",
                    "controls_path": "controls.npy",
                    "reset": index == 0,
                    "motion_path": "motion.npy",
                    "history_valid_path": "valid.npy",
                    "temporal_valid_path": "valid.npy",
                    "loss_mask_path": f"loss{sample}.npy",
                }
            )
        rows.append(
            {
                "schema": "dlssnr_pairs_v1",
                "sample_id": f"sample{sample}",
                "sequence_id": "train",
                "source_encoding": "srgb_proxy",
                "target_encoding": "srgb_proxy",
                "controls_encoding": "dlssnr_lanes_10_14_v1",
                "motion_layout": "chw",
                "frames": frames,
            }
        )
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config = {
        "schema_version": 1,
        "data": {
            "train_manifest": "data.jsonl",
            "source_encoding": "srgb_proxy",
            "target_encoding": "srgb_proxy",
            "controls_encoding": "dlssnr_lanes_10_14_v1",
            "bucket_size": [48, 48],
        },
        "training": {
            "mode": mode,
            "seed": 4,
            "device": "cpu",
            "development_smoke": True,
            "batch_size": batch,
            "sequence_length": length,
            "burn_in": 0 if length == 1 else 1,
            "tbptt_length": 1 if length == 1 else 2,
            "gradient_accumulation_steps": accum,
            "max_train_steps": 2,
        },
        "optimizer": {"learning_rate": 1e-3},
        "loss": {"edge": 0.0, "temporal": 0.0 if length == 1 else 0.1},
        "output": {"output_dir": "output", "save_every_n_steps": 1, "save_state": True},
    }
    if lora:
        config["lora"] = {"profile": "vit_only", "rank": 16, "alpha": 16, "dropout": 0.2}
    if evaluate:
        rows[0]["sequence_id"] = "validation"
        (tmp_path / "validation.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
        config["data"]["validation_manifest"] = "validation.jsonl"
        config["evaluation"] = {"sample_every_n_steps": 1, "compare_baseline": True}
    path = tmp_path / "train.toml"
    path.write_text(toml.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("lora", [False, True])
def test_output_names_isolate_weights_state_logs_and_evaluation(tmp_path, small_math, lora):
    path = make_config(tmp_path, lora=lora, evaluate=True)
    config = toml.load(path)
    config["output"]["output_name"] = "experiment_a"
    path.write_text(toml.dumps(config), encoding="utf-8")
    train = trainer.train_lora_from_config if lora else trainer.train_from_config
    train(path)
    output = tmp_path / "output"
    first_files = {file: file.read_bytes() for file in output.rglob("*") if file.is_file()}
    assert first_files
    config["output"]["output_name"] = "experiment_b"
    config["training"]["seed"] = 5
    path.write_text(toml.dumps(config), encoding="utf-8")
    train(path)
    for file, contents in first_files.items():
        assert file.read_bytes() == contents, f"the second run overwrote {file.relative_to(output)}"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    for name in ("experiment_a", "experiment_b"):
        run = output / name
        for relative in (
            f"final/{filename}",
            f"state-step000001/{filename}",
            "state-step000001/trainer_state.pt",
            "metrics.jsonl",
            "run_config.json",
            "evaluation/step000002.json",
        ):
            assert (run / relative).is_file()
        assert json.loads((run / "run_config.json").read_text())["config"]["output"]["output_name"] == name


def test_new_training_rejects_an_existing_run_directory(tmp_path, small_math):
    path = make_config(tmp_path)
    run = tmp_path / "output" / "dlssnr"
    run.mkdir(parents=True)
    previous = run / "existing.safetensors"
    previous.write_bytes(b"previous run must remain untouched")
    with pytest.raises(FileExistsError, match="output_name|resume"):
        trainer.train_from_config(path)
    assert previous.read_bytes() == b"previous run must remain untouched"
    assert list(run.iterdir()) == [previous]


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_full_and_lora_resume_with_accumulation_and_comments(tmp_path, small_math, lora, mode):
    config = make_config(tmp_path, lora=lora, mode=mode)
    train = trainer.train_lora_from_config if lora else trainer.train_from_config
    train(config)
    output = tmp_path / "output" / "dlssnr"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = load_file(output / "final" / filename)
    saved = output / "state-step000001"
    assert (saved / "trainer_state.pt").is_file()
    assert (output / "final" / "training_metadata.json").is_file()
    config.write_text(config.read_text() + "\n# comments do not change a run\n", encoding="utf-8")
    train(config, resume=saved)
    actual = load_file(output / "final" / filename)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    state = torch.load(output / "state-step000002" / "trainer_state.pt", weights_only=True)
    assert state["global_update"] == 2
    assert state["consumed_samples"] == 4
    Image.fromarray(np.full((48, 48, 3), 61, np.uint8)).save(tmp_path / "target.png")
    with pytest.raises(ValueError, match="identity|data"):
        train(config, resume=saved)


@pytest.mark.parametrize("lora", [False, True])
def test_accumulation_matches_a_larger_batch(tmp_path, small_math, lora):
    accumulated = make_config(tmp_path / "accum", lora=lora, accum=2, batch=1)
    batched = make_config(tmp_path / "batch", lora=lora, accum=1, batch=2)
    for path in (accumulated, batched):
        config = toml.load(path)
        if lora:
            config["lora"]["dropout"] = 0.0
        path.write_text(toml.dumps(config), encoding="utf-8")
    train = trainer.train_lora_from_config if lora else trainer.train_from_config
    train(accumulated)
    train(batched)
    filename = "adapter.safetensors" if lora else "model.safetensors"
    left = load_file(accumulated.parent / "output" / "dlssnr" / "final" / filename)
    right = load_file(batched.parent / "output" / "dlssnr" / "final" / filename)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=1e-5, atol=1e-7)


def test_evaluation_runs_without_changing_training_rng_or_dropout(tmp_path, small_math):
    plain = make_config(tmp_path / "plain", lora=True)
    evaluated = make_config(tmp_path / "evaluated", lora=True, evaluate=True)
    trainer.train_lora_from_config(plain)
    trainer.train_lora_from_config(evaluated)
    left = load_file(plain.parent / "output/dlssnr/final/adapter.safetensors")
    right = load_file(evaluated.parent / "output/dlssnr/final/adapter.safetensors")
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    report = json.loads((evaluated.parent / "output/dlssnr/evaluation/step000002.json").read_text())
    assert report["baseline"] and report["candidate"]
    assert report["candidate"]["validation"][0]["rgb_mae"] >= 0


def test_train_validation_overlap_is_rejected(tmp_path, small_math):
    path = make_config(tmp_path, evaluate=True)
    manifest = tmp_path / "validation.jsonl"
    row = json.loads(manifest.read_text())
    row["sequence_id"] = "train"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="sequence"):
        trainer.train_from_config(path)


def test_nonfinite_gradients_never_reach_optimizer_or_checkpoint(tmp_path, small_math, monkeypatch):
    def corrupt_model():
        model = SmallNR()
        model.blocks["70"].head.rgb.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        return model

    monkeypatch.setattr(trainer, "NRModel", corrupt_model)
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        trainer.train_from_config(make_config(tmp_path))
    assert not (tmp_path / "output/dlssnr/state-step000001").exists()


def test_save_state_is_written_at_the_final_update_without_a_periodic_save(tmp_path, small_math):
    path = make_config(tmp_path)
    config = toml.load(path)
    config["output"]["save_every_n_steps"] = 0
    path.write_text(toml.dumps(config), encoding="utf-8")
    trainer.train_from_config(path)
    assert (tmp_path / "output/dlssnr/state-step000002/trainer_state.pt").is_file()


def test_disabling_state_still_saves_periodic_adapter_weights(tmp_path, small_math):
    path = make_config(tmp_path, lora=True)
    config = toml.load(path)
    config["output"]["save_state"] = False
    path.write_text(toml.dumps(config), encoding="utf-8")
    trainer.train_lora_from_config(path)
    assert (tmp_path / "output/dlssnr/step000001/adapter.safetensors").is_file()
    assert not list((tmp_path / "output").rglob("trainer_state.pt"))


def test_checkpoint_checksum_rejects_modified_weights(tmp_path, small_math):
    path = make_config(tmp_path, lora=True)
    trainer.train_lora_from_config(path)
    folder = tmp_path / "output/dlssnr/state-step000001"
    from safetensors.torch import save_file

    weights = load_file(folder / "adapter.safetensors")
    weights[next(iter(weights))].add_(1)
    save_file(weights, str(folder / "adapter.safetensors"))
    with pytest.raises(ValueError, match="modified"):
        trainer.train_lora_from_config(path, resume=folder)


def test_fp32_training_disables_tf32_without_leaking_global_settings(tmp_path, small_math, monkeypatch):
    class CheckPrecision(SmallNR):
        def forward(self, features, geometry):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            return super().forward(features, geometry)

    monkeypatch.setattr(trainer, "NRModel", CheckPrecision)
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        trainer.train_from_config(make_config(tmp_path))
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn

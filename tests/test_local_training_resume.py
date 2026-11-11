import json
import random
from types import SimpleNamespace

import numpy as np
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from PIL import Image
import pytest
from safetensors.torch import load_file, save_file
import torch

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.utils.train_utils import LossRecorder


class _TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def prepare_optimizer_params(self, unet_lr):
        return [{"params": [self.weight], "lr": unet_lr}], ["tiny"]

    def prepare_grad_etc(self, transformer):
        pass

    def on_epoch_start(self, transformer):
        pass

    def on_step_start(self):
        pass

    def save_weights(self, path, dtype, metadata):
        save_file({"weight": self.weight.detach().cpu()}, path, metadata=metadata)


class _TinyTrainer(NetworkTrainer):
    architecture = architecture_full_name = "wan"

    def __init__(self):
        super().__init__()
        self.trace = []
        self.sampled = []

    def handle_model_specific_args(self, args):
        args.dit_dtype = None

    def _load_dit_and_swap(self, args, accelerator, dit_weight_dtype):
        return torch.nn.Identity()

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype, train_dataset_group):
        return _TinyNetwork()

    def on_train_start(self, args, accelerator, network, transformer, optimizer):
        self.optimizer = optimizer
        self.seedable_sampler = accelerator.dataloader_config.use_seedable_sampler

    def prepare_sampling(self, args, accelerator, vae_dtype):
        return None, None

    def sample_images(self, accelerator, args, epoch, steps, *unused):
        self.sampled.append(steps)
        random.random()
        np.random.random()
        torch.rand((), device=accelerator.device)

    def scale_shift_latents(self, latents):
        return latents

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise, *rest):
        global_step = rest[-1]
        timestep = self.get_bucketed_timestep()
        target = latents + noise + timestep + random.random() + np.random.random()
        loss = (network.weight * latents - target).square().mean()
        self.trace.append(
            (int(latents.item()), global_step, noise.item(), timestep, self.optimizer.param_groups[0]["lr"], loss.item())
        )
        return loss, {}


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    for state in (AcceleratorState, GradientState, PartialState):
        monkeypatch.setattr(state, "_shared_state", {})
    images = tmp_path / "images"
    images.mkdir()
    for sample in range(1, 6):
        Image.new("RGB", (64, 64)).save(images / f"sample{sample}.png")
        save_file(
            {"latents_1x1x1_float32": torch.full((1, 1, 1, 1), float(sample))}, images / f"sample{sample}_0064x0064_wan.safetensors"
        )
        save_file({"embedding": torch.ones(1)}, images / f"sample{sample}_wan_te.safetensors")
    config = tmp_path / "dataset.toml"
    config.write_text(f'[[datasets]]\nimage_directory = "{images.as_posix()}"\nresolution = [64, 64]\n', encoding="utf-8")
    output = tmp_path / "output"

    def run(name, *options, trainer=None):
        args = setup_parser_common().parse_args(
            [
                "--dit",
                "unused",
                "--dataset_config",
                str(config),
                "--output_dir",
                str(output),
                "--output_name",
                name,
                "--mixed_precision",
                "no",
                "--learning_rate",
                "0.001",
                "--optimizer_type",
                "AdamW",
                "--lr_scheduler",
                "linear",
                "--max_grad_norm",
                "0",
                "--num_timestep_buckets",
                "3",
                "--max_train_steps",
                "8",
                "--gradient_accumulation_steps",
                "2",
                "--max_data_loader_n_workers",
                "0",
                "--sample_prompts",
                "unused",
                "--sample_at_first",
                *options,
            ]
        )
        trainer = trainer or _TinyTrainer()
        trainer.train(args)
        return trainer

    return run, output


@pytest.mark.parametrize(
    ("device", "worker_options", "checkpoint_name", "consumed_batches", "cursor"),
    [
        pytest.param("cpu", (), "base-000001-state", 5, (3, 1, 0), id="cpu-epoch"),
        pytest.param(
            "cpu",
            ("--max_data_loader_n_workers", "2", "--persistent_data_loader_workers"),
            "base-step00000004-state",
            7,
            (4, 1, 2),
            id="cpu-mid-epoch-workers",
        ),
        pytest.param(
            "cuda",
            (),
            "base-000001-state",
            5,
            (3, 1, 0),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
            id="cuda-epoch",
        ),
    ],
)
def test_local_resume_matches_uninterrupted_training(
    tiny_training, monkeypatch, device, worker_options, checkpoint_name, consumed_batches, cursor
):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true" if device == "cpu" else "false")
    run, output = tiny_training
    sampling_options = ("--sample_every_n_epochs", "1")
    base = run(
        "base",
        "--seed",
        "123",
        "--save_state",
        "--save_every_n_steps",
        "4",
        "--save_every_n_epochs",
        "1",
        *sampling_options,
        *worker_options,
    )
    assert base.sampled == [0, 3, 6, 8]
    checkpoint = output / checkpoint_name
    saved = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert (saved["global_step"], saved["epoch"], saved["batch_index"]) == cursor

    glob = image_video_dataset.glob.glob
    monkeypatch.setattr(image_video_dataset, "glob", SimpleNamespace(glob=lambda pattern: list(reversed(glob(pattern)))))
    resumed = run("resumed", "--resume", str(checkpoint), *sampling_options, *worker_options)

    assert resumed.sampled == [6, 8]
    assert resumed.trace == base.trace[consumed_batches:]
    assert len(base.trace) == 14
    torch.testing.assert_close(
        load_file(output / "resumed.safetensors")["weight"], load_file(output / "base.safetensors")["weight"], rtol=0, atol=0
    )


def test_local_resume_rejects_a_changed_training_plan(tiny_training, monkeypatch):
    run, output = tiny_training
    run("base", "--seed", "123", "--save_state", "--save_every_n_steps", "4")
    checkpoint = output / "base-step00000004-state"
    for option, value in (
        ("seed", "456"),
        ("max_train_steps", "7"),
        ("gradient_accumulation_steps", "1"),
        ("lr_scheduler", "cosine"),
        ("timestep_sampling", "uniform"),
    ):
        with pytest.raises(ValueError, match=option):
            run("mismatch", "--resume", str(checkpoint), f"--{option}", value)

    finished = run("finished", "--resume", str(output / "base-state"))
    assert finished.trace == []
    assert finished.sampled == []
    assert not (output / "finished.safetensors").exists()

    load_state = Accelerator.load_state

    def load_state_with_changed_rng(self, directory):
        load_state(self, directory)
        random.random()

    with monkeypatch.context() as patch:
        patch.setattr(Accelerator, "load_state", load_state_with_changed_rng)
        with pytest.raises(ValueError, match="did not restore the RNG"):
            run("changed-rng", "--resume", str(checkpoint))

    rng_path = checkpoint / "random_states_0.pkl"
    rng_path.write_bytes(rng_path.read_bytes() + b"stale")
    with pytest.raises(ValueError, match="RNG state"):
        run("mismatched-rng", "--resume", str(checkpoint))
    rng_path.unlink()
    with pytest.raises(ValueError, match="RNG state"):
        run("missing-rng", "--resume", str(checkpoint))


def test_interrupted_state_save_cannot_resume_from_the_old_cursor(tiny_training, monkeypatch):
    run, output = tiny_training
    run("base", "--seed", "123", "--save_state_on_train_end", "--max_train_steps", "1")

    def interrupted_save(self, directory):
        raise RuntimeError("interrupted save")

    monkeypatch.setattr(Accelerator, "save_state", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted save"):
        run("base", "--seed", "123", "--save_state_on_train_end", "--max_train_steps", "1")

    with pytest.raises(ValueError, match="incomplete"):
        run("resume", "--resume", str(output / "base-state"), "--max_train_steps", "1")


def test_legacy_checkpoint_starts_at_zero_without_changing_sampler(tiny_training, caplog):
    run, output = tiny_training
    run("base", "--seed", "123", "--save_state", "--save_every_n_steps", "4")
    checkpoint = output / "base-step00000004-state"
    saved = (checkpoint / "trainer_state.json").read_bytes()
    (checkpoint / "trainer_state.json").unlink()
    reused_dir = output / "legacy-state"
    reused_dir.mkdir()
    (reused_dir / "trainer_state.json").write_bytes(saved)

    resumed = run("legacy", "--resume", str(checkpoint), "--max_train_steps", "1", "--save_state_on_train_end")

    assert resumed.trace[0][1] == 0
    assert resumed.sampled == [0]
    assert not resumed.seedable_sampler
    assert "additional" in caplog.text
    assert not (reused_dir / "trainer_state.json").exists()


def test_other_backends_and_distributed_or_hub_runs_do_not_use_the_new_sampler():
    args = setup_parser_common().parse_args([])
    trainer = NetworkTrainer()
    trainer._configure_local_resume(args, SimpleNamespace(num_processes=2))
    assert not trainer._deterministic_training

    trainer._configure_local_resume(args, SimpleNamespace(num_processes=1, device=torch.device("mps")))
    assert not trainer._deterministic_training

    args.resume = "owner/repo/state"
    args.resume_from_huggingface = True
    trainer._configure_local_resume(args, SimpleNamespace(num_processes=1))
    assert not trainer._deterministic_training


def test_loss_recorder_keeps_observed_batch_slots_after_mid_epoch_resume():
    losses = LossRecorder()
    losses.add(epoch=1, step=2, loss=6.0)
    losses.add(epoch=1, step=3, loss=10.0)
    assert losses.moving_average == 8.0
    losses.add(epoch=2, step=0, loss=8.0)
    assert losses.moving_average == 8.0
    losses.add(epoch=2, step=2, loss=3.0)
    assert losses.moving_average == 7.0

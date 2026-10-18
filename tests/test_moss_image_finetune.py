import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.moss_image import cache, model as model_module, utils
from musubi_tuner.moss_image.checkpoint import save_full_model
from musubi_tuner.moss_image_train import MossImageTrainer, setup_parser
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, setup_parser as lora_parser


CONFIG = dict(dim=32, ffn_dim=48, num_heads=4, num_layers=3, text_dim=4096, freq_dim=16, latent_channels=128, rope_base=10000.0)


@pytest.fixture
def training_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(model_module, "MOSS_IMAGE_CONFIG", CONFIG)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = model_module.MossImage(**CONFIG)
    checkpoint = tmp_path / "base.safetensors"
    save_full_model(model, checkpoint, torch.float32, {})
    images, caches = tmp_path / "images", tmp_path / "cache"
    images.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32)).save(images / f"sample{i}.png")
        (images / f"sample{i}.txt").write_text(f"caption {i}", encoding="utf-8")
        item = ItemInfo(f"sample{i}", f"caption {i}", (32, 32), (32, 32))
        item.latent_cache_path = str(caches / f"sample{i}_0032x0032_moss.safetensors")
        item.text_encoder_output_cache_path = str(caches / f"sample{i}_moss_te.safetensors")
        cache.save_latents(item, torch.randn(128, 2, 2))
        cache.save_text(item, torch.randn(i + 2, 4096, dtype=torch.bfloat16), torch.randn(2, 4096, dtype=torch.bfloat16))
    config = tmp_path / "dataset.toml"
    config.write_text(
        f'[general]\nresolution = 32\ncaption_extension = ".txt"\nbatch_size = 1\n'
        f'[[datasets]]\nimage_directory = "{images.as_posix()}"\ncache_directory = "{caches.as_posix()}"\n',
        encoding="utf-8",
    )
    yield tmp_path, checkpoint, config
    torch.set_num_threads(old_threads)


def training_args(fixture, name, *, lora=False, extra=()):
    root, checkpoint, config = fixture
    parser = lora_parser() if lora else setup_parser()
    return parser.parse_args(
        [
            "--dit",
            str(checkpoint),
            "--dataset_config",
            str(config),
            "--output_dir",
            str(root / name),
            "--output_name",
            "moss",
            "--mixed_precision",
            "no",
            "--sdpa",
            "--seed",
            "123",
            "--optimizer_type",
            "AdamW",
            "--learning_rate",
            "0.0001",
            "--max_train_steps",
            "4",
            "--lr_scheduler",
            "constant",
            "--max_data_loader_n_workers",
            "0",
            "--save_every_n_steps",
            "2",
            "--save_state",
            "--caption_dropout_rate",
            "0.3",
            *extra,
        ]
    )


def test_full_export_preserves_rope_and_loads_without_external_config(tmp_path):
    torch.manual_seed(10)
    model = model_module.MossImage(**CONFIG).bfloat16()
    path = tmp_path / "full.safetensors"
    save_full_model(model, path, torch.bfloat16, {})
    state = load_file(path)
    assert state["rope.bands"].dtype == torch.float32
    assert state["video_embedder.weight"].dtype == torch.bfloat16
    loaded = utils.load_moss_image_model("cpu", str(path), "torch", False, "cpu", torch.bfloat16)
    assert loaded.num_layers == 3
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_full_training_updates_every_module_group(training_fixture):
    args = training_args(training_fixture, "full")
    before = load_file(training_fixture[1])
    trainer = MossImageTrainer()
    trainer.train(args)
    path = Path(args.output_dir) / "moss.safetensors"
    after = load_file(path)
    for key in [
        "layers.0.proj.q.weight",
        "text_embedder.proj.weight",
        "text_embedder.blocks.0.qkv.weight",
        "time_embedding.0.weight",
        "shared_block_modulation.weight",
        "final_linear_video.weight",
    ]:
        assert not torch.equal(before[key], after[key]), key
        assert torch.isfinite(after[key]).all()
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    assert metadata["ss_steps"] == "4"
    assert metadata["modelspec.architecture"] == "moss-image-base"
    assert metadata["ss_training_type"] == "full-finetune"


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("preserve_distribution", [False, True])
def test_interrupted_resume_matches_uninterrupted(training_fixture, lora, accumulation, preserve_distribution):
    trainer_class = MossImageNetworkTrainer if lora else MossImageTrainer
    extra = (
        ["--gradient_accumulation_steps", str(accumulation), "--network_dim", "2"]
        if lora
        else ["--gradient_accumulation_steps", str(accumulation)]
    )
    if preserve_distribution:
        extra += ["--preserve_distribution_shape", "--num_timestep_buckets", "7", "--min_timestep", "100", "--max_timestep", "900"]
    direct = training_args(training_fixture, "direct", lora=lora, extra=extra)
    trainer_class().train(direct)
    state = Path(direct.output_dir) / "moss-step00000002-state"
    assert (state / "moss_progress.json").is_file()
    resumed = training_args(training_fixture, "resumed", lora=lora, extra=[*extra, "--resume", str(state)])
    trainer_class().train(resumed)
    expected = load_file(Path(direct.output_dir) / "moss.safetensors")
    actual = load_file(Path(resumed.output_dir) / "moss.safetensors")
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0, msg=key)
    progress = json.loads((Path(resumed.output_dir) / "moss-state" / "moss_progress.json").read_text())
    assert progress["global_step"] == 4


@pytest.mark.parametrize("fail_sampling", [False, True])
def test_training_samples_disable_adapter_dropout_and_restore_mode(training_fixture, monkeypatch, fail_sampling):
    args = training_args(training_fixture, "sample-dropout", lora=True, extra=["--sample_at_first", "--network_dropout", "0.5"])
    trainer = MossImageNetworkTrainer()
    adapter = None
    build_network = trainer._build_network

    def capture_network(*args, **kwargs):
        nonlocal adapter
        adapter = build_network(*args, **kwargs)
        return adapter

    def sample(*args, **kwargs):
        assert not adapter.training
        assert all(not layer.training for layer in adapter.unet_loras)
        if fail_sampling:
            raise RuntimeError("sample interrupted")

    monkeypatch.setattr(trainer, "_build_network", capture_network)
    monkeypatch.setattr(trainer, "prepare_sampling", lambda *args: ([{}], None))
    monkeypatch.setattr(trainer, "sample_images", sample)
    if fail_sampling:
        with pytest.raises(RuntimeError, match="sample interrupted"):
            trainer.train(args)
    else:
        trainer.train(args)
    assert adapter.training
    assert all(layer.training for layer in adapter.unet_loras)


@pytest.mark.parametrize(
    "flags",
    [
        ["--fp8_base"],
        ["--block_swap_h2d_only"],
        ["--network_dim", "4"],
        ["--network_weights", "adapter.safetensors"],
        ["--fused_backward_pass", "--optimizer_type", "AdamW"],
        ["--fused_backward_pass", "--optimizer_type", "Adafactor", "--gradient_accumulation_steps", "2"],
    ],
)
def test_full_training_rejects_unsafe_or_lora_only_options(flags):
    args = setup_parser().parse_args(["--sdpa", *flags])
    with pytest.raises(ValueError):
        MossImageTrainer().handle_model_specific_args(args)


def test_resume_rejects_changed_accumulation(training_fixture):
    args = training_args(training_fixture, "initial")
    MossImageTrainer().train(args)
    state = Path(args.output_dir) / "moss-step00000002-state"
    resumed = training_args(training_fixture, "wrong", extra=["--resume", str(state), "--gradient_accumulation_steps", "2"])
    with pytest.raises(ValueError, match="accumulation"):
        MossImageTrainer().train(resumed)


@pytest.mark.parametrize("module", ["lora_te_layers_0", "lora_unet_text_embedder_proj"])
def test_lora_training_rejects_weights_the_configured_adapter_would_ignore(training_fixture, module):
    from safetensors.torch import save_file

    path = training_fixture[0] / "unhandled-adapter.safetensors"
    save_file({module + ".lora_down.weight": torch.ones(2, 4096), module + ".lora_up.weight": torch.ones(32, 2)}, str(path))
    args = training_args(training_fixture, "unhandled", lora=True, extra=["--network_dim", "2", "--network_weights", str(path)])
    with pytest.raises(ValueError, match="unhandled|unsupported"):
        MossImageNetworkTrainer().train(args)


def test_fused_adafactor_updates_and_resumes(training_fixture):
    extra = [
        "--optimizer_type",
        "Adafactor",
        "--max_grad_norm",
        "0",
        "--fused_backward_pass",
        "--optimizer_args",
        "relative_step=False",
        "scale_parameter=False",
        "warmup_init=False",
    ]
    direct = training_args(training_fixture, "fused", extra=extra)
    MossImageTrainer().train(direct)
    state = Path(direct.output_dir) / "moss-step00000002-state"
    resumed = training_args(training_fixture, "fused-resume", extra=["--resume", str(state), *extra])
    MossImageTrainer().train(resumed)
    expected = load_file(Path(direct.output_dir) / "moss.safetensors")
    actual = load_file(Path(resumed.output_dir) / "moss.safetensors")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual block swapping")
@pytest.mark.parametrize("optimizer_type", ["AdamW", "Adafactor", "fused_adafactor"])
def test_full_model_block_swap_preserves_optimizer_updates(optimizer_type):
    from transformers.optimization import Adafactor
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
    from musubi_tuner.moss_image.training import _install_fused_adafactor, _move_full_gradients

    torch.manual_seed(234)
    baseline = model_module.MossImage(**CONFIG).cuda().train()
    swapped = model_module.MossImage(**CONFIG).cuda().train()
    swapped.load_state_dict(baseline.state_dict())
    for model in (baseline, swapped):
        model.enable_gradient_checkpointing()
    swapped.enable_block_swap(1, BlockSwapConfig(torch.device("cuda"), supports_backward=True))
    swapped.move_to_device_except_swap_blocks(torch.device("cuda"))
    swapped.prepare_block_swap_before_forward()
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.0001)
        if optimizer_type == "AdamW"
        else Adafactor(model.parameters(), lr=0.0001, relative_step=False, scale_parameter=False, warmup_init=False)
        for model in (baseline, swapped)
    ]
    handles = (
        [handle for optimizer in optimizers for handle in _install_fused_adafactor(optimizer)]
        if optimizer_type == "fused_adafactor"
        else []
    )
    x = torch.randn(1, 128, 2, 2, device="cuda")
    context = torch.randn(1, 3, 4096, device="cuda")
    times = torch.tensor([550.0], device="cuda")
    try:
        for _ in range(2):
            for model, optimizer in zip((baseline, swapped), optimizers):
                loss = model(x, times, context).square().mean()
                loss.backward()
                if model is swapped and optimizer_type != "fused_adafactor":
                    _move_full_gradients(model, optimizer)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        swapped.wait_for_pending_block_moves()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(swapped.state_dict()[key].cpu(), value.cpu(), atol=1e-5, rtol=1e-4, msg=key)
    finally:
        for handle in handles:
            handle.remove()

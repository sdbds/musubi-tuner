import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.flux_2.flux2_models import AutoEncoder, AutoEncoderParams
from musubi_tuner.moss_image import model as model_module, utils
from musubi_tuner.moss_image.checkpoint import save_full_model
from musubi_tuner.moss_image_generate_image import generate_image, setup_parser
from musubi_tuner.networks import lora_moss_image

from test_moss_image_finetune import CONFIG, training_args
from test_moss_image_finetune import training_fixture as training_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "moss_image_cache_latents.py",
        "moss_image_cache_text_encoder_outputs.py",
        "moss_image_train_network.py",
        "moss_image_train.py",
        "moss_image_generate_image.py",
    ],
)
def test_root_entrypoints_help_without_weights_or_network(script):
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True, cwd=ROOT, env=environment, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "--help" in result.stdout


def test_small_vae_checkpoint_is_inferred_and_decoded(tmp_path):
    from musubi_tuner.moss_image.vae import load_vae

    original = AutoEncoder(AutoEncoderParams(ch=32, ch_mult=[1, 1, 1, 1], decoder_ch=32)).eval()
    path = tmp_path / "vae.safetensors"
    save_file(original.state_dict(), path)
    loaded = load_vae(str(path), torch.float32, "cpu").eval()
    x = torch.randn(1, 128, 2, 2)
    torch.testing.assert_close(loaded.decode(x), original.decode(x))


def test_non_moss_vae_downscale_rejected(tmp_path):
    from musubi_tuner.moss_image.vae import load_vae

    model = AutoEncoder(AutoEncoderParams(ch=32, ch_mult=[1, 1], z_channels=32))
    path = tmp_path / "wrong-vae.safetensors"
    save_file(model.state_dict(), path)
    with pytest.raises(ValueError, match="downscale"):
        load_vae(str(path), torch.float32, "cpu")


def test_generation_uses_saved_full_checkpoint_and_native_text_cache(training_fixture):
    from musubi_tuner.moss_image_train import MossImageTrainer

    args = training_args(training_fixture, "fine-tuned", extra=["--max_train_steps", "1"])
    MossImageTrainer().train(args)
    root = training_fixture[0]
    vae = AutoEncoder(AutoEncoderParams(ch=32, ch_mult=[1, 1, 1, 1], decoder_ch=32)).eval()
    vae_path = root / "vae.safetensors"
    save_file(vae.state_dict(), vae_path)
    output = generate_image(
        setup_parser().parse_args(
            [
                "--dit",
                str(Path(args.output_dir) / "moss.safetensors"),
                "--vae",
                str(vae_path),
                "--text_cache",
                str(root / "cache" / "sample0_moss_te.safetensors"),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--width",
                "32",
                "--height",
                "32",
                "--infer_steps",
                "2",
                "--uncond_skip_layers",
                "",
                "--output_dir",
                str(root / "images-out"),
                "--output_name",
                "result",
                "--seed",
                "123",
            ]
        )
    )
    pixels = np.asarray(Image.open(output))
    assert pixels.shape == (32, 32, 3)
    assert pixels.std() > 0
    assert (root / "images-out" / "result.png").is_file()


def test_streaming_lora_merge_matches_live_adapter(tmp_path):
    torch.manual_seed(10)
    model = model_module.MossImage(**CONFIG).eval().requires_grad_(False)
    checkpoint = tmp_path / "base.safetensors"
    save_full_model(model, checkpoint, torch.float32, {})
    network = lora_moss_image.create_arch_network(1.0, 2, 2.0, None, None, model, train_text_embedder="true")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for module in network.unet_loras:
            module.lora_up.weight.normal_(std=0.03)
    weights = network.state_dict()
    merged = utils.load_moss_image_model(
        "cpu",
        str(checkpoint),
        "torch",
        False,
        "cpu",
        torch.float32,
        lora_weights_list=[weights],
        lora_multipliers=[1.0],
    )
    x, context, time = torch.randn(1, 128, 2, 2), torch.randn(1, 3, 4096), torch.tensor([500.0])
    torch.testing.assert_close(model(x, time, context), merged(x, time, context), atol=1e-5, rtol=1e-4)


def test_loader_rejects_unmatched_lora_instead_of_ignoring_it(tmp_path):
    model = model_module.MossImage(**CONFIG)
    checkpoint = tmp_path / "base.safetensors"
    save_full_model(model, checkpoint, torch.float32, {})
    weights = {"lora_te_layers_0_q.lora_down.weight": torch.ones(2, 32)}
    with pytest.raises(ValueError):
        utils.load_moss_image_model("cpu", str(checkpoint), "torch", False, "cpu", torch.float32, lora_weights_list=[weights])


@pytest.mark.parametrize(
    "script,extra",
    [
        ("moss_image_train.py", ["--full_bf16"]),
        ("moss_image_train_network.py", ["--network_dim", "2"]),
    ],
)
def test_training_cli_bfloat16_smoke(training_fixture, script, extra):
    root, checkpoint, dataset = training_fixture
    environment = dict(os.environ, ACCELERATE_USE_CPU="true", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
    output = root / script.removesuffix(".py")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / script),
            "--dit",
            str(checkpoint),
            "--dataset_config",
            str(dataset),
            "--output_dir",
            str(output),
            "--output_name",
            "moss",
            "--mixed_precision",
            "bf16",
            "--sdpa",
            "--max_train_steps",
            "1",
            "--max_data_loader_n_workers",
            "0",
            "--optimizer_type",
            "AdamW",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    assert (output / "moss.safetensors").is_file()
    weights = load_file(output / "moss.safetensors")
    assert all(torch.isfinite(value).all() for value in weights.values())

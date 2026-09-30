import importlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.dlssnr.model import ChannelLinear
from musubi_tuner.networks.lora_dlssnr import base_target_sha256


def test_adapter_identity_covers_non_target_base_parameters():
    model = torch.nn.ModuleDict({"target": ChannelLinear(2, 2), "other": ChannelLinear(2, 2)})
    for parameter in model.parameters():
        torch.nn.init.ones_(parameter)
    before = base_target_sha256(model, ["target.weight"])
    with torch.no_grad():
        model["other"].weight.add_(1)
    assert base_target_sha256(model, ["target.weight"]) != before


def test_lane_constraint_does_not_rewrite_signed_zero_in_a_frozen_lora_base():
    from test_dlssnr_training import SmallNR, small_inject
    from musubi_tuner.training.dlssnr_trainer import clear_lane15_state

    model = SmallNR()
    with torch.no_grad():
        model.blocks["0"].input_adapter.weight[:, 15].fill_(-0.0)
    network = small_inject(model, {})
    before = base_target_sha256(model, network.target_names)
    optimizer = torch.optim.AdamW(network.parameters())
    clear_lane15_state(model, optimizer)
    assert base_target_sha256(model, network.target_names) == before


def test_canonical_save_preserves_metadata_and_opaque_bytes(tmp_path):
    artifacts = importlib.import_module("musubi_tuner.dlssnr.artifacts")
    source = tmp_path / "base"
    source.mkdir()
    model = torch.nn.Linear(2, 2, bias=False)
    save_file(
        {"weight": model.weight.detach(), "blocks.31.opaque.layer3": torch.tensor([7, 128], dtype=torch.uint8)},
        str(source / "model.safetensors"),
    )
    save_file({"block31.layer3.layer": torch.tensor([7, 128], dtype=torch.uint8)}, str(source / "opaque_records.safetensors"))
    (source / "model_config.json").write_text(json.dumps({"schema": "dlssnr_canonical_v1", "profile": "dlss_nr_310_8_0"}))
    (source / "source_manifest.json").write_text('{"origin":"fixture"}')
    before = (source / "opaque_records.safetensors").read_bytes()
    metadata = {"experimental_surrogate": True, "temporal_trained": False}
    output = tmp_path / "final"
    artifacts.save_canonical(model, output, source_dir=source, metadata=metadata)
    assert (output / "opaque_records.safetensors").read_bytes() == before
    assert json.loads((output / "source_manifest.json").read_text()) == {"origin": "fixture"}
    tensors = load_file(output / "model.safetensors")
    assert tensors["blocks.31.opaque.layer3"].tolist() == [7, 128]
    assert json.loads((output / "training_metadata.json").read_text())["experimental_surrogate"] is True
    assert (source / "opaque_records.safetensors").read_bytes() == before


def test_formal_training_rejects_missing_numerical_evidence(tmp_path):
    artifacts = importlib.import_module("musubi_tuner.dlssnr.artifacts")
    save_file({"weight": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model_config.json").write_text(json.dumps({"schema": "dlssnr_canonical_v1", "profile": "dlss_nr_310_8_0"}))
    with pytest.raises(ValueError, match="validation|conversion|provenance"):
        artifacts.inspect_canonical(tmp_path, development_smoke=False)


def test_canonical_loader_rejects_hidden_unknown_keys_and_nonfinite_weights(tmp_path):
    from test_dlssnr_training import SmallNR

    model = SmallNR()
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for extra, corrupt in [(True, False), (False, True)]:
        tensors = {key: value.clone() for key, value in original.items()}
        if extra:
            tensors["unrecognized.opaque.weight"] = torch.zeros(1)
        if corrupt:
            tensors["blocks.70.head.rgb.weight"][0, 0] = float("nan")
        file = tmp_path / "bad.safetensors"
        save_file(tensors, str(file))
        with pytest.raises((ValueError, RuntimeError)):
            model.load_canonical(str(file))


def test_adapter_loader_rejects_mismatched_scaling(tmp_path):
    from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA, load_adapter, save_adapter

    first, second = DLSSNRLoRA(), DLSSNRLoRA()
    first.add("linear.weight", ChannelLinear(4, 4), 2, 2, 0)
    second.add("linear.weight", ChannelLinear(4, 4), 2, 4, 0)
    file = tmp_path / "adapter.safetensors"
    save_adapter(first, file, "base")
    with pytest.raises(ValueError, match="alpha|scal|configuration"):
        load_adapter(second, file, "base")

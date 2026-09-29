from pathlib import Path

import pytest
import toml

from musubi_tuner.training.dlssnr_trainer import load_train_config


ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path, edit=None, *, lora=False):
    name = "dlssnr_lora_vit.toml" if lora else "dlssnr_full_single.toml"
    config = toml.load(ROOT / "configs" / name)
    if edit:
        edit(config)
    path = tmp_path / "train.toml"
    path.write_text(toml.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "table,key,value",
    [
        ("model", "profile", "other"),
        ("model", "numerics_profile", "native_reference"),
        ("model", "fp8_base", True),
        ("optimizer", "learning_raet", 1e-3),
        ("data", "bucket_szie", [512, 512]),
        ("precision", "fp8", True),
        ("loss", "unknown", 1.0),
        ("evaluation", "unknown", True),
        ("output", "unknown", True),
        ("parameter_groups", "unknown", 0.1),
        ("optimizer", "learning_rate", float("nan")),
        ("training", "batch_size", 0),
        ("training", "gradient_accumulation_steps", 0),
        ("training", "max_train_steps", -1),
        ("training", "tbptt_length", 2),
        ("loss", "temporal", 0.1),
    ],
)
def test_rejects_invalid_or_ignored_configuration(tmp_path, table, key, value):
    path = write_config(tmp_path, lambda c: c[table].update({key: value}))
    with pytest.raises(ValueError):
        load_train_config(path)


def test_rejects_unknown_schema(tmp_path):
    path = write_config(tmp_path, lambda c: c.update(schema_version=999))
    with pytest.raises(ValueError, match="schema"):
        load_train_config(path)


def test_pretrained_weights_are_required_unless_smoke_is_explicit(tmp_path):
    path = write_config(tmp_path, lambda c: c["model"].pop("model_dir"))
    with pytest.raises(ValueError, match="model_dir"):
        load_train_config(path)
    config = toml.load(path)
    config["training"]["development_smoke"] = True
    path.write_text(toml.dumps(config), encoding="utf-8")
    assert load_train_config(path)["training"]["development_smoke"]


def test_paths_resolve_from_toml_not_process_directory(tmp_path):
    path = write_config(tmp_path)
    config = load_train_config(path)
    assert Path(config["data"]["train_manifest"]) == (tmp_path / "../data/train_single.jsonl").resolve()
    assert Path(config["model"]["model_dir"]) == (tmp_path / "../models/canonical_dlssnr").resolve()


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside", "nested/run", "nested\\run", "C:run", "CON", "nul.txt", "run.", "run ", "bad\tname"]
)
def test_output_name_must_be_a_portable_directory_name(tmp_path, name):
    path = write_config(tmp_path, lambda c: c["output"].update(output_name=name))
    with pytest.raises(ValueError, match="output_name"):
        load_train_config(path)


def test_lora_rejects_unknown_fields_and_invalid_dropout(tmp_path):
    for key, value in [("droput", 0.1), ("dropout", 1.0), ("rank", 0), ("alpha", float("inf"))]:
        path = write_config(tmp_path, lambda c: c["lora"].update({key: value}), lora=True)
        with pytest.raises(ValueError):
            load_train_config(path, lora=True)

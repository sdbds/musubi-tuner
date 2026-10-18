import json

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file


@pytest.mark.parametrize("architecture", ["qie", "qi21", "mfe"])
@pytest.mark.parametrize("source", ["directory", "jsonl"])
def test_edit_datasets_keep_all_reference_images(tmp_path, architecture, source):
    from musubi_tuner.dataset.image_video_dataset import ImageDataset

    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    targets.mkdir()
    controls.mkdir()
    target = targets / "sample.png"
    Image.new("RGB", (64, 64)).save(target)
    (targets / "sample.txt").write_text("caption", encoding="utf-8")
    references = [controls / f"sample_{index}.png" for index in range(2)]
    for path, value in zip(references, [40, 80]):
        Image.new("RGB", (64, 64), color=(value, 0, 0)).save(path)

    if source == "directory":
        source_args = {"image_directory": str(targets), "control_directory": str(controls)}
    else:
        manifest = tmp_path / "images.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "image_path": str(target),
                    "caption": "caption",
                    "control_path_0": str(references[0]),
                    "control_path_1": str(references[1]),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        source_args = {"image_jsonl_file": str(manifest)}

    dataset = ImageDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        cache_directory=str(tmp_path / "cache"),
        architecture=architecture,
        **source_args,
    )
    items = [item for _, batch in dataset.retrieve_latent_cache_batches(num_workers=1) for item in batch]

    assert len(items) == 1
    assert [int(image[0, 0, 0]) for image in items[0].control_content] == [40, 80]


@pytest.mark.parametrize("cache_dtype", [None, "fp8"])
def test_krea2_text_cache_preserves_precision_metadata(tmp_path, cache_dtype):
    from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_krea2
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

    item = ItemInfo("sample", "caption", (32, 32), (32, 32))
    item.text_encoder_output_cache_path = str(tmp_path / "sample_kr2_te.safetensors")
    dtype = torch.bfloat16 if cache_dtype is None else torch.float8_e4m3fn
    embed = torch.arange(48, dtype=torch.float32).reshape(3, 2, 8).to(dtype)

    if cache_dtype is None:
        save_text_encoder_output_cache_krea2(item, embed)
    else:
        save_text_encoder_output_cache_krea2(item, embed, cache_dtype=cache_dtype)

    with safe_open(item.text_encoder_output_cache_path, framework="pt") as handle:
        assert handle.metadata()["architecture"] == "krea2"
        assert handle.metadata()["text_cache_dtype"] == (cache_dtype or "bf16")
    state = load_file(item.text_encoder_output_cache_path)
    expected_key = "varlen_krea2_vl_embed_bfloat16" if cache_dtype is None else "varlen_krea2_vl_embed_float8_e4m3fn"
    assert set(state) == {expected_key}
    assert state[expected_key].dtype == dtype
    torch.testing.assert_close(state[expected_key].float(), embed.float())

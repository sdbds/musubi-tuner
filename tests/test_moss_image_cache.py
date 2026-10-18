from multiprocessing import Value

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
from musubi_tuner.moss_image import cache


def make_item(tmp_path, name="sample", length=3):
    item = ItemInfo(name, "a caption", (32, 32), (32, 32))
    item.latent_cache_path = str(tmp_path / f"{name}_0032x0032_moss.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / f"{name}_moss_te.safetensors")
    latent = torch.randn(128, 2, 2, dtype=torch.float32)
    text = torch.randn(length, 4096, dtype=torch.bfloat16)
    empty = torch.randn(2, 4096, dtype=torch.bfloat16)
    return item, latent, text, empty


def test_cache_roundtrip_and_actual_batch_collation(tmp_path):
    from musubi_tuner.dataset.bucket import BucketBatchManager

    items = []
    rows = []
    for i, length in enumerate([3, 5]):
        item, latent, text, empty = make_item(tmp_path, f"sample{i}", length)
        cache.save_latents(item, latent)
        cache.save_text(item, text, empty, max_length=4096)
        items.append(item)
        rows.append((latent, text, empty))
    manager = BucketBatchManager({(32, 32): items}, 2, architecture="moss")
    batch = manager[0]
    assert batch["latents"].shape == (2, 128, 2, 2)
    assert batch["latents"].dtype == torch.float32
    assert sorted(row.shape[0] for row in batch["moss_embed"]) == [3, 5]
    actual_texts = {row.shape[0]: row for row in batch["moss_embed"]}
    for _, text, _ in rows:
        assert torch.equal(actual_texts[text.shape[0]].view(torch.int16), text.view(torch.int16))


@pytest.mark.parametrize("shape", [(32, 4, 4), (128, 0, 2), (1, 128, 2, 2)])
def test_raw_or_invalid_latents_rejected(tmp_path, shape):
    item, _, _, _ = make_item(tmp_path)
    with pytest.raises(ValueError):
        cache.save_latents(item, torch.zeros(shape))


def test_nonfinite_cache_is_not_silently_repaired(tmp_path):
    item, latent, text, empty = make_item(tmp_path)
    latent[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        cache.save_latents(item, latent)
    text[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        cache.save_text(item, text, empty, max_length=4096)


def test_skip_existing_validates_architecture_and_caption(tmp_path):
    item, latent, text, empty = make_item(tmp_path)
    assert not cache.cache_is_current(item, "latent")
    cache.save_latents(item, latent)
    cache.save_text(item, text, empty, max_length=4096)
    assert cache.cache_is_current(item, "latent")
    assert cache.cache_is_current(item, "text", max_length=4096)
    item.caption = "a changed caption"
    assert not cache.cache_is_current(item, "text", max_length=4096)
    with safe_open(item.latent_cache_path, framework="pt") as f:
        metadata = f.metadata()
    metadata["architecture"] = "flux_2_dev"
    save_file({"latents_2x2_float32": latent}, item.latent_cache_path, metadata=metadata)
    with pytest.raises(ValueError, match="architecture"):
        cache.cache_is_current(item, "latent")


def test_invalid_text_shape_dtype_and_version_rejected(tmp_path):
    item, latent, text, empty = make_item(tmp_path)
    with pytest.raises(ValueError):
        cache.save_text(item, text[:, :32], empty, max_length=4096)
    cache.save_text(item, text, empty, max_length=4096)
    with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
        metadata = f.metadata()
    metadata["moss_cache_version"] = "999"
    save_file(load_file(item.text_encoder_output_cache_path), item.text_encoder_output_cache_path, metadata=metadata)
    with pytest.raises(ValueError, match="version"):
        cache.load_cache(item.text_encoder_output_cache_path, "text")


def test_dropout_uses_empty_encoding_without_mutating_cache():
    prompts = [torch.ones(3, 4096, dtype=torch.bfloat16), torch.full((5, 4096), 2, dtype=torch.bfloat16)]
    empty = [torch.full((2, 4096), -1, dtype=torch.bfloat16)] * 2
    embeds, mask = cache.prepare_conditioning(prompts, empty, 0.0, torch.device("cpu"))
    assert mask.sum(1).tolist() == [3, 5]
    assert embeds[0, 0, 0] == 1
    embeds, mask = cache.prepare_conditioning(prompts, empty, 1.0, torch.device("cpu"))
    assert mask.sum(1).tolist() == [2, 2]
    assert torch.equal(embeds, torch.stack(empty))
    assert prompts[0].shape[0] == 3 and prompts[0][0, 0] == 1


def test_dataset_discovers_moss_cache_and_checks_metadata(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "sample.png")
    (images / "sample.txt").write_text("a caption", encoding="utf-8")
    item, latent, text, empty = make_item(tmp_path / "cache")
    cache.save_latents(item, latent)
    cache.save_text(item, text, empty, max_length=4096)
    dataset = ImageDataset(
        resolution=(32, 32),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        image_directory=str(images),
        cache_directory=str(tmp_path / "cache"),
        architecture="moss",
    )
    dataset.set_seed(0, Value("i", 1))
    dataset.prepare_for_training()
    assert len(dataset) == 1
    assert dataset[0]["latents"].shape == (1, 128, 2, 2)


def test_latent_entrypoint_uses_normalized_vae_output(tmp_path):
    from musubi_tuner.flux_2.flux2_models import AutoEncoder, AutoEncoderParams
    from musubi_tuner.moss_image_cache_latents import encode_and_save_batch

    ae = AutoEncoder(AutoEncoderParams(ch=32, ch_mult=[1, 1, 1, 1], z_channels=32)).eval()
    item, _, _, _ = make_item(tmp_path)
    item.content = np.full((32, 32, 3), 100, dtype=np.uint8)
    pixels = (torch.from_numpy(item.content).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).contiguous()
    with torch.no_grad():
        expected = ae.encode(pixels)[0]
    encode_and_save_batch(ae, [item])
    actual = next(iter(cache.load_cache(item.latent_cache_path, "latent").values()))
    torch.testing.assert_close(actual, expected)


def test_wrong_latent_bucket_size_rejected(tmp_path):
    item, latent, _, _ = make_item(tmp_path)
    with pytest.raises(ValueError, match="bucket"):
        cache.save_latents(item, torch.zeros(128, 4, 4))
    cache.save_latents(item, latent)
    item.bucket_size = (64, 64)
    assert not cache.cache_is_current(item, "latent")

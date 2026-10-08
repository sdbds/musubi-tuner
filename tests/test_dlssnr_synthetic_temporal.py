"""Subpixel still-to-clip geometry against independent pixel-space interpolation."""

import pytest
import torch

from musubi_tuner.dlssnr.controls import fixed_control_tensor, resolve_fixed_controls
from musubi_tuner.training.dlssnr_services import capture_rng
from test_dlssnr_control_randomization_training import assert_nested_equal


def synthetic_api():
    from musubi_tuner.dlssnr import synthetic_temporal

    return synthetic_temporal


def still_sample():
    y, x = torch.meshgrid(torch.arange(48), torch.arange(64), indexing="ij")
    source = torch.stack((x.float() / 64, y.float() / 48, ((x == 20) & (y == 20)).float()))
    fixed = resolve_fixed_controls({"nr_tone": 0.7, "nr_structure": 0.3, "nr_style": 15})
    return {
        "source": source,
        "target": source * 0.7 + 0.1,
        "controls": fixed_control_tensor(fixed, 64, 48),
        "loss_mask": torch.ones(1, 48, 64),
        "motion": torch.zeros(2, 48, 64),
        "history_valid": torch.zeros(1, 48, 64, dtype=torch.bool),
        "temporal_valid": torch.zeros(1, 48, 64, dtype=torch.bool),
        "reset": True,
        "frame_index": 10,
        "sample_id": "still-a",
        "sequence_id": "original-scene",
        "crop_id": 3,
        "fixed_controls": fixed,
    }


def bilinear_oracle(image, offset, *, border=True):
    height, width = image.shape[-2:]
    x = torch.arange(width, dtype=torch.float64) + float(offset[0])
    y = torch.arange(height, dtype=torch.float64) + float(offset[1])
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = x - x0, y - y0
    result = torch.zeros_like(image, dtype=torch.float64)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            xx, yy = x0 + dx, y0 + dy
            weights = wy[:, None] * wx[None, :]
            if not border:
                weights *= ((xx >= 0) & (xx < width))[None, :] * ((yy >= 0) & (yy < height))[:, None]
            result += image[:, yy.clamp(0, height - 1)[:, None], xx.clamp(0, width - 1)[None, :]].double() * weights
    return result.float()


def test_translation_preserves_first_frame_metadata_and_matches_original_grid():
    sample = still_sample()
    offsets = torch.tensor([[0, 0], [0.5, -0.25], [-0.25, 0.75]])
    clip = synthetic_api().synthesize_clip(sample, offsets)
    assert [frame["frame_index"] for frame in clip["frames"]] == [10, 11, 12]
    assert [frame["reset"] for frame in clip["frames"]] == [True, False, False]
    assert clip["sequence_id"] == "original-scene" and clip["sample_id"] == "still-a"
    assert clip["crop_id"] == 3 and clip["temporal_support"] == "joint_loss_mask"
    for key in ("source", "target", "controls", "loss_mask"):
        torch.testing.assert_close(clip["frames"][0][key], sample[key], rtol=0, atol=0)
    assert not clip["frames"][0]["history_valid"].any() and not clip["frames"][0]["temporal_valid"].any()
    for index, frame in enumerate(clip["frames"]):
        torch.testing.assert_close(frame["source"], bilinear_oracle(sample["source"], offsets[index]), rtol=1e-5, atol=3e-6)
        delta = torch.zeros(2) if index == 0 else offsets[index] - offsets[index - 1]
        torch.testing.assert_close(frame["motion"], delta[:, None, None].expand(2, 48, 64), rtol=0, atol=0)
        torch.testing.assert_close(frame["controls"], sample["controls"], rtol=0, atol=0)
        assert frame["history_valid"].dtype == frame["temporal_valid"].dtype == torch.bool


def test_impulse_frames_resample_the_original_without_accumulating_blur():
    sample = still_sample()
    clip = synthetic_api().synthesize_clip(sample, torch.tensor([[0, 0], [0.5, 0], [0.5, 0]]))
    first, second = clip["frames"][1:]
    assert torch.equal(first["source"], second["source"])
    torch.testing.assert_close(first["source"][2, 20, 19:21], torch.tensor([0.5, 0.5]), rtol=0, atol=3e-6)
    assert not torch.equal(second["source"], bilinear_oracle(first["source"], (0.5, 0)))
    assert not second["motion"].any()


@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1)])
def test_history_rejects_invalid_previous_footprint(axis, sign):
    sample = still_sample()
    offsets = torch.zeros(3, 2)
    offsets[1, axis] = sign * 0.5
    clip = synthetic_api().synthesize_clip(sample, offsets)
    last = clip["frames"][2]
    edge = -1 if sign > 0 else 0
    line = last["history_valid"][..., edge] if axis == 0 else last["history_valid"][..., edge, :]
    assert not line.any()
    assert last["history_valid"][..., 2:-2, 2:-2].all()
    assert torch.equal(last["history_valid"], last["temporal_valid"])


def test_zero_shift_keeps_edge_mask_support():
    sample = still_sample()
    sample["loss_mask"].zero_()
    sample["loss_mask"][..., 0] = 0.25
    synthetic_api().validate_synthetic_support(sample, 0)
    clip = synthetic_api().synthesize_clip(sample, torch.zeros(3, 2))
    for frame in clip["frames"]:
        torch.testing.assert_close(frame["source"], sample["source"], rtol=0, atol=0)
        torch.testing.assert_close(frame["loss_mask"], sample["loss_mask"], rtol=0, atol=0)
    assert clip["frames"][1]["history_valid"].all()
    with pytest.raises(ValueError, match="support"):
        synthetic_api().validate_synthetic_support(sample, 0.5)


def test_soft_masks_resample_labels_without_reading_excluded_values():
    sample = still_sample()
    sample["loss_mask"] *= 0.4
    sample["loss_mask"][..., 10:30, 15:35] = 0
    offsets = torch.tensor([[0, 0], [0.5, 0.25], [-0.25, -0.5]])
    clip = synthetic_api().synthesize_clip(sample, offsets)
    changed = {**sample, "target": sample["target"].masked_fill(sample["loss_mask"].expand_as(sample["target"]) == 0, 500)}
    other = synthetic_api().synthesize_clip(changed, offsets)
    for index, (frame, altered) in enumerate(zip(clip["frames"], other["frames"])):
        assert torch.equal(frame["loss_mask"], altered["loss_mask"])
        torch.testing.assert_close(frame["target"] * frame["loss_mask"], altered["target"] * frame["loss_mask"], rtol=0, atol=0)
        assert torch.isfinite(frame["target"]).all()
        if index:
            coverage = bilinear_oracle(sample["loss_mask"], offsets[index], border=False)
            weighted = bilinear_oracle(sample["target"] * sample["loss_mask"], offsets[index], border=False)
            expected = weighted / torch.where(coverage > 0, coverage, 1)
            valid = frame["loss_mask"] > 0
            torch.testing.assert_close(
                torch.where(valid, frame["target"], 0), torch.where(valid, expected, 0), rtol=1e-5, atol=3e-6
            )
            assert frame["target"][~valid.expand_as(frame["target"])].eq(0).all()
            assert not frame["loss_mask"][..., -1 if offsets[index, 0] > 0 else 0].any()
            assert not frame["loss_mask"][..., -1 if offsets[index, 1] > 0 else 0, :].any()
            interior = (..., slice(1, -1), slice(1, -1))
            torch.testing.assert_close(frame["loss_mask"][interior], coverage[interior], rtol=1e-5, atol=3e-6)


def test_spatial_control_maps_translate_without_using_the_label_mask():
    sample = still_sample()
    del sample["fixed_controls"]
    sample["controls"] = torch.cat((sample["source"], sample["source"][:2]), dim=0)
    sample["loss_mask"][..., 8:20, 8:20] = 0
    clip = synthetic_api().synthesize_clip(sample, torch.tensor([[0, 0], [0.25, -0.5]]))
    torch.testing.assert_close(
        clip["frames"][1]["controls"], bilinear_oracle(sample["controls"], (0.25, -0.5)), rtol=1e-5, atol=3e-6
    )
    torch.testing.assert_close(clip["frames"][1]["source"], bilinear_oracle(sample["source"], (0.25, -0.5)), rtol=1e-5, atol=3e-6)


@pytest.mark.parametrize("maximum", [0, 0.5, 1])
def test_private_offsets_repeat_per_logical_identity_and_epoch(maximum):
    api = synthetic_api()
    before = capture_rng()
    kwargs = {"seed": 4, "epoch": 2, "sample_id": "d0-r0-a", "crop_id": 3}
    offsets = api.sample_offsets(12, maximum, **kwargs)
    assert offsets.shape == (12, 2) and offsets.dtype == torch.float32 and offsets.device.type == "cpu"
    assert offsets[0].eq(0).all() and offsets.abs().max() <= maximum
    assert torch.equal(offsets, api.sample_offsets(12, maximum, **kwargs))
    if maximum:
        for changed in ({"epoch": 3}, {"sample_id": "d0-r1-a"}, {"crop_id": 4}):
            assert not torch.equal(offsets, api.sample_offsets(12, maximum, **{**kwargs, **changed}))
    assert_nested_equal(capture_rng(), before)


@pytest.mark.parametrize("maximum", [-0.1, 1.01, float("nan"), float("inf"), True])
def test_invalid_shift_settings_are_rejected(maximum):
    with pytest.raises(ValueError, match="shift"):
        synthetic_api().sample_offsets(3, maximum, seed=1, epoch=0, sample_id="a", crop_id=0)
    with pytest.raises(ValueError, match="shift"):
        synthetic_api().validate_synthetic_support(still_sample(), maximum)


@pytest.mark.parametrize(
    "offsets",
    [
        torch.ones(3, 2) * 0.2,
        torch.zeros(3, 1),
        torch.empty(0, 2),
        torch.tensor([[0, 0], [float("inf"), 0]]),
        torch.tensor([[0, 0], [2, 0]]),
    ],
)
def test_invalid_offsets_are_rejected(offsets):
    with pytest.raises(ValueError, match="offset|first"):
        synthetic_api().synthesize_clip(still_sample(), offsets)

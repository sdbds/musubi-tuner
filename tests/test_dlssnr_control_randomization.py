"""Effective controls, detached targets and augmented temporal supervision."""

import random

import numpy as np
import pytest
import torch


SETTINGS = {"anchor_probability": 0.25, "corner_probability": 0.25, "residual_sigma": 6.0}


def encode(reference, ratios):
    from musubi_tuner.dlssnr.control_randomization import encode_control_point

    return encode_control_point(reference, ratios)


def sample(settings=SETTINGS, *, epoch=2, sample_id="d0-r0-a", crop_id=0):
    from musubi_tuner.dlssnr.control_randomization import sample_control_point

    return sample_control_point({}, settings, seed=4, epoch=epoch, sample_id=sample_id, crop_id=crop_id)


def test_effective_control_alias_uses_exact_endpoint():
    point = encode({"nr_tone": 1, "nr_structure": 1}, (0.9999, 0.9999))
    assert point["ratios"].tolist() == [1.0, 1.0]
    assert torch.equal(point["sampled_lanes"], point["reference_lanes"])


@pytest.mark.parametrize("name", ["nr_tone", "nr_structure"])
@pytest.mark.parametrize("value", [0, 1e-12])
def test_reference_tone_that_rounds_to_zero_is_rejected(name, value):
    with pytest.raises(ValueError, match="reference"):
        encode({name: value}, (1, 1))


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("skin", [-1, 0.75])
def test_encoding_keeps_style_and_skin_rule_and_uses_effective_ratios(auto, skin):
    point = encode({"nr_style": 128, "nr_auto_mask": auto, "nr_skin": skin, "nr_tone": 0.7, "nr_structure": 0.9}, (0.25, 0.5))
    expected = torch.tensor([0.7 * 0.25, 0.9 * 0.5]).half().float()
    assert point["values"].tolist() == expected.tolist()
    torch.testing.assert_close(point["ratios"], expected / torch.tensor([0.7, 0.9]).half().float(), rtol=0, atol=0)
    expected_lanes = (
        [1, expected[0], 1, expected[1] if skin == -1 else skin, expected[1]] if auto else [1, expected[0], expected[1], -1, -1]
    )
    assert point["sampled_lanes"].tolist() == expected_lanes
    assert all(value.dtype == torch.float32 and value.device.type == "cpu" for value in point.values())


@pytest.mark.parametrize(
    "ratios", [(-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5), (0.5, float("inf")), (True, 0), (0.5,), (0, 0, 0)]
)
def test_invalid_requested_ratios_are_rejected(ratios):
    with pytest.raises(ValueError, match="ratios"):
        encode({}, ratios)


@pytest.mark.parametrize(
    "updates",
    [
        {"anchor_probability": -0.1},
        {"corner_probability": 1.1},
        {"anchor_probability": 0.8},
        {"corner_probability": float("nan")},
        {"anchor_probability": True},
        {"residual_sigma": 0},
        {"residual_sigma": 33},
    ],
)
def test_invalid_sampling_settings_are_rejected(updates):
    with pytest.raises(ValueError):
        sample({**SETTINGS, **updates})


def test_sampling_branches_and_default_endpoint_mass():
    assert sample({**SETTINGS, "anchor_probability": 1, "corner_probability": 0})["ratios"].tolist() == [1, 1]
    corners = {
        tuple(sample({**SETTINGS, "anchor_probability": 0, "corner_probability": 1}, sample_id=str(i))["ratios"].tolist())
        for i in range(100)
    }
    assert corners == {(0, 0), (0, 1), (1, 0), (1, 1)}
    continuous = sample({**SETTINGS, "anchor_probability": 0, "corner_probability": 0})["ratios"]
    assert ((continuous > 0) & (continuous < 1)).all()
    draws = torch.stack([sample(sample_id=str(i))["ratios"] for i in range(1000)])
    assert float((draws == 1).all(dim=1).float().mean()) == pytest.approx(0.3125, abs=0.05)
    assert float((draws == 0).all(dim=1).float().mean()) == pytest.approx(0.0625, abs=0.025)


def test_seeds_vary_by_epoch_repeat_crop_and_domain_without_global_rng():
    from musubi_tuner.dlssnr.temporal import augmentation_seed

    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    settings = {**SETTINGS, "anchor_probability": 0, "corner_probability": 0}
    point = sample(settings)
    assert augmentation_seed(4, 2, "d0-r0-a", 0, domain="controls") != augmentation_seed(
        4, 2, "d0-r0-a", 0, domain="synthetic_jitter"
    )
    for kwargs in ({"epoch": 3}, {"sample_id": "d0-r1-a"}, {"crop_id": 1}):
        assert not torch.equal(point["sampled_lanes"], sample(settings, **kwargs)["sampled_lanes"])
    assert torch.equal(point["sampled_lanes"], sample(settings)["sampled_lanes"])
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0] and after_numpy[2:] == before[1][2:]
    np.testing.assert_array_equal(after_numpy[1], before[1][1])
    assert torch.equal(torch.random.get_rng_state(), before[2])

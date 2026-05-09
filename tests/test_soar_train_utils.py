import argparse
import fnmatch
import unittest
from argparse import Namespace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musubi_tuner.soar_train_utils import (  # noqa: E402
    add_soar_arguments,
    combine_cfg_standard_velocity,
    compute_loss_weighting_from_sigma,
    compute_per_sample_loss,
    default_flow_matching_target,
    get_legacy_soar_empty_prompt_cache_path,
    get_soar_empty_prompt_cache_path,
    is_soar_enabled,
    load_soar_empty_prompt_tensor,
    save_soar_empty_prompt_cache,
    validate_soar_args,
    zimage_flow_matching_target,
)


class TestSoarTrainUtils(unittest.TestCase):
    def test_add_soar_arguments_is_idempotent(self):
        parser = argparse.ArgumentParser()
        add_soar_arguments(parser)
        add_soar_arguments(parser)

        args = parser.parse_args([])
        self.assertFalse(args.soar)
        self.assertEqual(args.soar_lambda_aux, 1.0)
        self.assertEqual(args.soar_trajectory_length, 6)
        self.assertEqual(args.soar_num_sampling_steps, 40)
        self.assertEqual(args.soar_sigma_upper_ratio, 1.5)
        self.assertEqual(args.soar_cfg_scale_sampling, 4.5)

    def test_validate_soar_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=-1.0,
                    soar_trajectory_length=6,
                    soar_num_sampling_steps=40,
                    soar_sigma_upper_ratio=1.5,
                )
            )
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=1.0,
                    soar_trajectory_length=0,
                    soar_num_sampling_steps=40,
                    soar_sigma_upper_ratio=1.5,
                )
            )
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=1.0,
                    soar_trajectory_length=6,
                    soar_num_sampling_steps=1,
                    soar_sigma_upper_ratio=1.5,
                )
            )
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=1.0,
                    soar_trajectory_length=6,
                    soar_num_sampling_steps=40,
                    soar_sigma_upper_ratio=0.9,
                    soar_cfg_scale_sampling=1.0,
                )
            )
        with self.assertRaises(ValueError):
            validate_soar_args(
                Namespace(
                    soar=True,
                    soar_lambda_aux=1.0,
                    soar_trajectory_length=6,
                    soar_num_sampling_steps=40,
                    soar_sigma_upper_ratio=1.5,
                    soar_cfg_scale_sampling=0.0,
                )
            )

    def test_is_soar_enabled_respects_zero_lambda(self):
        self.assertFalse(is_soar_enabled(Namespace(soar=True, soar_lambda_aux=0.0, soar_trajectory_length=6)))
        self.assertTrue(is_soar_enabled(Namespace(soar=True, soar_lambda_aux=1.0, soar_trajectory_length=6)))

    def test_loss_weighting_keeps_target_rank(self):
        loss = torch.zeros(2, 4, 8, 8)
        sigmas = torch.tensor([0.25, 0.5], dtype=torch.float32).view(2, 1, 1, 1, 1)

        weighting = compute_loss_weighting_from_sigma("sigma_sqrt", sigmas, loss.ndim)

        self.assertEqual(weighting.shape, torch.Size([2, 1, 1, 1]))
        self.assertEqual((loss * weighting).shape, loss.shape)

    def test_per_sample_loss_rejects_batch_broadcast(self):
        model_pred = torch.zeros(2, 1, 2, 2)
        target = torch.ones_like(model_pred)
        bad_weighting = torch.ones(2, 2, 1, 1)

        with self.assertRaises(ValueError):
            compute_per_sample_loss(model_pred, target, bad_weighting)

    def test_target_conventions_are_opposite(self):
        clean = torch.full((1, 1, 1, 1), 2.0)
        aux = torch.full_like(clean, 5.0)
        sigma = torch.full_like(clean, 0.5)

        self.assertTrue(torch.equal(default_flow_matching_target(clean, aux, sigma), torch.full_like(clean, 6.0)))
        self.assertTrue(torch.equal(zimage_flow_matching_target(clean, aux, sigma), torch.full_like(clean, -6.0)))

    def test_cfg_combination_uses_standard_formula(self):
        uncond = torch.tensor([1.0, 2.0])
        cond = torch.tensor([3.0, 5.0])

        self.assertTrue(torch.equal(combine_cfg_standard_velocity(uncond, cond, 4.5), torch.tensor([10.0, 15.5])))

    def test_empty_prompt_cache_path_does_not_match_item_te_glob(self):
        path = get_soar_empty_prompt_cache_path("cache", "zi")
        legacy_path = get_legacy_soar_empty_prompt_cache_path("cache", "zi")

        self.assertFalse(fnmatch.fnmatch(Path(path).name, "*_zi_te.safetensors"))
        self.assertFalse(fnmatch.fnmatch(Path(path).name, "*_zi.safetensors"))
        self.assertEqual(Path(path).parent.name, "__soar_empty_prompt")
        self.assertTrue(fnmatch.fnmatch(Path(legacy_path).name, "*_zi.safetensors"))

    def test_save_and_load_empty_prompt_cache_tensor(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        with TemporaryDirectory() as temp_dir:
            legacy_path = Path(get_legacy_soar_empty_prompt_cache_path(temp_dir, "zi"))
            legacy_path.write_text("legacy cache placeholder", encoding="utf-8")
            saved_paths = save_soar_empty_prompt_cache(
                cache_directories=[temp_dir],
                architecture="zi",
                tensor_base_key="varlen_llm_embed",
                tensor=tensor,
                metadata={"model_version": "test"},
            )
            loaded = load_soar_empty_prompt_tensor(
                cache_directories=[temp_dir],
                architecture="zi",
                tensor_base_key="varlen_llm_embed",
            )
            self.assertEqual(len(saved_paths), 1)
            self.assertTrue(torch.equal(loaded, tensor))
            self.assertFalse(legacy_path.exists())

    def test_load_empty_prompt_cache_reports_missing_cache(self):
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                load_soar_empty_prompt_tensor(
                    cache_directories=[temp_dir],
                    architecture="zi",
                    tensor_base_key="varlen_llm_embed",
                )


if __name__ == "__main__":
    unittest.main()

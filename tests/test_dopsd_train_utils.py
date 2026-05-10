import argparse
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from musubi_tuner.dopsd_cache_utils import (
    normalize_qwen_vl_single_file_state_dict,
    prepare_qwen_vl_state_dict_for_load,
    qwen3_vl_processor_id_for_variant,
)
from musubi_tuner.dopsd_train_utils import (
    AdapterEma,
    add_dopsd_arguments,
    run_dopsd_stepwise_backward,
    validate_dopsd_args,
)
from musubi_tuner.zimage.zimage_utils import normalize_qwen3_text_encoder_state_dict


class DummyAccelerator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.backward_calls = []

    def unwrap_model(self, model):
        return model

    def backward(self, loss):
        self.backward_calls.append(float(loss.detach().item()))
        loss.backward()


class DopsdTrainUtilsTest(unittest.TestCase):
    def test_add_dopsd_arguments_is_idempotent(self):
        parser = argparse.ArgumentParser()
        add_dopsd_arguments(parser)
        add_dopsd_arguments(parser)
        args = parser.parse_args([])
        self.assertFalse(args.dopsd)
        self.assertEqual(args.dopsd_num_sampling_steps, 8)
        self.assertEqual(args.dopsd_ema_decay, 0.9999)

    def test_qwen3_vl_processor_ids_are_fixed_by_architecture(self):
        self.assertEqual(qwen3_vl_processor_id_for_variant("4B"), "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(qwen3_vl_processor_id_for_variant("8b"), "Qwen/Qwen3-VL-8B-Instruct")
        with self.assertRaises(ValueError):
            qwen3_vl_processor_id_for_variant("14B")

    def test_qwen_vl_single_file_keys_normalize_to_transformers_format(self):
        state = {
            "model.layers.0.input_layernorm.weight": torch.ones(1),
            "visual.patch_embed.proj.weight": torch.zeros(1),
            "lm_head.weight": torch.full((1,), 2.0),
        }

        normalized = normalize_qwen_vl_single_file_state_dict(state)

        self.assertIn("model.language_model.layers.0.input_layernorm.weight", normalized)
        self.assertIn("model.visual.patch_embed.proj.weight", normalized)
        self.assertIn("lm_head.weight", normalized)

    def test_qwen_vl_loader_prepares_tied_lm_head(self):
        class DummyQwenVl(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(2, 2, bias=False)

        model = DummyQwenVl()
        embed = torch.ones(2, 2)
        prepared = prepare_qwen_vl_state_dict_for_load(model, {"model.language_model.embed_tokens.weight": embed})

        self.assertIs(prepared["lm_head.weight"], embed)

    def test_zimage_qwen3_loader_can_extract_llm_from_hf_qwen_vl_state_dict(self):
        state = {
            "model.language_model.embed_tokens.weight": torch.ones(1),
            "model.language_model.layers.0.input_layernorm.weight": torch.full((1,), 2.0),
            "model.visual.patch_embed.proj.weight": torch.zeros(1),
            "lm_head.weight": torch.full((1,), 3.0),
        }

        normalized = normalize_qwen3_text_encoder_state_dict(state)

        self.assertEqual(set(normalized), {"model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"})
        self.assertEqual(normalized["model.layers.0.input_layernorm.weight"].item(), 2.0)

    def test_validate_dopsd_args_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_dopsd_args(
                argparse.Namespace(dopsd=True, dopsd_loss_weight=0.0, dopsd_num_sampling_steps=8, dopsd_ema_decay=0.9999)
            )
        with self.assertRaises(ValueError):
            validate_dopsd_args(
                argparse.Namespace(dopsd=True, dopsd_loss_weight=1.0, dopsd_num_sampling_steps=0, dopsd_ema_decay=0.9999)
            )
        with self.assertRaises(ValueError):
            validate_dopsd_args(
                argparse.Namespace(dopsd=True, dopsd_loss_weight=1.0, dopsd_num_sampling_steps=8, dopsd_ema_decay=1.1)
            )

    def test_adapter_ema_swaps_and_restores_parameters(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = AdapterEma(model)
        model.weight.data.fill_(3.0)

        with ema.use_ema_weights(model):
            self.assertEqual(model.weight.item(), 2.0)

        self.assertEqual(model.weight.item(), 3.0)
        ema.update(model, decay=0.5)
        self.assertEqual(ema.shadow["weight"].item(), 2.5)

    def test_adapter_ema_can_keep_shadow_and_backup_on_cpu(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = AdapterEma(model, shadow_device="cpu", backup_device="cpu")
        model.weight.data.fill_(3.0)

        self.assertEqual(ema.shadow["weight"].device.type, "cpu")
        with ema.use_ema_weights(model):
            self.assertEqual(model.weight.item(), 2.0)

        self.assertEqual(model.weight.item(), 3.0)
        ema.update(model, decay=0.5)
        self.assertEqual(ema.shadow["weight"].device.type, "cpu")
        self.assertEqual(ema.shadow["weight"].item(), 2.5)

    def test_run_dopsd_stepwise_backward_serializes_each_timestep(self):
        args = argparse.Namespace(dopsd_loss_weight=1.0)
        accelerator = DummyAccelerator()
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = AdapterEma(model)
        model.weight.data.fill_(3.0)

        batch = {"offset": torch.tensor(0.0)}
        teacher_batch = {"offset": torch.tensor(1.0)}
        latents = torch.ones(2, 1)
        timesteps = torch.tensor([1.0, 0.5, 0.25])
        sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0])

        def predict_fn(active_batch, state, step_timesteps):
            del step_timesteps
            return model(state) + active_batch["offset"]

        def rollout_step_fn(state, model_pred, rollout_sigmas, step_index):
            del rollout_sigmas, step_index
            return state + model_pred

        loss, count = run_dopsd_stepwise_backward(
            args=args,
            accelerator=accelerator,
            network=model,
            ema=ema,
            batch=batch,
            latents=latents,
            timesteps=timesteps,
            sigmas=sigmas,
            predict_fn=predict_fn,
            make_teacher_batch_fn=lambda active_batch: teacher_batch,
            rollout_step_fn=rollout_step_fn,
        )

        self.assertEqual(count, 3)
        self.assertEqual(len(accelerator.backward_calls), 3)
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(model.weight.grad)

    def test_run_dopsd_stepwise_backward_uses_accelerator_device(self):
        args = argparse.Namespace(dopsd_loss_weight=1.0)
        accelerator = DummyAccelerator(device="cpu")
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = AdapterEma(model)

        batch = {"offset": torch.tensor(0.0)}
        teacher_batch = {"offset": torch.tensor(1.0)}
        latents = torch.empty(2, 1, device="meta")
        timesteps = torch.tensor([1.0, 0.5])
        sigmas = torch.tensor([1.0, 0.5, 0.0])

        def predict_fn(active_batch, state, step_timesteps):
            self.assertEqual(state.device, accelerator.device)
            self.assertEqual(step_timesteps.device, accelerator.device)
            return state.new_ones(state.shape) * model.weight.sum() + active_batch["offset"].to(accelerator.device)

        def rollout_step_fn(state, model_pred, rollout_sigmas, step_index):
            del rollout_sigmas, step_index
            return state + model_pred

        loss, count = run_dopsd_stepwise_backward(
            args=args,
            accelerator=accelerator,
            network=model,
            ema=ema,
            batch=batch,
            latents=latents,
            timesteps=timesteps,
            sigmas=sigmas,
            predict_fn=predict_fn,
            make_teacher_batch_fn=lambda active_batch: teacher_batch,
            rollout_step_fn=rollout_step_fn,
        )

        self.assertEqual(count, 2)
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(model.weight.grad)

    def test_zimage_cache_can_add_teacher_embedding_without_dropping_student_embedding(self):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and int(np.__version__.split(".", 1)[0]) >= 2:
            self.skipTest("local cv2 build is not compatible with NumPy 2.x")

        try:
            from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_z_image
        except ImportError as exc:
            self.skipTest(f"dataset dependencies are not importable in this environment: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            item = ItemInfo("item", "caption", (0, 0))
            item.text_encoder_output_cache_path = str(Path(temp_dir) / "item_zi_te.safetensors")
            save_text_encoder_output_cache_z_image(item, embed=torch.ones(2, 3, dtype=torch.bfloat16))
            save_text_encoder_output_cache_z_image(
                item,
                dopsd_teacher_embed=torch.zeros(4, 3, dtype=torch.bfloat16),
                dopsd_teacher_key="dopsd_teacher_llm_embed",
            )

            tensors = load_file(item.text_encoder_output_cache_path)
            self.assertIn("varlen_llm_embed_bfloat16", tensors)
            self.assertIn("varlen_dopsd_teacher_llm_embed_bfloat16", tensors)

    def test_content_retrieval_resets_caption_only_after_text_cache_retrieval(self):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and int(np.__version__.split(".", 1)[0]) >= 2:
            self.skipTest("local cv2 build is not compatible with NumPy 2.x")

        try:
            from PIL import Image

            from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_Z_IMAGE, ImageDataset
        except ImportError as exc:
            self.skipTest(f"dataset dependencies are not importable in this environment: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "item.png"
            caption_path = Path(temp_dir) / "item.txt"
            Image.new("RGB", (64, 64), color=(255, 0, 0)).save(image_path)
            caption_path.write_text("caption", encoding="utf-8")

            dataset = ImageDataset(
                resolution=(64, 64),
                batch_size=1,
                caption_extension=".txt",
                enable_bucket=False,
                bucket_no_upscale=False,
                cache_directory=temp_dir,
                debug_dataset=False,
                architecture=ARCHITECTURE_Z_IMAGE,
                image_directory=temp_dir,
            )

            list(dataset.retrieve_text_encoder_output_cache_batches(num_workers=1))
            batches = list(dataset.retrieve_latent_cache_batches(num_workers=1))

            self.assertEqual(len(batches), 1)
            self.assertEqual(len(batches[0][1]), 1)
            self.assertIsNotNone(batches[0][1][0].content)

    def test_flux2_cache_can_add_teacher_embedding_without_dropping_student_embedding(self):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and int(np.__version__.split(".", 1)[0]) >= 2:
            self.skipTest("local cv2 build is not compatible with NumPy 2.x")

        try:
            from musubi_tuner.dataset.image_video_dataset import (
                ARCHITECTURE_FLUX_2_KLEIN_4B_FULL,
                ItemInfo,
                save_text_encoder_output_cache_flux_2,
            )
        except ImportError as exc:
            self.skipTest(f"dataset dependencies are not importable in this environment: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            flux_item = ItemInfo("flux_item", "caption", (0, 0))
            flux_item.text_encoder_output_cache_path = str(Path(temp_dir) / "flux_item_te.safetensors")
            save_text_encoder_output_cache_flux_2(
                flux_item,
                ctx_vec=torch.ones(2, 4, dtype=torch.bfloat16),
                arch_full=ARCHITECTURE_FLUX_2_KLEIN_4B_FULL,
            )
            save_text_encoder_output_cache_flux_2(
                flux_item,
                arch_full=ARCHITECTURE_FLUX_2_KLEIN_4B_FULL,
                dopsd_teacher_ctx_vec=torch.zeros(2, 4, dtype=torch.bfloat16),
                dopsd_teacher_key="dopsd_teacher_ctx_vec",
            )

            flux_tensors = load_file(flux_item.text_encoder_output_cache_path)
            self.assertIn("ctx_vec_bfloat16", flux_tensors)
            self.assertIn("dopsd_teacher_ctx_vec_bfloat16", flux_tensors)


if __name__ == "__main__":
    unittest.main()

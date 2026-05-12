import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from musubi_tuner.dopsd_cache_utils import (
    DOPSD_QWEN3_VL_MAX_PIXELS,
    DOPSD_QWEN3_VL_MIN_PIXELS,
    extract_masked_hidden_state,
    load_qwen3_vl_processor,
    normalize_qwen_vl_single_file_state_dict,
    prepare_qwen_vl_state_dict_for_load,
    qwen3_vl_processor_id_for_variant,
)
from musubi_tuner.dopsd_train_utils import (
    AdapterEma,
    DOPSD_FLUX2_IDENTITY_EDIT_PROMPT,
    DOPSD_FLUX2_TEACHER_EMBED_KEY,
    add_dopsd_full_finetune_arguments,
    add_dopsd_arguments,
    dopsd_flow_x0_loss,
    dopsd_x0_loss,
    resolve_full_ema_devices,
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

    def test_add_dopsd_full_finetune_arguments_is_idempotent(self):
        parser = argparse.ArgumentParser()
        add_dopsd_full_finetune_arguments(parser)
        add_dopsd_full_finetune_arguments(parser)
        args = parser.parse_args([])

        self.assertEqual(args.dopsd_full_ema_device, "auto")

    def test_qwen3_vl_processor_ids_are_fixed_by_architecture(self):
        self.assertEqual(qwen3_vl_processor_id_for_variant("4B"), "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(qwen3_vl_processor_id_for_variant("8b"), "Qwen/Qwen3-VL-8B-Instruct")
        with self.assertRaises(ValueError):
            qwen3_vl_processor_id_for_variant("14B")

    def test_qwen3_vl_processor_uses_official_pixel_bounds(self):
        calls = {}

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls["args"] = args
                calls["kwargs"] = kwargs
                return "processor"

        fake_transformers = types.SimpleNamespace(__version__="4.57.6", AutoProcessor=FakeAutoProcessor)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            processor = load_qwen3_vl_processor("4B")

        self.assertEqual(processor, "processor")
        self.assertEqual(calls["args"], ("Qwen/Qwen3-VL-4B-Instruct",))
        self.assertTrue(calls["kwargs"]["trust_remote_code"])
        self.assertEqual(calls["kwargs"]["min_pixels"], DOPSD_QWEN3_VL_MIN_PIXELS)
        self.assertEqual(calls["kwargs"]["max_pixels"], DOPSD_QWEN3_VL_MAX_PIXELS)

    def test_masked_teacher_hidden_state_respects_sequence_cap(self):
        hidden = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        mask = torch.tensor([True, True, False, True, True, True, True, True, True, True])

        embed = extract_masked_hidden_state(hidden, mask, max_sequence_length=4)

        self.assertEqual(embed.shape, (4, 3))
        self.assertTrue(torch.equal(embed, hidden[mask][:4]))

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

    def test_adapter_ema_uses_storage_swap_when_shadow_is_on_parameter_device(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = AdapterEma(model)
        ema_ptr = ema.shadow["weight"].data_ptr()
        student_ptr = model.weight.data.data_ptr()
        model.weight.data.fill_(3.0)

        with ema.use_ema_weights(model):
            self.assertEqual(model.weight.item(), 2.0)
            self.assertEqual(model.weight.data_ptr(), ema_ptr)
            self.assertEqual(ema.shadow["weight"].data_ptr(), student_ptr)
            self.assertEqual(ema.shadow["weight"].item(), 3.0)

        self.assertEqual(model.weight.item(), 3.0)
        self.assertEqual(model.weight.data_ptr(), student_ptr)
        self.assertEqual(ema.shadow["weight"].data_ptr(), ema_ptr)

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

    def test_adapter_ema_refreshes_parameter_cache_on_update(self):
        class DynamicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2.0]))

        model = DynamicModel()
        ema = AdapterEma(model)
        model.extra = torch.nn.Parameter(torch.tensor([4.0]))

        ema.update(model, decay=0.5)

        self.assertIn("extra", ema.shadow)
        self.assertEqual(ema.shadow["extra"].item(), 4.0)

    def test_resolve_full_ema_devices_has_explicit_tiers(self):
        model = torch.nn.Linear(1, 1, bias=False)

        shadow_device, backup_device, label = resolve_full_ema_devices(
            argparse.Namespace(dopsd_full_ema_device="cpu"), model, torch.device("cpu")
        )
        self.assertEqual(shadow_device, torch.device("cpu"))
        self.assertEqual(backup_device, torch.device("cpu"))
        self.assertEqual(label, "cpu")

        shadow_device, backup_device, label = resolve_full_ema_devices(
            argparse.Namespace(dopsd_full_ema_device="gpu"), model, torch.device("cuda")
        )
        self.assertIsNone(shadow_device)
        self.assertIsNone(backup_device)
        self.assertEqual(label, "gpu")

        with self.assertRaisesRegex(ValueError, "requires a CUDA"):
            resolve_full_ema_devices(argparse.Namespace(dopsd_full_ema_device="gpu"), model, torch.device("cpu"))

        with patch("musubi_tuner.dopsd_train_utils._has_cuda_memory_for_gpu_ema", return_value=True):
            shadow_device, backup_device, label = resolve_full_ema_devices(
                argparse.Namespace(dopsd_full_ema_device="auto"), model, torch.device("cuda")
            )
        self.assertIsNone(shadow_device)
        self.assertIsNone(backup_device)
        self.assertEqual(label, "gpu-auto")

        with patch("musubi_tuner.dopsd_train_utils._has_cuda_memory_for_gpu_ema", return_value=False):
            shadow_device, backup_device, label = resolve_full_ema_devices(
                argparse.Namespace(dopsd_full_ema_device="auto"), model, torch.device("cuda")
            )
        self.assertEqual(shadow_device, torch.device("cpu"))
        self.assertEqual(backup_device, torch.device("cpu"))
        self.assertEqual(label, "cpu-auto")

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

    def test_dopsd_x0_loss_matches_official_objective(self):
        state = torch.tensor([[1.0, -2.0]])
        student_pred = torch.tensor([[2.0, 0.5]])
        teacher_pred = torch.tensor([[0.0, 4.0]])
        sigmas = torch.tensor([0.25, 0.0])

        loss = dopsd_x0_loss(state, student_pred, teacher_pred, sigmas, 0)
        expected = torch.nn.functional.mse_loss(state + 0.25 * student_pred, state + 0.25 * teacher_pred)

        self.assertTrue(torch.allclose(loss, expected))

    def test_dopsd_flow_x0_loss_uses_flux_velocity_sign(self):
        state = torch.tensor([[1.0, -2.0]])
        student_pred = torch.tensor([[2.0, 0.5]])
        teacher_pred = torch.tensor([[0.0, 4.0]])
        sigmas = torch.tensor([0.25, 0.0])

        loss = dopsd_flow_x0_loss(state, student_pred, teacher_pred, sigmas, 0)
        expected = torch.nn.functional.mse_loss(state - 0.25 * student_pred, state - 0.25 * teacher_pred)

        self.assertTrue(torch.allclose(loss, expected))

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

    def test_flux2_dopsd_teacher_cache_uses_identity_edit_qwen_context(self):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and int(np.__version__.split(".", 1)[0]) >= 2:
            self.skipTest("local cv2 build is not compatible with NumPy 2.x")

        try:
            from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_FLUX_2_KLEIN_4B_FULL, ItemInfo
            from musubi_tuner.flux_2_cache_text_encoder_outputs import encode_and_save_dopsd_teacher_batch
        except ImportError as exc:
            self.skipTest(f"FLUX.2 dependencies are not importable in this environment: {exc}")

        class FakeTextEmbedder(torch.nn.Module):
            @property
            def dtype(self):
                return torch.bfloat16

            def forward(self, prompts):
                self.prompts = prompts
                return torch.ones(len(prompts), 2, 4, dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as temp_dir:
            items = []
            for index in range(2):
                item = ItemInfo(f"flux_item_{index}", "caption", (0, 0))
                item.text_encoder_output_cache_path = str(Path(temp_dir) / f"flux_item_{index}_te.safetensors")
                items.append(item)

            embedder = FakeTextEmbedder()
            encode_and_save_dopsd_teacher_batch(
                embedder,
                items,
                torch.device("cpu"),
                expected_dim=4,
                teacher_embed_key=DOPSD_FLUX2_TEACHER_EMBED_KEY,
                arch_full=ARCHITECTURE_FLUX_2_KLEIN_4B_FULL,
            )

            self.assertEqual(embedder.prompts, [DOPSD_FLUX2_IDENTITY_EDIT_PROMPT, DOPSD_FLUX2_IDENTITY_EDIT_PROMPT])
            for item in items:
                tensors = load_file(item.text_encoder_output_cache_path)
                self.assertIn("dopsd_teacher_ctx_vec_bfloat16", tensors)
                self.assertEqual(tuple(tensors["dopsd_teacher_ctx_vec_bfloat16"].shape), (2, 4))

    def test_flux2_dopsd_teacher_batch_injects_target_latents_as_reference(self):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and int(np.__version__.split(".", 1)[0]) >= 2:
            self.skipTest("local cv2 build is not compatible with NumPy 2.x")

        try:
            from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer
        except ImportError as exc:
            self.skipTest(f"FLUX.2 dependencies are not importable in this environment: {exc}")

        trainer = Flux2NetworkTrainer()
        trainer.model_version_info = types.SimpleNamespace(params=types.SimpleNamespace(context_in_dim=4))
        batch = {
            "ctx_vec": torch.zeros(2, 3, 4),
            DOPSD_FLUX2_TEACHER_EMBED_KEY: torch.ones(2, 3, 4),
        }
        latents = torch.randn(2, 128, 4, 4)

        teacher_batch = trainer.make_dopsd_teacher_batch(argparse.Namespace(), batch, latents)

        self.assertIs(teacher_batch["ctx_vec"], batch[DOPSD_FLUX2_TEACHER_EMBED_KEY])
        self.assertTrue(torch.equal(teacher_batch["latents_control_0"], latents))
        self.assertFalse(teacher_batch["latents_control_0"].requires_grad)
        self.assertNotIn("latents_control_0", batch)

    def test_flux2_dopsd_teacher_batch_requires_target_latents(self):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None and int(np.__version__.split(".", 1)[0]) >= 2:
            self.skipTest("local cv2 build is not compatible with NumPy 2.x")

        try:
            from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer
        except ImportError as exc:
            self.skipTest(f"FLUX.2 dependencies are not importable in this environment: {exc}")

        trainer = Flux2NetworkTrainer()
        trainer.model_version_info = types.SimpleNamespace(params=types.SimpleNamespace(context_in_dim=4))
        batch = {DOPSD_FLUX2_TEACHER_EMBED_KEY: torch.ones(2, 3, 4)}

        with self.assertRaisesRegex(ValueError, "requires current target latents"):
            trainer.make_dopsd_teacher_batch(argparse.Namespace(), batch)


if __name__ == "__main__":
    unittest.main()

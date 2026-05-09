import argparse
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from musubi_tuner.dopsd_train_utils import (
    AdapterEma,
    add_dopsd_arguments,
    run_dopsd_stepwise_backward,
    validate_dopsd_args,
)


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


if __name__ == "__main__":
    unittest.main()

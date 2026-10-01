"""Shared single-process FP32 runner for NR full and LoRA supervised training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from accelerate.utils import set_seed

from musubi_tuner.dlssnr.artifacts import inspect_canonical, save_canonical, write_json
from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from musubi_tuner.dlssnr.dataset import (
    NRBatchPlan,
    collate_clips,
    collate_single_frames,
    load_single_frame_manifest,
    load_temporal_manifest,
)
from musubi_tuner.dlssnr.evaluation import evaluate
from musubi_tuner.dlssnr.identity import file_sha256, implementation_identity
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.numerics import SURROGATE_FLAGS, fp32_execution
from musubi_tuner.dlssnr.profiles import PROFILE_ID
from musubi_tuner.dlssnr.temporal import SEED_POLICY, stable_frame_seed
from musubi_tuner.dlssnr.training_step import loss_denominators, training_loss
from musubi_tuner.training.dlssnr_services import (
    assert_finite_gradients,
    assert_finite_parameters,
    create_accelerator,
    create_nr_optimizer,
    evaluation_mode,
    move_batch,
    reject_unsupported_runtime,
)
from musubi_tuner.training.dlssnr_state import read_state, restore_state, save_state

logger = logging.getLogger(__name__)


def parameter_group(name: str) -> str:
    if "head.logit" in name:
        return "temporal_head"
    if name.endswith("blend_scale"):
        return "temporal_blend"
    if name.endswith("prior"):
        return "priors"
    if name.endswith(("skip_scale", "temperature", "up_scale", "adapter_scale")):
        return "scales"
    if "input_adapter" in name or "head.rgb" in name:
        return "input_rgb_head"
    return "matrices"


def build_optimizer(model, learning_rate, multipliers, weight_decay, *, include_temporal=False, optimizer_config=None):
    if include_temporal:
        model.blocks["70"].head.logit.weight.requires_grad_(True)
        model.blocks["70"].blend_scale.requires_grad_(True)
    else:
        model.freeze_single_frame()
    grouped = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped.setdefault(parameter_group(name), []).append(parameter)
    groups = [
        {"params": parameters, "lr": learning_rate * multipliers.get(name, 1.0), "name": name}
        for name, parameters in grouped.items()
    ]
    if optimizer_config is None:
        optimizer_config = {
            "type": "AdamW",
            "args": [f"weight_decay={weight_decay!r}"],
            "learning_rate": learning_rate,
            "lr_scheduler": "constant",
            "max_grad_norm": 0.0,
        }
    optimizer = create_nr_optimizer(groups, optimizer_config)
    _check_optimizer(model, optimizer)
    return optimizer


def _check_optimizer(module, optimizer):
    opted = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    trainable = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    if len(opted) != len(set(opted)) or set(opted) != trainable:
        raise RuntimeError("optimizer parameters must exactly match the unique trainable parameters")


def clear_lane15_state(model, optimizer):
    model.enforce_lane15()
    weight = model.blocks["0"].input_adapter.weight
    state = optimizer.state.get(weight, {})
    for value in state.values():
        # Integer-coded quantized states cannot be cleared as floating-point moments.
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.shape == weight.shape:
            value[:, 15] = 0


class NRTrainModule(torch.nn.Module):
    """The only prepared model; adapters have exactly one registered owner."""

    def __init__(self, model, loss_weights, burn_in=0, network=None):
        super().__init__()
        self.model = model
        self.network = network
        self.loss_weights = loss_weights
        self.burn_in = burn_in

    def forward(self, batch, seeds, normalizers=None):
        return training_loss(self.model, batch, seeds, self.loss_weights, self.burn_in, normalizers)


def _one_update(model, optimizer, batch, seeds, weights, burn_in):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = training_loss(model, batch, seeds, {"pre": 1.0, "out": 1.0, "edge": 0.05, **weights}, burn_in)
    loss.backward()
    model.enforce_lane15()
    assert_finite_gradients([parameter for group in optimizer.param_groups for parameter in group["params"]])
    optimizer.step()
    clear_lane15_state(model, optimizer)
    assert_finite_parameters(model)
    return metrics


def single_frame_update(model, optimizer, source, target, controls, frame_seed, loss_weights):
    return _one_update(model, optimizer, {"source": source, "target": target, "controls": controls}, frame_seed, loss_weights, 0)


def temporal_clip_update(model, optimizer, batch, seeds, loss_weights, burn_in):
    return _one_update(model, optimizer, batch, seeds, loss_weights, burn_in)


def _datasets(config):
    data, training = config["data"], config["training"]
    width, height = data["bucket_size"]
    buckets = {name: data[name] for name in ("enable_bucket", "bucket_no_upscale")}
    if training["mode"] == "single_frame":
        train = load_single_frame_manifest(data["train_manifest"], width, height, **buckets)
    else:
        train = load_temporal_manifest(
            data["train_manifest"],
            width,
            height,
            training["sequence_length"],
            require_temporal_mask=bool(config["loss"]["temporal"]),
            **buckets,
        )
    validation = {}
    if data.get("validation_manifest"):
        validation["validation"] = load_temporal_manifest(
            data["validation_manifest"], width, height, None, require_temporal_mask=False, **buckets
        )
    if config["evaluation"].get("sequence_manifest"):
        sequences = load_temporal_manifest(
            config["evaluation"]["sequence_manifest"], width, height, None, require_temporal_mask=False, **buckets
        )
        if any(len(row["frames"]) < config["evaluation"]["min_sequence_frames"] for row in sequences.rows):
            raise ValueError("evaluation sequence is shorter than min_sequence_frames")
        validation["sequences"] = sequences
    for name, dataset in validation.items():
        if overlap := train.sequence_ids & dataset.sequence_ids:
            raise ValueError(f"train/{name} sequence overlap: {sorted(overlap)}")
    for dataset in (train, *validation.values()):
        dataset.validate()
    return train, validation


def _microbatch(dataset, microbatch_index, config, plan):
    training = config["training"]
    samples = [dataset[index] for index in plan.indices(microbatch_index)]
    epoch = microbatch_index // len(plan)
    seeds = []
    for sample in samples:
        frames = [sample] if training["mode"] == "single_frame" else sample["frames"]
        seeds.append(
            [
                stable_frame_seed(training["seed"], epoch, sample["sample_id"], frame["frame_index"], sample["crop_id"])
                for frame in frames
            ]
        )
    if training["mode"] == "single_frame":
        return collate_single_frames(samples), [item[0] for item in seeds]
    return collate_clips(samples), seeds


class NRSupervisedTrainer:
    def __init__(self, config, *, lora=False):
        self.config = config
        self.lora = lora

    @fp32_execution()
    def train(self, resume=None):
        from musubi_tuner.networks.lora_dlssnr import base_target_sha256, inject, load_adapter
        from musubi_tuner.dlssnr.convert import _git_commit

        config = self.config
        training, output = config["training"], config["output"]
        train_data, validation = _datasets(config)
        batch_plan = NRBatchPlan(train_data, training["batch_size"])
        batch_plan.manager.show_bucket_info()
        source_dir = config["model"].get("model_dir")
        source_identity = (
            inspect_canonical(
                source_dir,
                development_smoke=training["development_smoke"],
                validation_report=config["model"].get("forward_validation_report"),
            )
            if source_dir
            else {}
        )
        source_forward_validated = "forward_validation_report" in source_identity
        accelerator = create_accelerator(training)
        set_seed(training["seed"])
        model = NRModel().to(dtype=torch.float32)
        if source_dir:
            model.load_canonical(str(Path(source_dir) / "model.safetensors"))
        base_identity = base_target_sha256(model, [])
        network = inject(model, config["lora"]) if self.lora else None
        optimizer_cfg = config["optimizer"]
        if self.lora:
            optimizer = create_nr_optimizer(network.parameters(), optimizer_cfg)
        else:
            groups = config["parameter_groups"]
            optimizer = build_optimizer(
                model,
                optimizer_cfg["learning_rate"],
                {
                    "priors": groups["prior_lr_multiplier"],
                    "scales": groups["scale_lr_multiplier"],
                    "temporal_blend": groups["temporal_blend_lr_multiplier"],
                },
                0.0,
                include_temporal=training["mode"] == "temporal",
                optimizer_config=optimizer_cfg,
            )
        train_module = NRTrainModule(model, config["loss"], training["burn_in"], network)
        _check_optimizer(train_module, optimizer)
        parameter_map = {
            name: {"shape": list(parameter.shape), "elements": parameter.numel()}
            for name, parameter in train_module.named_parameters()
            if parameter.requires_grad
        }
        optimizer_groups = [
            {
                "name": group.get("name", "lora"),
                "lr": group["lr"],
                "weight_decay": group.get("weight_decay", 0.0),
                "elements": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        ]
        identity = {
            "config_sha256": config_sha256(config),
            "source": source_identity,
            "base_parameters_sha256": base_identity,
            "data": {name: dataset.fingerprint() for name, dataset in {"train": train_data, **validation}.items()},
            "implementation": implementation_identity(),
            "lora": self.lora,
            "device_type": accelerator.device.type,
            "torch_version": str(torch.__version__),
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "seed_policy": SEED_POLICY,
            "bucket_plan": batch_plan.report(),
            "cuda_version": torch.version.cuda if accelerator.device.type == "cuda" else None,
            "cuda_device": torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == "cuda" else None,
        }
        for name in ("config.py", "dataset.py", "filenames.py", "training_step.py", "losses.py"):
            identity["implementation"][name] = file_sha256(Path(__file__).parents[1] / "dlssnr" / name)
        for name in ("dlssnr_trainer.py", "dlssnr_services.py", "dlssnr_state.py", "optimizer_setup.py"):
            identity["implementation"][name] = file_sha256(Path(__file__).parent / name)
        for name in ("bucket.py", "architectures.py"):
            identity["implementation"][f"dataset/{name}"] = file_sha256(Path(__file__).parents[1] / "dataset" / name)
        identity["implementation"]["lora_dlssnr.py"] = file_sha256(Path(__file__).parents[1] / "networks/lora_dlssnr.py")
        restored = read_state(resume, identity) if resume else None
        completed = restored["global_update"] if restored else 0
        cursor = restored["consumed_samples"] if restored else 0
        steps, accum = training["max_train_steps"], training["gradient_accumulation_steps"]
        if completed > steps or cursor != batch_plan.sample_count(completed * accum):
            raise ValueError("resume counters are inconsistent with max_train_steps or the consumed batch plan")
        output_dir = Path(output["output_dir"]) / output["output_name"]
        try:
            output_dir.mkdir(parents=True, exist_ok=restored is not None)
        except FileExistsError:
            raise FileExistsError(
                f"run directory already exists: {output_dir}; choose a different output_name or use --resume"
            ) from None
        wrapped, optimizer = accelerator.prepare(train_module, optimizer)
        logger.info(
            "NR device=%s trainable=%d mode=%s lora=%s",
            accelerator.device,
            sum(item["elements"] for item in parameter_map.values()),
            training["mode"],
            self.lora,
        )
        metadata = {
            "config": config,
            "identity": identity,
            "profile": PROFILE_ID,
            "numerics": SURROGATE_FLAGS,
            "trainer_commit": _git_commit(),
            "trainable_parameters": parameter_map,
            "optimizer_groups": optimizer_groups,
            "bucket_plan": batch_plan.report(),
            "device": str(accelerator.device),
            "source_forward_validated": source_forward_validated,
            "experimental_surrogate": training["development_smoke"] or not source_forward_validated,
            "temporal_trained": training["mode"] == "temporal",
        }
        baseline = None
        if validation and config["evaluation"]["compare_baseline"]:
            with evaluation_mode(train_module):
                baseline = evaluate(model, validation, training["seed"], accelerator.device)
        if restored:
            if network is not None:
                load_adapter(network, Path(resume) / "adapter.safetensors", base_identity)
            else:
                model.load_canonical(str(Path(resume) / "model.safetensors"))
            restore_state(restored, optimizer)
        write_json(output_dir / "run_config.json", metadata)
        if network is not None:
            write_json(output_dir / "lora_report.json", network.report)
        if training["development_smoke"]:
            logger.warning("EXPERIMENTAL NR smoke run: native/float compatibility has not been validated")
        elif not source_forward_validated:
            logger.warning("NR source forward compatibility is unvalidated; training does not certify native/DLL compatibility")
        try:
            for step in range(completed, steps):
                train_module.train()
                optimizer.zero_grad(set_to_none=True)
                microbatches = [_microbatch(train_data, step * accum + micro, config, batch_plan) for micro in range(accum)]
                denominators = {name: 0.0 for name in config["loss"]}
                for batch, _ in microbatches:
                    for name, count in loss_denominators(batch, training["burn_in"]).items():
                        denominators[name] += count
                metrics = {}
                for batch, seeds in microbatches:
                    with accelerator.accumulate(wrapped):
                        loss, measured = wrapped(move_batch(batch, accelerator.device), seeds, denominators)
                        # Losses already use the effective batch's valid-element denominator.
                        accelerator.backward(loss * accum)
                    for name, value in measured.items():
                        metrics[name] = max(metrics.get(name, 0), value) if name == "blend_max" else metrics.get(name, 0) + value
                if not accelerator.sync_gradients:
                    raise RuntimeError("optimizer update is not at an accumulation boundary")
                model.enforce_lane15()
                assert_finite_gradients(train_module)
                if optimizer_cfg["max_grad_norm"]:
                    accelerator.clip_grad_norm_(train_module.parameters(), optimizer_cfg["max_grad_norm"])
                optimizer.step()
                if accelerator.optimizer_step_was_skipped:
                    raise RuntimeError("unexpected skipped FP32 optimizer update")
                clear_lane15_state(model, optimizer)
                assert_finite_parameters(train_module)
                cursor += sum(len(seeds) for _, seeds in microbatches)
                update = step + 1
                logger.info("NR update %d/%d loss=%.6f", update, steps, metrics["loss"])
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"update": update, "consumed_samples": cursor, **metrics}, allow_nan=False) + "\n")
                interval = config["evaluation"]["sample_every_n_steps"]
                if validation and ((interval and update % interval == 0) or update == steps):
                    with evaluation_mode(train_module):
                        candidate = evaluate(model, validation, training["seed"], accelerator.device)
                    write_json(
                        output_dir / "evaluation" / f"step{update:06d}.json",
                        {"update": update, "numerics_profile": "train_surrogate", "baseline": baseline, "candidate": candidate},
                    )
                save_every = output["save_every_n_steps"]
                if save_every and update % save_every == 0:
                    prefix = "state-step" if output["save_state"] else "step"
                    folder = output_dir / f"{prefix}{update:06d}"
                    self._save_product(folder, model, network, source_dir, metadata, base_identity, update)
                    if output["save_state"]:
                        save_state(folder, optimizer, identity, update, cursor)
            if output["save_state"] and (not output["save_every_n_steps"] or steps % output["save_every_n_steps"]):
                folder = output_dir / f"state-step{steps:06d}"
                self._save_product(folder, model, network, source_dir, metadata, base_identity, steps)
                save_state(folder, optimizer, identity, steps, cursor)
            self._save_product(output_dir / "final", model, network, source_dir, metadata, base_identity, steps)
        finally:
            accelerator.end_training()
            accelerator.free_memory()

    @staticmethod
    def _save_product(folder, model, network, source_dir, metadata, base_identity, update):
        from musubi_tuner.networks.lora_dlssnr import save_adapter

        metadata = {
            **metadata,
            "global_update": update,
            "float_validated": False,
            "temporal_validated": False,
            "native_export_validated": False,
        }
        if network is None:
            save_canonical(model, folder, source_dir=source_dir, metadata=metadata)
        else:
            save_adapter(network, folder / "adapter.safetensors", base_identity)
            write_json(folder / "training_metadata.json", metadata)
            write_json(
                folder / "base_identity.json",
                {
                    "base_profile": PROFILE_ID,
                    "base_weight_sha256": base_identity,
                    "identity_scope": "all_canonical_parameters",
                    "targets": network.target_names,
                },
            )
        write_json(folder / "run_config.json", metadata["config"])


def _train_from_args(args, *, lora):
    reject_unsupported_runtime()
    config = build_train_config(args, lora=lora)
    NRSupervisedTrainer(config, lora=lora).train(args.resume)


def train_from_args(args):
    _train_from_args(args, lora=False)


def train_lora_from_args(args):
    _train_from_args(args, lora=True)

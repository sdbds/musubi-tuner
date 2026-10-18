"""MOSS-local training loop shared by DiT adapters and full fine-tuning."""

import json
import logging
import math
import os
from pathlib import Path
import shutil
import time

import toml
import torch
from tqdm import tqdm

from musubi_tuner.moss_image.checkpoint import TrainingProgress, dataset_fingerprint, read_progress, save_full_model
from musubi_tuner.training.sampling_prompts import should_sample_images
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils

logger = logging.getLogger(__name__)


def _remove_output(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Checkpoint retention path is outside output_dir")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _save_state(args, accelerator, kind, number):
    if kind == "step":
        name = train_utils.STEP_STATE_NAME.format(args.output_name, number)
    elif kind == "epoch":
        name = train_utils.EPOCH_STATE_NAME.format(args.output_name, number)
    else:
        name = train_utils.LAST_STATE_NAME.format(args.output_name)
    directory = os.path.join(args.output_dir, name)
    accelerator.save_state(directory)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    if args.save_state_to_huggingface:
        huggingface_utils.upload(args, directory, "/" + name)
    if kind == "step":
        keep = args.save_last_n_steps_state or args.save_last_n_steps
        if keep is not None:
            old = number - keep - 1
            old -= old % args.save_every_n_steps
            if old > 0:
                _remove_output(args.output_dir, train_utils.STEP_STATE_NAME.format(args.output_name, old))
    elif kind == "epoch":
        keep = args.save_last_n_epochs_state or args.save_last_n_epochs
        if keep is not None:
            old = number - args.save_every_n_epochs * keep
            if old > 0:
                _remove_output(args.output_dir, train_utils.EPOCH_STATE_NAME.format(args.output_name, old))


def _metadata(args, progress, optimizer_name, started, full_finetune):
    metadata = {
        "ss_base_model_version": "moss_image",
        "ss_training_type": "full-finetune" if full_finetune else "lora",
        "ss_steps": str(progress.global_step),
        "ss_epoch": str(progress.epoch),
        "ss_training_started_at": str(started),
        "ss_training_finished_at": str(time.time()),
        "ss_sd_model_name": os.path.basename(args.dit),
        "ss_seed": str(args.seed),
        "ss_optimizer": optimizer_name,
        "ss_learning_rate": str(args.learning_rate),
        "ss_mixed_precision": str(args.mixed_precision),
        "ss_timestep_sampling": args.timestep_sampling,
        "ss_weighting_scheme": args.weighting_scheme,
    }
    if not full_finetune:
        metadata.update(
            {
                "ss_network_module": args.network_module,
                "ss_network_dim": str(args.network_dim),
                "ss_network_alpha": str(args.network_alpha),
                "ss_network_args": json.dumps(dict(arg.split("=", 1) for arg in args.network_args or [])),
            }
        )
    if args.no_metadata:
        required = {
            "ss_base_model_version",
            "ss_training_type",
            "ss_network_module",
            "ss_network_dim",
            "ss_network_alpha",
            "ss_network_args",
        }
        return {key: value for key, value in metadata.items() if key in required}
    metadata.update(
        sai_model_spec.build_metadata(
            None,
            "moss",
            time.time(),
            args.metadata_title or args.output_name,
            args.metadata_reso,
            args.metadata_author,
            args.metadata_description,
            args.metadata_license,
            args.metadata_tags,
            is_lora=not full_finetune,
            custom_arch=args.metadata_arch,
        )
    )
    return metadata


def _move_full_gradients(model, optimizer):
    model.wait_for_pending_block_moves()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and parameter.grad.device != parameter.device:
                parameter.grad = parameter.grad.to(parameter.device)
            for name, value in optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor) and name != "step" and value.device != parameter.device:
                    optimizer.state[parameter][name] = value.to(parameter.device)


def _synchronize_lora_gradients(network, accelerator):
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    present = torch.tensor([parameter.grad is not None for parameter in parameters], device=accelerator.device, dtype=torch.int32)
    present = accelerator.reduce(present, reduction="sum").cpu().tolist()
    # Module dropout can leave different parameters unused on different ranks.
    for parameter, used in zip(parameters, present):
        if used:
            gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            parameter.grad = accelerator.reduce(gradient, reduction="mean")


def _install_fused_adafactor(optimizer):
    from musubi_tuner.modules.adafactor_fused import patch_adafactor_fused

    patch_adafactor_fused(optimizer)
    handles = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:

            def update(param, param_group=group):
                optimizer.step_param(param, param_group)
                param.grad = None

            handles.append(parameter.register_post_accumulate_grad_hook(update))
    return handles


def train(trainer, args, full_finetune=False):
    resume_progress = read_progress(args.resume) if args.resume else None
    if resume_progress is not None:
        if args.seed is not None and args.seed != resume_progress["seed"]:
            raise ValueError("MOSS resume seed mismatch")
        args.seed = resume_progress["seed"]
    if not trainer._validate_args_and_init(args):
        return
    if Path(args.output_name).name != args.output_name or args.output_name in (".", ".."):
        raise ValueError("output_name must be a filename, not a path")
    if args.gradient_accumulation_steps < 1 or args.max_train_steps < 1:
        raise ValueError("Training and accumulation steps must be positive")
    if args.resume_from_huggingface:
        raise ValueError("Download MOSS training state locally before passing --resume")
    if args.xm_best_of_k != 1:
        raise ValueError("MOSS training currently supports xm_best_of_k=1 only")
    if args.dynamo_backend.upper() != "NO":
        raise ValueError("Use MOSS per-block --compile rather than the Accelerate dynamo backend")
    for option in ("save_every_n_steps", "save_every_n_epochs"):
        value = getattr(args, option)
        if value is not None and value < 1:
            raise ValueError(f"{option} must be positive")
    _, started = trainer._init_session(args)
    dataset, collator, current_epoch = trainer._build_dataset(args)
    accelerator, weight_dtype, dit_dtype, dit_weight_dtype, vae_dtype = trainer._prepare_accelerator_and_dtypes(args)
    fused = bool(full_finetune and args.fused_backward_pass)
    if full_finetune and accelerator.num_processes > 1 and (args.blocks_to_swap or fused):
        raise ValueError("Full-model block swap and fused backward currently require a single process")
    if args.blocks_to_swap and accelerator.device.type != "cuda":
        raise ValueError("MOSS block swapping requires CUDA")
    progress = TrainingProgress(
        args.seed,
        "full-finetune" if full_finetune else "lora",
        args.gradient_accumulation_steps,
        dataset_fingerprint(dataset),
        accelerator.num_processes,
        num_timestep_buckets=args.num_timestep_buckets,
        preserve_distribution_shape=args.preserve_distribution_shape,
    )
    if resume_progress is not None:
        progress.load_state_dict(resume_progress)
    samples, vae = trainer.prepare_sampling(args, accelerator, vae_dtype)
    transformer = trainer._load_dit_and_swap(args, accelerator, dit_weight_dtype)
    if full_finetune:
        transformer.requires_grad_(True).train()
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
        network = None
        groups = [{"params": list(transformer.parameters()), "lr": args.learning_rate}]
        lr_descriptions = None
    else:
        network = trainer._build_network(args, accelerator, transformer, vae, weight_dtype, dataset)
        network.to(device=accelerator.device, dtype=torch.float32)
        network.prepare_grad_etc(transformer)
        groups, lr_descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        transformer.train(args.gradient_checkpointing)
    optimizer_name, _, optimizer, optimizer_train, optimizer_eval = trainer.get_optimizer(args, groups)
    fused_handles = _install_fused_adafactor(optimizer) if fused else []
    loader_generator = torch.Generator().manual_seed(args.seed)
    workers = min(args.max_data_loader_n_workers, os.cpu_count() or 1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collator,
        num_workers=workers,
        persistent_workers=args.persistent_data_loader_workers and workers > 0,
    )
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
    dataset.set_max_train_steps(args.max_train_steps)
    scheduler = trainer.get_lr_scheduler(args, optimizer, accelerator.num_processes)
    if args.compile:
        transformer = trainer.compile_transformer(args, transformer)
    if full_finetune:
        transformer = accelerator.prepare(transformer, device_placement=[not bool(trainer.blocks_to_swap)])
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        training_model = transformer
    else:
        network, optimizer, dataloader, scheduler = accelerator.prepare(network, optimizer, dataloader, scheduler)
        training_model = network
    raw_transformer = accelerator.unwrap_model(transformer)
    if trainer.blocks_to_swap:
        raw_transformer.move_to_device_except_swap_blocks(accelerator.device)
        raw_transformer.prepare_block_swap_before_forward()
    accelerator.register_for_checkpointing(progress)

    def save_progress(models, weights, directory):
        progress.timestep_range_pool = list(trainer.timestep_range_pool)
        progress.write(directory, accelerator.process_index)
        if accelerator.is_main_process:
            progress.write(directory)

    accelerator.register_save_state_pre_hook(save_progress)
    if args.resume:
        accelerator.load_state(args.resume)
        progress.load_state_dict(read_progress(args.resume, accelerator.process_index))
        trainer.timestep_range_pool = list(progress.timestep_range_pool)
    if progress.next_batch >= len(dataloader):
        raise ValueError("MOSS resume data-loader position is outside this epoch")
    init_kwargs = toml.load(args.log_tracker_config) if args.log_tracker_config else {}
    if args.wandb_run_name:
        init_kwargs.setdefault("wandb", {})["name"] = args.wandb_run_name
    accelerator.init_trackers(
        args.log_tracker_name or ("fine-tuning" if full_finetune else "network_train"),
        config=train_utils.get_sanitized_config_or_none(args),
        init_kwargs=init_kwargs,
    )
    noise_scheduler = trainer.get_noise_scheduler(args)
    network_dtype = dit_dtype if full_finetune else torch.float32
    save_dtype = model_utils.str_to_dtype(args.save_precision) if args.save_precision else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_sampled_step = None

    def sample(epoch):
        nonlocal last_sampled_step
        if not should_sample_images(args, progress.global_step, epoch) or samples is None:
            return
        was_training = network.training if network is not None else None
        try:
            if network is not None:
                network.eval()
            trainer.sample_images(accelerator, args, epoch, progress.global_step, vae, transformer, samples, dit_dtype)
        finally:
            if network is not None:
                network.train(was_training)
        last_sampled_step = progress.global_step

    def export(name, force_upload=False):
        accelerator.wait_for_everyone()
        raw_transformer.wait_for_pending_block_moves()
        if not accelerator.is_main_process:
            return
        metadata = _metadata(args, progress, optimizer_name, started, full_finetune)
        path = output_dir / name
        if full_finetune:
            save_full_model(raw_transformer, path, save_dtype, metadata)
        else:
            accelerator.unwrap_model(network).save_weights(str(path), save_dtype or torch.float32, metadata)
        if args.huggingface_repo_id:
            huggingface_utils.upload(args, str(path), "/" + name, force_sync_upload=force_upload)
        trainer.on_post_save(args, accelerator, network, transformer, name, save_dtype, metadata, force_upload)

    progress_bar = tqdm(
        total=args.max_train_steps, initial=progress.global_step, desc="steps", disable=not accelerator.is_local_main_process
    )
    try:
        if progress.global_step == 0:
            optimizer_eval()
            sample(0)
        optimizer_train()
        optimizer.zero_grad(set_to_none=True)
        while progress.global_step < args.max_train_steps:
            epoch = progress.epoch
            current_epoch.value = epoch + 1
            loader_generator.manual_seed(args.seed + epoch)
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            if network is not None:
                accelerator.unwrap_model(network).on_epoch_start(transformer)
            start_batch = progress.next_batch
            epoch_loader = accelerator.skip_first_batches(dataloader, start_batch) if start_batch else dataloader
            completed_epoch = False
            for batch_index, batch in enumerate(epoch_loader, start=start_batch):
                with accelerator.accumulate(training_model):
                    if network is not None:
                        accelerator.unwrap_model(network).on_step_start()
                    latents = trainer.scale_shift_latents(batch["latents"])
                    noise = torch.randn_like(latents)
                    with accelerator.autocast():
                        loss, metrics = trainer.process_batch(
                            args,
                            accelerator,
                            transformer,
                            network,
                            batch,
                            latents,
                            noise,
                            noise_scheduler,
                            dit_dtype,
                            network_dtype,
                            vae,
                            progress.global_step,
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("MOSS training loss is not finite")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if network is not None and accelerator.num_processes > 1:
                            _synchronize_lora_gradients(network, accelerator)
                        if full_finetune and trainer.blocks_to_swap and not fused:
                            _move_full_gradients(raw_transformer, optimizer)
                        if args.max_grad_norm and not fused:
                            accelerator.clip_grad_norm_(training_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                progress.next_batch = batch_index + 1
                completed_epoch = progress.next_batch == len(dataloader)
                if completed_epoch:
                    progress.epoch = epoch + 1
                    progress.next_batch = 0
                if not accelerator.sync_gradients:
                    continue
                progress.global_step += 1
                trainer.on_post_optimizer_step(args, accelerator, network, transformer, True, progress.global_step)
                if network is not None and args.scale_weight_norms:
                    accelerator.unwrap_model(network).apply_max_norm_regularization(args.scale_weight_norms, accelerator.device)
                progress_bar.update(1)
                progress_bar.set_postfix(loss=float(loss.detach()))
                logs = trainer.generate_step_logs(
                    args, loss.detach().item(), loss.detach().item(), scheduler, lr_descriptions, optimizer, None, None, None
                )
                logs.update(metrics)
                if accelerator.trackers:
                    accelerator.log(logs, step=progress.global_step)
                should_save = args.save_every_n_steps and progress.global_step % args.save_every_n_steps == 0
                should_sample = should_sample_images(args, progress.global_step, None)
                if should_save or should_sample:
                    optimizer_eval()
                    sample(None)
                    if should_save:
                        export(train_utils.get_step_ckpt_name(args.output_name, progress.global_step))
                        if args.save_state:
                            _save_state(args, accelerator, "step", progress.global_step)
                        remove_step = train_utils.get_remove_step_no(args, progress.global_step)
                        if accelerator.is_main_process and remove_step is not None:
                            _remove_output(output_dir, train_utils.get_step_ckpt_name(args.output_name, remove_step))
                    optimizer_train()
                if progress.global_step >= args.max_train_steps:
                    break
            if completed_epoch:
                trainer.on_epoch_end(args, accelerator, network, transformer, progress.epoch)
                optimizer_eval()
                if args.save_every_n_epochs and progress.epoch % args.save_every_n_epochs == 0:
                    export(train_utils.get_epoch_ckpt_name(args.output_name, progress.epoch))
                    if args.save_state:
                        _save_state(args, accelerator, "epoch", progress.epoch)
                    remove_epoch = train_utils.get_remove_epoch_no(args, progress.epoch)
                    if accelerator.is_main_process and remove_epoch is not None:
                        _remove_output(output_dir, train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch))
                if last_sampled_step != progress.global_step:
                    sample(progress.epoch)
                optimizer_train()
        optimizer_eval()
        export(train_utils.get_last_ckpt_name(args.output_name), force_upload=True)
        if args.save_state or args.save_state_on_train_end:
            _save_state(args, accelerator, "last", progress.global_step)
        return {
            "global_step": progress.global_step,
            "checkpoint": str(output_dir / train_utils.get_last_ckpt_name(args.output_name)),
        }
    finally:
        progress_bar.close()
        for handle in fused_handles:
            handle.remove()
        accelerator.end_training()

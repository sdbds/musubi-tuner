from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from accelerate import Accelerator


DOPSD_TEACHER_EMBED_KEY = "dopsd_teacher_llm_embed"
DOPSD_ZIMAGE_TEACHER_EMBED_KEY = DOPSD_TEACHER_EMBED_KEY
DOPSD_FLUX2_TEACHER_EMBED_KEY = "dopsd_teacher_ctx_vec"


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_dopsd_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if not _parser_has_option(parser, "--dopsd"):
        parser.add_argument("--dopsd", action="store_true", help="Enable experimental D-OPSD distillation")
    if not _parser_has_option(parser, "--dopsd_loss_weight"):
        parser.add_argument("--dopsd_loss_weight", type=float, default=1.0, help="Weight for the D-OPSD distillation loss")
    if not _parser_has_option(parser, "--dopsd_num_sampling_steps"):
        parser.add_argument(
            "--dopsd_num_sampling_steps",
            type=int,
            default=8,
            help="Few-step schedule length used for D-OPSD on-policy rollouts",
        )
    if not _parser_has_option(parser, "--dopsd_ema_decay"):
        parser.add_argument(
            "--dopsd_ema_decay",
            type=float,
            default=0.9999,
            help="EMA decay for the teacher trainable weights. 1.0 freezes the initial trainable weights as teacher.",
        )
    return parser


def validate_dopsd_args(args: argparse.Namespace) -> None:
    if not getattr(args, "dopsd", False):
        return
    if getattr(args, "dopsd_loss_weight", 0.0) <= 0:
        raise ValueError("--dopsd_loss_weight must be positive")
    if getattr(args, "dopsd_num_sampling_steps", 0) < 1:
        raise ValueError("--dopsd_num_sampling_steps must be at least 1")
    ema_decay = getattr(args, "dopsd_ema_decay", 0.0)
    if ema_decay < 0.0 or ema_decay > 1.0:
        raise ValueError("--dopsd_ema_decay must be between 0.0 and 1.0")


def is_dopsd_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dopsd", False) and getattr(args, "dopsd_loss_weight", 0.0) > 0.0)


def get_named_trainable_parameters(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in module.named_parameters() if param.requires_grad]


class AdapterEma:
    def __init__(
        self,
        module: torch.nn.Module,
        shadow_device: torch.device | str | None = None,
        backup_device: torch.device | str | None = None,
    ):
        named_params = get_named_trainable_parameters(module)
        if not named_params:
            raise ValueError("D-OPSD requires at least one trainable parameter for EMA teacher")
        self.shadow_device = torch.device(shadow_device) if shadow_device is not None else None
        self.backup_device = torch.device(backup_device) if backup_device is not None else None
        self.shadow = {name: self._clone_for_shadow(param) for name, param in named_params}

    def _clone_for_shadow(self, param: torch.nn.Parameter) -> torch.Tensor:
        tensor = param.detach()
        if self.shadow_device is not None:
            tensor = tensor.to(device=self.shadow_device)
        return tensor.clone()

    def _clone_for_backup(self, param: torch.nn.Parameter) -> torch.Tensor:
        tensor = param.detach()
        if self.backup_device is not None:
            tensor = tensor.to(device=self.backup_device)
        return tensor.clone()

    def update(self, module: torch.nn.Module, decay: float) -> None:
        with torch.no_grad():
            for name, param in get_named_trainable_parameters(module):
                if name not in self.shadow:
                    self.shadow[name] = self._clone_for_shadow(param)
                    continue
                shadow = self.shadow[name]
                shadow_device = self.shadow_device if self.shadow_device is not None else param.device
                if shadow.device != shadow_device or shadow.dtype != param.dtype:
                    shadow = shadow.to(device=shadow_device, dtype=param.dtype)
                    self.shadow[name] = shadow
                current = param.detach().to(device=shadow.device, dtype=shadow.dtype)
                shadow.mul_(decay).add_(current, alpha=1.0 - decay)

    @contextmanager
    def use_ema_weights(self, module: torch.nn.Module) -> Iterator[None]:
        named_params = dict(get_named_trainable_parameters(module))
        if self._can_swap_without_copy(named_params):
            with self._swap_ema_weights(module, named_params):
                yield
            return

        backups: dict[str, torch.Tensor] = {}
        was_training = module.training
        with torch.no_grad():
            for name, shadow in self.shadow.items():
                param = named_params.get(name)
                if param is None:
                    continue
                backups[name] = self._clone_for_backup(param)
                param.copy_(shadow.to(device=param.device, dtype=param.dtype))
            module.eval()
        try:
            yield
        finally:
            with torch.no_grad():
                named_params = dict(get_named_trainable_parameters(module))
                for name, backup in backups.items():
                    param = named_params[name]
                    param.copy_(backup.to(device=param.device, dtype=param.dtype))
                module.train(was_training)

    def _can_swap_without_copy(self, named_params: dict[str, torch.nn.Parameter]) -> bool:
        if self.backup_device is not None:
            return False

        for name, shadow in self.shadow.items():
            param = named_params.get(name)
            if param is None:
                continue
            if shadow.device != param.device or shadow.dtype != param.dtype or shadow.shape != param.shape:
                return False
        return True

    @contextmanager
    def _swap_ema_weights(self, module: torch.nn.Module, named_params: dict[str, torch.nn.Parameter]) -> Iterator[None]:
        swapped_names: list[str] = []
        was_training = module.training
        with torch.no_grad():
            for name, shadow in self.shadow.items():
                param = named_params.get(name)
                if param is None:
                    continue
                student_data = param.data
                param.data = shadow
                self.shadow[name] = student_data
                swapped_names.append(name)
            module.eval()
        try:
            yield
        finally:
            with torch.no_grad():
                named_params = dict(get_named_trainable_parameters(module))
                for name in swapped_names:
                    param = named_params[name]
                    ema_data = param.data
                    param.data = self.shadow[name]
                    self.shadow[name] = ema_data
                module.train(was_training)


DopsdPredictFn = Callable[[dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]
DopsdTeacherBatchFn = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
DopsdRolloutStepFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
DopsdLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


def dopsd_velocity_loss(
    state: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    del state, sigmas, step_index
    return F.mse_loss(student_pred.float(), teacher_pred.float().detach(), reduction="mean")


def dopsd_x0_loss(
    state: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    sigma = sigmas[step_index].to(device=state.device, dtype=torch.float32)
    state = state.float()
    x0_student = state + sigma * student_pred.float()
    x0_teacher = state + sigma * teacher_pred.float()
    return F.mse_loss(x0_student, x0_teacher.detach(), reduction="mean")


def dopsd_flow_x0_loss(
    state: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    sigma = sigmas[step_index].to(device=state.device, dtype=torch.float32)
    state = state.float()
    x0_student = state - sigma * student_pred.float()
    x0_teacher = state - sigma * teacher_pred.float()
    return F.mse_loss(x0_student, x0_teacher.detach(), reduction="mean")


def run_dopsd_stepwise_backward(
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    network: torch.nn.Module,
    ema: AdapterEma,
    batch: dict[str, torch.Tensor],
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    predict_fn: DopsdPredictFn,
    make_teacher_batch_fn: DopsdTeacherBatchFn,
    rollout_step_fn: DopsdRolloutStepFn,
    loss_fn: DopsdLossFn | None = None,
) -> tuple[torch.Tensor, int]:
    teacher_batch = make_teacher_batch_fn(batch)
    if loss_fn is None:
        loss_fn = dopsd_velocity_loss
    device = accelerator.device
    state_dtype = latents.dtype if latents.is_floating_point() else torch.float32
    state = torch.randn(latents.shape, device=device, dtype=state_dtype)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    step_count = int(timesteps.shape[0])
    batch_size = int(latents.shape[0])
    unwrapped_network = accelerator.unwrap_model(network)

    for step_index in range(step_count):
        step_timesteps = timesteps[step_index].expand(batch_size).to(device=device, dtype=torch.float32)

        with ema.use_ema_weights(unwrapped_network):
            with torch.inference_mode():
                teacher_pred = predict_fn(teacher_batch, state, step_timesteps)
        teacher_pred = teacher_pred.clone().detach()

        student_pred = predict_fn(batch, state, step_timesteps)
        step_loss = loss_fn(state, student_pred, teacher_pred, sigmas, step_index)
        scaled_loss = step_loss * (float(args.dopsd_loss_weight) / float(step_count))
        accelerator.backward(scaled_loss)

        total_loss = total_loss + step_loss.detach()

        if step_index + 1 < step_count:
            with torch.no_grad():
                state = rollout_step_fn(state, student_pred.detach(), sigmas, step_index).detach()

    return total_loss / float(step_count), step_count


def update_dopsd_metadata(metadata: dict, args: argparse.Namespace, train_mode: str | None = None) -> None:
    if not getattr(args, "dopsd", False):
        return
    metadata["ss_dopsd"] = bool(getattr(args, "dopsd", False))
    metadata["ss_dopsd_loss_weight"] = getattr(args, "dopsd_loss_weight", None)
    metadata["ss_dopsd_num_sampling_steps"] = getattr(args, "dopsd_num_sampling_steps", None)
    metadata["ss_dopsd_ema_decay"] = getattr(args, "dopsd_ema_decay", None)
    if train_mode is not None:
        metadata["ss_dopsd_train_mode"] = train_mode

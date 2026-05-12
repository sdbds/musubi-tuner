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
DOPSD_FLUX2_IDENTITY_EDIT_PROMPT = (
    "Reconstruct the reference image exactly. Do not change its content, composition, style, or colors."
)


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


def add_dopsd_full_finetune_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if not _parser_has_option(parser, "--dopsd_full_ema_device"):
        parser.add_argument(
            "--dopsd_full_ema_device",
            type=str,
            default="auto",
            choices=("cpu", "gpu", "auto"),
            help=(
                "EMA teacher storage for full-parameter D-OPSD. "
                "'auto' uses GPU only when free CUDA memory appears sufficient, "
                "'cpu' uses less VRAM and is slower, 'gpu' keeps a full EMA copy on GPU and is faster."
            ),
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


def resolve_full_ema_devices(
    args: argparse.Namespace,
    module: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.device | None, torch.device | None, str]:
    requested = getattr(args, "dopsd_full_ema_device", "auto")
    if requested == "gpu":
        if device.type != "cuda":
            raise ValueError("--dopsd_full_ema_device=gpu requires a CUDA training device")
        return None, None, "gpu"
    if requested == "auto" and _has_cuda_memory_for_gpu_ema(module, device):
        return None, None, "gpu-auto"
    return torch.device("cpu"), torch.device("cpu"), "cpu" if requested != "auto" else "cpu-auto"


def _has_cuda_memory_for_gpu_ema(module: torch.nn.Module, device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    trainable_bytes = sum(param.numel() * param.element_size() for _, param in get_named_trainable_parameters(module))
    if trainable_bytes <= 0:
        return False
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        return False
    # A full GPU EMA needs one extra parameter copy plus allocator slack.
    return free_bytes > int(trainable_bytes * 1.25)


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
        self._named_params = named_params
        self._param_lookup = dict(named_params)
        self.shadow = {name: self._clone_for_shadow(param) for name, param in named_params}

    def _refresh_named_params(self, module: torch.nn.Module) -> None:
        named_params = get_named_trainable_parameters(module)
        if len(named_params) != len(self._named_params) or any(
            name != old_name or param is not old_param
            for (name, param), (old_name, old_param) in zip(named_params, self._named_params)
        ):
            self._named_params = named_params
            self._param_lookup = dict(named_params)

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
            self._refresh_named_params(module)
            for name, param in self._named_params:
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
        if self._can_swap_without_copy():
            with self._swap_ema_weights(module):
                yield
            return

        backups: dict[str, torch.Tensor] = {}
        was_training = module.training
        with torch.no_grad():
            for name, shadow in self.shadow.items():
                param = self._param_lookup.get(name)
                if param is None:
                    continue
                backups[name] = self._clone_for_backup(param)
                param.copy_(shadow.to(device=param.device, dtype=param.dtype))
            module.eval()
        try:
            yield
        finally:
            with torch.no_grad():
                for name, backup in backups.items():
                    param = self._param_lookup[name]
                    param.copy_(backup.to(device=param.device, dtype=param.dtype))
                module.train(was_training)

    def _can_swap_without_copy(self) -> bool:
        if self.backup_device is not None:
            return False

        for name, shadow in self.shadow.items():
            param = self._param_lookup.get(name)
            if param is None:
                continue
            if shadow.device != param.device or shadow.dtype != param.dtype or shadow.shape != param.shape:
                return False
        return True

    @contextmanager
    def _swap_ema_weights(self, module: torch.nn.Module) -> Iterator[None]:
        swapped_names: list[str] = []
        was_training = module.training
        with torch.no_grad():
            for name, shadow in self.shadow.items():
                param = self._param_lookup.get(name)
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
                for name in swapped_names:
                    param = self._param_lookup[name]
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
    return dopsd_sigma_weighted_velocity_loss(state, student_pred, teacher_pred, sigmas, step_index)


def dopsd_flow_x0_loss(
    state: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    return dopsd_sigma_weighted_velocity_loss(state, student_pred, teacher_pred, sigmas, step_index)


def dopsd_sigma_weighted_velocity_loss(
    state: torch.Tensor,
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    del state
    sigma = sigmas[step_index].to(device=student_pred.device, dtype=torch.float32)
    velocity_loss = F.mse_loss(student_pred.float(), teacher_pred.float().detach(), reduction="mean")
    return velocity_loss * sigma.square()


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
    if hasattr(args, "dopsd_full_ema_device"):
        metadata["ss_dopsd_full_ema_device"] = getattr(args, "dopsd_full_ema_device", None)
    if train_mode is not None:
        metadata["ss_dopsd_train_mode"] = train_mode

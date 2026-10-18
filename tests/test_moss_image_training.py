from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.moss_image.model import MossImage
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, setup_parser
from musubi_tuner.networks import lora_moss_image
from musubi_tuner.training.trainer_base import DiTOutput


CONFIG = dict(dim=32, ffn_dim=48, num_heads=4, num_layers=3, text_dim=4096, freq_dim=16, latent_channels=128)


def tiny_model():
    torch.manual_seed(41)
    return MossImage(**CONFIG)


def args_for_test(*extra):
    return setup_parser().parse_args(["--sdpa", "--mixed_precision", "no", *extra])


def batch_for_test():
    return {
        "latents": torch.randn(2, 128, 2, 2),
        "moss_embed": [torch.randn(3, 4096, dtype=torch.bfloat16), torch.randn(5, 4096, dtype=torch.bfloat16)],
        "moss_uncond": [torch.randn(2, 4096, dtype=torch.bfloat16)] * 2,
        "timesteps": [0.25, 0.75],
    }


def accelerator_stub():
    return SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext, unwrap_model=lambda model: model)


@pytest.mark.parametrize("sampling", ["uniform", "sigmoid", "shift", "uniform_shift", "logsnr"])
def test_moss_timestep_matches_actual_noise_coefficient(sampling):
    trainer = MossImageNetworkTrainer()
    args = args_for_test("--timestep_sampling", sampling)
    trainer.handle_model_specific_args(args)
    trainer.num_timestep_buckets = 2
    clean = torch.zeros(2, 128, 2, 2)
    noise = torch.ones_like(clean)
    noisy, times = trainer.get_noisy_model_input_and_timesteps(
        args, noise, clean, [0.25, 0.75321], None, torch.device("cpu"), torch.bfloat16
    )
    assert times.dtype == torch.float32
    torch.testing.assert_close(times / 1000, noisy[:, 0, 0, 0], rtol=1e-6, atol=1e-7)
    assert noisy.dtype == torch.float32


def test_loss_is_fp32_and_weights_use_actual_sigma():
    trainer = MossImageNetworkTrainer()
    args = args_for_test("--weighting_scheme", "sigma_sqrt")
    prediction = torch.ones(2, 128, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    output = DiTOutput(prediction, torch.zeros_like(prediction))
    losses = trainer.compute_per_sample_loss(args, output, torch.tensor([250.0, 500.0]), None, torch.bfloat16, torch.bfloat16, 0)
    assert losses.dtype == torch.float32
    torch.testing.assert_close(losses, torch.tensor([16.0, 4.0]))
    losses.mean().backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("train_embedder", [False, True])
def test_lora_update_and_weight_roundtrip(tmp_path, train_embedder):
    model = tiny_model().requires_grad_(False)
    state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    adapter = lora_moss_image.create_arch_network(1.0, 2, 2.0, None, None, model, train_text_embedder=str(train_embedder).lower())
    names = [module.lora_name for module in adapter.unet_loras]
    assert any(name.startswith("lora_unet_layers_0_proj_q") for name in names)
    assert any(name.startswith("lora_unet_text_embedder_") for name in names) == train_embedder
    adapter.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    trainer = MossImageNetworkTrainer()
    args = args_for_test()
    trainer.handle_model_specific_args(args)
    batch = batch_for_test()
    latents = batch["latents"]
    noise = torch.randn_like(latents)
    times = torch.tensor([250.0, 750.0])
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    before = trainer.call_dit(args, accelerator_stub(), model, latents, batch, noise, latents, times, torch.float32)
    torch.testing.assert_close(before.target, noise - latents)
    before.pred.square().mean().backward()
    optimizer.step()
    after = trainer.call_dit(args, accelerator_stub(), model, latents, batch, noise, latents, times, torch.float32).pred.detach()
    assert not torch.equal(before.pred, after)
    assert all(torch.equal(tensor, state[key]) for key, tensor in model.state_dict().items())
    path = str(tmp_path / "lora.safetensors")
    adapter.save_weights(path, torch.float32, {"ss_base_model_version": "moss_image"})
    weights = load_file(path)
    reloaded = tiny_model().requires_grad_(False)
    loaded_adapter = lora_moss_image.create_arch_network_from_weights(1.0, weights, unet=reloaded)
    loaded_adapter.apply_to(None, reloaded, apply_text_encoder=False, apply_unet=True)
    loaded_adapter.load_state_dict(weights, strict=True)
    actual = trainer.call_dit(args, accelerator_stub(), reloaded, latents, batch, noise, latents, times, torch.float32).pred
    torch.testing.assert_close(actual, after)


def test_text_encoder_and_unknown_lora_weights_rejected():
    for name in ["lora_te_layers_0_self_attn_q_proj", "lora_unet_layers_99_proj_q"]:
        weights = {f"{name}.lora_down.weight": torch.ones(2, 32), f"{name}.lora_up.weight": torch.ones(32, 2)}
        with pytest.raises(ValueError):
            lora_moss_image.create_arch_network_from_weights(1.0, weights, unet=tiny_model())


def test_sampling_exception_restores_training_mode(monkeypatch, tmp_path):
    trainer = MossImageNetworkTrainer()
    args = args_for_test()
    trainer.handle_model_specific_args(args)
    model = tiny_model().train()
    vae = torch.nn.Linear(2, 2)

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(trainer, "do_inference", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        trainer.sample_image_inference(
            accelerator_stub(),
            args,
            model,
            torch.float32,
            vae,
            str(tmp_path),
            {"width": 32, "height": 32, "seed": 1},
            0,
            0,
        )
    assert model.training


def test_moss_metadata_identifies_adapter_and_full_model():
    from musubi_tuner.utils.sai_model_spec import build_metadata

    for is_lora in [True, False]:
        metadata = build_metadata(None, "moss", 0, "test", is_lora=is_lora)
        assert metadata["modelspec.architecture"] == "moss-image-base" + ("/lora" if is_lora else "")


def test_sigma_training_respects_requested_flow_shift(monkeypatch):
    from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
    from musubi_tuner.training import trainer_base

    trainer = MossImageNetworkTrainer()
    args = args_for_test("--timestep_sampling", "sigma", "--discrete_flow_shift", "5")
    trainer.handle_model_specific_args(args)
    monkeypatch.setattr(trainer_base, "compute_density_for_timestep_sampling", lambda **kwargs: torch.tensor([0.5]))
    clean = torch.zeros(1, 128, 2, 2)
    noisy, times = trainer.get_noisy_model_input_and_timesteps(
        args, torch.ones_like(clean), clean, None, FlowMatchDiscreteScheduler(shift=5), torch.device("cpu"), torch.bfloat16
    )
    torch.testing.assert_close(noisy, torch.full_like(noisy, 5 / 6))
    torch.testing.assert_close(times, torch.tensor([1000 * 5 / 6]))


def test_resolution_shift_counts_already_patchified_tokens():
    trainer = MossImageNetworkTrainer()
    assert trainer.latent_tokens_for_timestep_sampling(torch.zeros(1, 128, 64, 96)) == 6144


@pytest.mark.parametrize(
    "flag", [["--log_grad_metrics"], ["--show_timesteps", "console"], ["--gradient_checkpointing_cpu_offload"]]
)
def test_moss_rejects_unsupported_or_incomplete_flags(flag):
    with pytest.raises(ValueError):
        MossImageNetworkTrainer().handle_model_specific_args(args_for_test(*flag))


@pytest.mark.parametrize("full", [False, True])
def test_no_metadata_keeps_only_loading_contract(full):
    from musubi_tuner.moss_image.training import _metadata

    args = args_for_test("--no_metadata", "--metadata_author", "Private Author", "--dit", "private-model.safetensors")
    metadata = _metadata(args, SimpleNamespace(global_step=17, epoch=3), "AdamW", 1, full)
    assert metadata["ss_base_model_version"] == "moss_image"
    assert "ss_seed" not in metadata and "ss_sd_model_name" not in metadata
    assert "modelspec.author" not in metadata
    if not full:
        assert metadata["ss_network_module"] == args.network_module


def _distributed_gradient_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as distributed
    from musubi_tuner.moss_image.training import _synchronize_lora_gradients

    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        network = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(3)])
        network[rank].grad = torch.full_like(network[rank], 2.0 if rank == 0 else 6.0)

        def reduce(tensor, reduction):
            result = tensor.clone()
            distributed.all_reduce(result)
            return result / 2 if reduction == "mean" else result

        _synchronize_lora_gradients(network, SimpleNamespace(device=torch.device("cpu"), reduce=reduce))
        torch.testing.assert_close(network[0].grad, torch.ones(2))
        torch.testing.assert_close(network[1].grad, torch.full((2,), 3.0))
        assert network[2].grad is None
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="Gloo required")
def test_distributed_lora_handles_different_unused_parameters(tmp_path):
    import time
    from musubi_tuner.moss_image.training import _synchronize_lora_gradients

    assert callable(_synchronize_lora_gradients)
    context = torch.multiprocessing.spawn(
        _distributed_gradient_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=max(1, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                raise TimeoutError("MOSS gradient collectives did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)

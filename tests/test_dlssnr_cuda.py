"""Opt-in real-weight CUDA smoke, separate from numerical/native quality acceptance."""

import os

import pytest
import torch

from musubi_tuner.dlssnr.checkpoint import load_source, unpack_record
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.networks.lora_dlssnr import base_target_sha256, inject, merge_adapter
from musubi_tuner.training.dlssnr_services import assert_finite_gradients
from musubi_tuner.training.dlssnr_trainer import NRTrainModule, build_optimizer, clear_lane15_state
from test_dlssnr_packing import _source_dir


pytestmark = pytest.mark.skipif(
    os.environ.get("DLSSNR_RUN_CUDA_SMOKE") != "1" or not torch.cuda.is_available() or _source_dir() is None,
    reason="set DLSSNR_RUN_CUDA_SMOKE=1 with CUDA and the original 310.8.0 source weights available",
)


@pytest.fixture(scope="module")
def canonical_weights():
    source = load_source(_source_dir())
    tensors = {}
    for record in source.records:
        tensors.update(
            {
                name: torch.from_numpy(value)
                for name, value in unpack_record(record, source.blobs[record.name]).items()
                if not name.startswith("opaque.")
            }
        )
    return tensors


@pytest.mark.parametrize("lora", [False, True])
def test_original_weights_complete_a_finite_cuda_update(canonical_weights, lora):
    with fp32_execution():
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        model = NRModel()
        model.load_state_dict(canonical_weights, strict=True)
        model.cuda()
        network = inject(model, {"profile": "vit_only", "rank": 16, "alpha": 16, "dropout": 0.0}) if lora else None
        if lora:
            identity = base_target_sha256(model, network.target_names)
            optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=0)
        else:
            optimizer = build_optimizer(model, 1e-5, {"priors": 0.1, "scales": 0.1}, 0.0)
        module = NRTrainModule(model, {"pre": 1.0, "out": 1.0, "edge": 0.05}, network=network)
        assert all(parameter.device.type == "cuda" for parameter in module.parameters())
        source = torch.linspace(0.2, 0.8, 48 * 48, device="cuda").reshape(1, 1, 48, 48).expand(1, 3, 48, 48)
        controls = torch.zeros(1, 5, 48, 48, device="cuda")
        controls[:, 1] = 0.5
        batch = {"source": source, "target": (source * 0.9).clone(), "controls": controls}
        before = model.blocks["70"].head.rgb.weight.detach().clone()
        loss, metrics = module(batch, [123])
        loss.backward()
        assert_finite_gradients(module)
        if lora:
            assert all(
                adapter.lora_up.grad is not None and torch.isfinite(adapter.lora_up.grad).all() for adapter in network.adapters
            )
            assert all(torch.count_nonzero(adapter.lora_up.grad) > 0 for adapter in network.adapters)
        model.enforce_lane15()
        optimizer.step()
        clear_lane15_state(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        if lora:
            assert base_target_sha256(model, network.target_names) == identity
            fresh = NRModel().cuda()
            fresh.load_state_dict(merge_adapter(model.state_dict(), network))
            with torch.no_grad():
                original = forward_frame(model, source, controls, [123])
                merged = forward_frame(fresh, source, controls, [123])
            torch.testing.assert_close(merged["raw_head"], original["raw_head"], rtol=2e-4, atol=2e-4)
        else:
            assert not torch.equal(before, model.blocks["70"].head.rgb.weight)
        assert torch.count_nonzero(model.blocks["0"].input_adapter.weight[:, 15]) == 0
        print({"lora": lora, "loss": metrics["loss"], "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)})

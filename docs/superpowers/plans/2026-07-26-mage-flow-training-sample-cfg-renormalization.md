# Mage-Flow Training-Sample CFG Renormalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Always enable Mage-Flow's existing CFG renormalization for preview images generated during training.

**Architecture:** Keep the reusable Mage-Flow sampler and standalone generation CLI unchanged. Select the existing behavior only at the Mage-Flow training-preview call site by passing `renormalize_cfg=True`.

**Tech Stack:** Python, PyTorch, pytest

## Global Constraints

- Change only Mage-Flow training-preview behavior.
- Add no training CLI or per-prompt parameter.
- Do not change the shared sampling-prompt parser or other architectures.
- Keep standalone Mage-Flow generation and direct `sample_latents` defaults unchanged.

---

### Task 1: Enable renormalization for Mage-Flow training previews

**Files:**
- Modify: `src/musubi_tuner/mage_flow_train_network.py:221-233`
- Test: `tests/test_mage_flow_training.py`

**Interfaces:**
- Consumes: `sample_latents(..., renormalize_cfg: bool = False) -> list[torch.Tensor]`
- Produces: `MageFlowNetworkTrainer.do_inference` always invokes that interface with `renormalize_cfg=True`

- [ ] **Step 1: Write the failing forwarding test**

Add this focused test to `tests/test_mage_flow_training.py`:

```python
def test_training_preview_always_enables_mage_cfg_renormalization(monkeypatch):
    captured = {}

    class FakeVAE:
        dtype = torch.float32

        def to(self, _device):
            return self

        def decode(self, latents):
            return latents

    def fake_sample_latents(*_args, **kwargs):
        captured.update(kwargs)
        return [torch.zeros(3, 2, 2)]

    monkeypatch.setattr(train_module, "sample_latents", fake_sample_latents)
    monkeypatch.setattr(train_module, "clean_memory_on_device", lambda _device: None)

    MageFlowNetworkTrainer().do_inference(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(),
        {"mage_flow_embed": torch.zeros(2, 5)},
        FakeVAE(),
        torch.float32,
        object(),
        6.0,
        2,
        32,
        32,
        1,
        torch.Generator(device="cpu").manual_seed(42),
        False,
        1.0,
        1.0,
    )

    assert captured.get("renormalize_cfg") is True
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mage_flow_training.py::test_training_preview_always_enables_mage_cfg_renormalization -q
```

Expected: FAIL because the captured `sample_latents` call has no `renormalize_cfg` key.

- [ ] **Step 3: Implement the minimal call-site change**

In `MageFlowNetworkTrainer.do_inference`, extend the existing `sample_latents` call:

```python
        latents = sample_latents(
            transformer,
            [positive],
            [(height // 16, width // 16)],
            steps=sample_steps,
            seeds=[generator.initial_seed()],
            device=device,
            dtype=dit_dtype,
            controls=controls,
            negative_text_tokens=negative,
            cfg_scale=cfg_scale if negative is not None else 1.0,
            shift=discrete_flow_shift,
            renormalize_cfg=True,
        )
```

- [ ] **Step 4: Run focused and Mage-Flow regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mage_flow_training.py tests/test_mage_flow_entrypoints.py tests/test_uniform_shift_timesteps.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run repository regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: no new failures. If the known unrelated Ideogram caption-cache test still fails, record it separately and rerun the suite excluding that test to prove this change introduces no additional failures.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
git diff --check
git status --short
git diff -- src/musubi_tuner/mage_flow_train_network.py tests/test_mage_flow_training.py
git add docs/superpowers/plans/2026-07-26-mage-flow-training-sample-cfg-renormalization.md src/musubi_tuner/mage_flow_train_network.py tests/test_mage_flow_training.py
git commit -m "fix: renormalize Mage training preview CFG"
```

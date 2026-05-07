# Z-Image SOAR-lite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental `--soar` training path for `zimage` that uses ODE cond-only self-correction, preserves current behavior when disabled, and exposes only real v1 knobs with official-aligned defaults.

**Architecture:** Keep SOAR math in a small helper module and keep `zimage`-specific tensor preparation in `zimage_train_network.py`. Extract the auxiliary pass into a narrow helper in `zimage_train.py` so the main loop only branches once, and use stdlib `unittest` for new tests because the repo currently has no existing `tests/` tree or `pytest` dependency. Run verification inside the project's existing managed Python environment rather than baking a specific `uv --extra` CUDA profile into the plan.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, stdlib `unittest`, Ruff.

---

## File Structure

**Create:**

- `src/musubi_tuner/soar_utils.py`
- `tests/__init__.py`
- `tests/test_soar_utils.py`
- `tests/test_zimage_soar_args.py`
- `tests/test_zimage_predict_velocity.py`
- `tests/test_zimage_soar_aux.py`
- `docs/superpowers/plans/2026-04-21-zimage-soar-implementation.md`

**Modify:**

- `src/musubi_tuner/zimage_train.py`
- `src/musubi_tuner/zimage_train_network.py`
- `docs/zimage.md`

**Why this split:**

- `soar_utils.py` owns pure math and scheduling conversions.
- `zimage_train_network.py` owns reusable forward-only prediction.
- `zimage_train.py` owns parser defaults, validation, and loop integration.
- `tests/` stays small and focused on pure math, parser behavior, and the extracted auxiliary-pass contract.

## Task 1: Add SOAR Helper Skeleton and Pure Math Tests

**Files:**

- Create: `tests/__init__.py`
- Create: `tests/test_soar_utils.py`
- Create: `src/musubi_tuner/soar_utils.py`

- [ ] **Step 1: Write the failing tests for the helper contract**

Add `tests/test_soar_utils.py` with `unittest` cases for:

- `per_point_aux_scale(lambda_aux=1.0, trajectory_length=6) -> 1.0 / 6`
- `per_point_aux_scale(lambda_aux=0.5, trajectory_length=1) -> 0.5`
- `build_single_step_ode_aux_points(...)` returns exactly `trajectory_length` auxiliary points
- returned sigma values stay within the requested valid interval
- returned auxiliary tensors preserve the input latent shape

Example test skeleton:

```python
import unittest
import torch

from musubi_tuner.soar_utils import per_point_aux_scale, build_single_step_ode_aux_points


class TestSoarUtils(unittest.TestCase):
    def test_per_point_aux_scale_normalizes_lambda(self):
        self.assertAlmostEqual(per_point_aux_scale(1.0, 6), 1.0 / 6)

    def test_build_single_step_ode_aux_points_returns_requested_count(self):
        z_t1 = torch.zeros(1, 4, 2, 2)
        z_1 = torch.ones(1, 4, 2, 2)
        points = build_single_step_ode_aux_points(
            z_t1=z_t1,
            z_1=z_1,
            sigma_t1=torch.tensor([0.2]),
            trajectory_length=3,
        )
        self.assertEqual(len(points), 3)
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run:

```bash
python -m unittest discover -s tests -p "test_soar_utils.py" -v
```

Expected:

- import failure because `musubi_tuner.soar_utils` does not exist yet

- [ ] **Step 3: Write the minimal helper implementation**

Create `src/musubi_tuner/soar_utils.py` with:

- `per_point_aux_scale(lambda_aux: float, trajectory_length: int) -> float`
- `build_single_step_ode_aux_points(...) -> list[tuple[torch.Tensor, torch.Tensor]]`

Keep the first version narrow:

- ODE-only
- one rollout path
- no CFG branch
- no SDE branch

- [ ] **Step 4: Run the helper tests to verify they pass**

Run:

```bash
python -m unittest discover -s tests -p "test_soar_utils.py" -v
```

Expected:

- PASS for all helper tests

- [ ] **Step 5: Review the diff and verify scope**

Run:

```bash
git diff --stat -- tests/__init__.py tests/test_soar_utils.py src/musubi_tuner/soar_utils.py
git diff -- tests/__init__.py tests/test_soar_utils.py src/musubi_tuner/soar_utils.py
```

## Task 2: Add Parser Defaults and Early Validation

**Files:**

- Modify: `src/musubi_tuner/zimage_train.py`
- Create: `tests/test_zimage_soar_args.py`

- [ ] **Step 1: Write failing parser/validation tests**

Add `tests/test_zimage_soar_args.py` with `unittest` coverage for:

- `--soar_lambda_aux` default is `1.0`
- `--soar_trajectory_length` default is `6`
- `--soar_num_sampling_steps` default is `40`
- `--soar --fused_backward_pass` is rejected by validation
- `--soar_trajectory_length 0` is rejected by validation

Example skeleton:

```python
import unittest

from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner import zimage_train, zimage_train_network


class TestZImageSoarArgs(unittest.TestCase):
    def build_parser(self):
        parser = setup_parser_common()
        parser = zimage_train_network.zimage_setup_parser(parser)
        parser = zimage_train.zimage_finetune_setup_parser(parser)
        return parser

    def test_soar_defaults(self):
        args = self.build_parser().parse_args([])
        self.assertEqual(args.soar_lambda_aux, 1.0)
        self.assertEqual(args.soar_trajectory_length, 6)
        self.assertEqual(args.soar_num_sampling_steps, 40)
```

- [ ] **Step 2: Run the parser tests to verify they fail**

Run:

```bash
python -m unittest discover -s tests -p "test_zimage_soar_args.py" -v
```

Expected:

- parser has no SOAR arguments yet

- [ ] **Step 3: Implement parser flags and validation**

In `src/musubi_tuner/zimage_train.py`:

- add `--soar`
- add `--soar_lambda_aux` default `1.0`
- add `--soar_trajectory_length` default `6`
- add `--soar_num_sampling_steps` default `40`
- add `validate_soar_args(args)` and call it during startup

Validation rules:

- if `args.soar and args.fused_backward_pass`: raise `ValueError`
- if `args.soar_trajectory_length < 1`: raise `ValueError`
- if `args.soar_num_sampling_steps < 2`: raise `ValueError`

- [ ] **Step 4: Run the parser tests to verify they pass**

Run:

```bash
python -m unittest discover -s tests -p "test_zimage_soar_args.py" -v
```

Expected:

- PASS for defaults and validation coverage

- [ ] **Step 5: Review the diff and verify scope**

Run:

```bash
git diff --stat -- tests/test_zimage_soar_args.py src/musubi_tuner/zimage_train.py
git diff -- tests/test_zimage_soar_args.py src/musubi_tuner/zimage_train.py
```

## Task 3: Refactor Z-Image Forward Path for Reuse

**Files:**

- Modify: `src/musubi_tuner/zimage_train_network.py`
- Create: `tests/test_zimage_predict_velocity.py`

- [ ] **Step 1: Write a failing test for forward-only prediction**

Add `tests/test_zimage_predict_velocity.py` with a fake accelerator and dummy transformer that returns its input. Cover:

- `predict_velocity(...)` returns `[B, C, H, W]`
- `call_dit(...)` still returns `target = latents - noise`

Example test skeleton:

```python
import contextlib
import unittest
import torch

from musubi_tuner.zimage_train_network import ZImageNetworkTrainer


class DummyAccelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return contextlib.nullcontext()
```

- [ ] **Step 2: Run the prediction test to verify it fails**

Run:

```bash
python -m unittest discover -s tests -p "test_zimage_predict_velocity.py" -v
```

Expected:

- missing `predict_velocity(...)` helper

- [ ] **Step 3: Implement the refactor**

In `src/musubi_tuner/zimage_train_network.py`:

- extract `predict_velocity(...)`
- keep `call_dit(...)` as the main-path wrapper
- keep target construction in `call_dit(...)`, not inside `predict_velocity(...)`

Do not change:

- padding logic
- timestep transform logic
- output squeeze shape

- [ ] **Step 4: Run the prediction test to verify it passes**

Run:

```bash
python -m unittest discover -s tests -p "test_zimage_predict_velocity.py" -v
```

Expected:

- PASS for helper shape contract

- [ ] **Step 5: Review the diff and verify scope**

Run:

```bash
git diff --stat -- tests/test_zimage_predict_velocity.py src/musubi_tuner/zimage_train_network.py
git diff -- tests/test_zimage_predict_velocity.py src/musubi_tuner/zimage_train_network.py
```

## Task 4: Extract and Test the Auxiliary Pass Contract

**Files:**

- Modify: `src/musubi_tuner/zimage_train.py`
- Create: `tests/test_zimage_soar_aux.py`

- [ ] **Step 1: Write the failing aux-pass tests**

Add `tests/test_zimage_soar_aux.py` with a narrow contract test around an extracted helper, for example:

- `_run_soar_auxiliary_pass(...)`

Cover:

- predictor is called `trajectory_length` times
- each backward uses `lambda_aux / trajectory_length`
- aux loss aggregation does not depend on the number of points

Example skeleton:

```python
import unittest
import torch

from musubi_tuner.zimage_train import _run_soar_auxiliary_pass


class TestZImageSoarAux(unittest.TestCase):
    def test_auxiliary_pass_normalizes_lambda_by_trajectory_length(self):
        calls = []

        def fake_predict(*args, **kwargs):
            calls.append(1)
            return torch.zeros(1, 4, 2, 2)
```

- [ ] **Step 2: Run the aux-pass test to verify it fails**

Run:

```bash
python -m unittest discover -s tests -p "test_zimage_soar_aux.py" -v
```

Expected:

- missing helper or missing normalization

- [ ] **Step 3: Implement the extracted helper**

In `src/musubi_tuner/zimage_train.py`:

- extract the aux branch into a helper that accepts:
  - main batch tensors
  - already-computed rollout state
  - `predict_velocity_fn`
  - `accelerator`
  - `args`
- process aux points serially
- apply normalized per-point backward scaling

Keep this helper independent from:

- optimizer stepping
- progress bar updates
- checkpoint saving

- [ ] **Step 4: Run the aux-pass test to verify it passes**

Run:

```bash
python -m unittest discover -s tests -p "test_zimage_soar_aux.py" -v
```

Expected:

- PASS for call count and normalization behavior

- [ ] **Step 5: Review the diff and verify scope**

Run:

```bash
git diff --stat -- tests/test_zimage_soar_aux.py src/musubi_tuner/zimage_train.py
git diff -- tests/test_zimage_soar_aux.py src/musubi_tuner/zimage_train.py
```

## Task 5: Wire SOAR Into the Main Training Loop

**Files:**

- Modify: `src/musubi_tuner/zimage_train.py`
- Modify: `src/musubi_tuner/zimage_train_network.py`
- Modify: `src/musubi_tuner/soar_utils.py`

- [ ] **Step 1: Insert the SOAR branch into the loop**

In `src/musubi_tuner/zimage_train.py`:

- keep current path unchanged when `not args.soar`
- when `args.soar`:
  - compute main loss
  - backward main loss
  - build aux points under `no_grad`
  - run extracted aux helper
  - run a single optimizer step

- [ ] **Step 2: Run the focused unit test suite**

Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected:

- PASS for all new SOAR tests

- [ ] **Step 3: Run syntax and lint checks on touched files**

Run:

```bash
python -m compileall src/musubi_tuner/soar_utils.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py
ruff check src/musubi_tuner/soar_utils.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py tests
```

Expected:

- compile succeeds
- Ruff reports no new issues in touched files

- [ ] **Step 4: Review the diff and verify scope**

Run:

```bash
git diff --stat -- src/musubi_tuner/soar_utils.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py tests
git diff -- src/musubi_tuner/soar_utils.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py tests
```

## Task 6: Add Operator-Facing Docs and Smoke Commands

**Files:**

- Modify: `docs/zimage.md`

- [ ] **Step 1: Add a short experimental SOAR section**

Document:

- `--soar`
- `--soar_lambda_aux`
- `--soar_trajectory_length`
- `--soar_num_sampling_steps`
- the smoke-test override `--soar_trajectory_length 1`
- the current limitation: cond-only rollout, no CFG support

- [ ] **Step 2: Review docs locally**

Run:

```bash
rg -n "soar|SOAR" docs/zimage.md
```

Expected:

- one concise experimental section

- [ ] **Step 3: Review the diff and verify scope**

Run:

```bash
git diff --stat -- docs/zimage.md
git diff -- docs/zimage.md
```

## Final Verification Checklist

- [ ] `python -m unittest discover -s tests -p "test_*.py" -v`
- [ ] `python -m compileall src/musubi_tuner/soar_utils.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py`
- [ ] `ruff check src/musubi_tuner/soar_utils.py src/musubi_tuner/zimage_train.py src/musubi_tuner/zimage_train_network.py tests`
- [ ] confirm `--soar` off path is unchanged by reading the final diff
- [ ] confirm `--soar --fused_backward_pass` fails fast
- [ ] confirm docs mention cond-only limitation clearly

## Notes for Execution

- Do not add fake CLI knobs for CFG or stochastic rollout in this batch.
- Do not generalize this plan to `flux2` in the same implementation pass.
- If `trajectory_length` normalization becomes awkward inside the loop, fix the structure first. Do not patch it with ad-hoc scaling in multiple places.
- If the extracted aux helper grows beyond a small, readable function, split out another pure helper rather than deepening the training loop branch.

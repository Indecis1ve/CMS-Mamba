# Paper-Aligned CMS-Mamba Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the partial CMS-Mamba implementation with a code-only, paper-aligned pipeline whose explicit missingness masks reach LMMT, TMM, DTF, training selection, and robustness evaluation.

**Architecture:** Corruption is generated deterministically per sample and represented by explicit `1 = missing` masks. TMM aligns both continuous features and masks; branch-specific DTF controls forward and backward TC-Mamba steps; key-only RoPE fusion, bidirectional TQ-Mamba, pooling, RNL, and an MSE-only head complete the model. Validation-only mean MAE selects checkpoints.

**Tech Stack:** Python 3.10, NumPy, PyTorch, Transformers/BERT, Mamba SSM, PyYAML, and the standard-library `unittest` runner.

## Global Constraints

- Existing checkpoints are intentionally incompatible and must fail strict loading with a clear message.
- Do not download datasets or pretrained weights.
- Do not train, fine-tune, benchmark, run research-data evaluation, or simulate manuscript performance.
- Run only deterministic unit tests on small synthetic arrays/tensors plus syntax/configuration checks.
- Use `tau = 0.1` for DTF only in evaluation/inference; use the continuous gate during training.
- Preserve `[CLS]`, `[SEP]`, and padding during textual corruption; token ID `100` is `[UNK]`.
- Public missing masks use `1.0 = missing`, `0.0 = observed or padding` and must be subsets of validity masks.
- Checkpoint selection uses validation rates `(0.0, 0.1, 0.5, 0.9, 1.0)` crossed with mask seeds `(1111, 2222, 3333)` and never consumes test metrics.
- Keep `train.py`, `robust_evaluation.py`, and `build_model(args)` as user-facing entry points.

---

### Task 1: Explicit missingness generation and dataset contract

**Files:**
- Create: `core/missingness.py`
- Modify: `core/dataset.py`
- Create: `tests/__init__.py`
- Create: `tests/test_missingness.py`

**Interfaces:**
- Produces `MissingnessResult(eta, text, audio, vision)` with NumPy boolean masks.
- Produces `training_missingness(valid_masks, text_eligible, seed, epoch, index)`.
- Produces `evaluation_missingness(valid_masks, text_eligible, pattern, rates, seed, index, block_rate=0.0)`.
- Produces `corrupt_text(input_ids, missing_mask, unk_id=100)` and `corrupt_continuous(features, missing_mask)`.
- `MMDataset` exposes `set_epoch(epoch)` and `set_evaluation_corruption(pattern, rates, seed, block_rate=0.0)`.
- Dataset samples expose `*_valid_mask` and `*_missing_mask` with the global convention.

- [ ] **Step 1: Write failing missingness tests**

```python
# tests/test_missingness.py
import unittest
import numpy as np

from core.missingness import (
    corrupt_continuous,
    corrupt_text,
    evaluation_missingness,
    training_missingness,
)


class MissingnessTest(unittest.TestCase):
    def setUp(self):
        self.valid = {
            "text": np.array([1, 1, 1, 1, 0], dtype=bool),
            "audio": np.array([1, 1, 1, 0, 0], dtype=bool),
            "vision": np.array([1, 1, 1, 1, 1], dtype=bool),
        }
        self.text_eligible = np.array([0, 1, 1, 0, 0], dtype=bool)

    def test_training_result_is_deterministic_and_preserves_boundaries(self):
        first = training_missingness(self.valid, self.text_eligible, 2024, 3, 17)
        second = training_missingness(self.valid, self.text_eligible, 2024, 3, 17)
        self.assertEqual(first.eta, second.eta)
        np.testing.assert_array_equal(first.text, second.text)
        self.assertFalse(first.text[0])
        self.assertFalse(first.text[3])
        self.assertFalse(first.text[4])

    def test_epoch_changes_training_corruption(self):
        first = training_missingness(self.valid, self.text_eligible, 2024, 3, 17)
        second = training_missingness(self.valid, self.text_eligible, 2024, 4, 17)
        self.assertNotEqual(first.eta, second.eta)

    def test_missing_masks_never_include_padding(self):
        result = evaluation_missingness(
            self.valid, self.text_eligible, "independent", (1.0, 1.0, 1.0), 1111, 0
        )
        for name in ("text", "audio", "vision"):
            mask = getattr(result, name)
            self.assertTrue(np.all(mask <= self.valid[name]))

    def test_corruption_uses_unk_and_zeros_only_at_explicit_positions(self):
        ids = np.array([101, 12, 13, 102, 0])
        text_missing = np.array([0, 1, 0, 0, 0], dtype=bool)
        np.testing.assert_array_equal(corrupt_text(ids, text_missing), [101, 100, 13, 102, 0])
        features = np.arange(10, dtype=np.float32).reshape(5, 2)
        missing = np.array([0, 1, 0, 0, 0], dtype=bool)
        corrupted = corrupt_continuous(features, missing)
        np.testing.assert_array_equal(corrupted[1], np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(corrupted[0], features[0])

    def test_block_and_mixed_burst_patterns_obey_invariants(self):
        for pattern in ("block", "mixed_burst"):
            result = evaluation_missingness(
                self.valid, self.text_eligible, pattern, (0.5, 0.5, 0.5), 2222, 9, block_rate=0.3
            )
            for name in ("text", "audio", "vision"):
                self.assertTrue(np.all(getattr(result, name) <= self.valid[name]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_missingness -v`

Expected: import failure for `core.missingness`.

- [ ] **Step 3: Implement the missingness utility**

```python
# core/missingness.py
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MissingnessResult:
    eta: float
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray


def _rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(index)]))


def _validate_rate(rate: float) -> float:
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"missing rate must be in [0, 1], got {rate}")
    return rate


def _independent(valid: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(valid.shape) < _validate_rate(rate)) & valid.astype(bool)


def _block(valid: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    positions = np.flatnonzero(valid)
    result = np.zeros_like(valid, dtype=bool)
    count = min(len(positions), int(round(_validate_rate(rate) * len(positions))))
    if count:
        start = int(rng.integers(0, len(positions) - count + 1))
        result[positions[start:start + count]] = True
    return result


def training_missingness(valid_masks, text_eligible, seed, epoch, index):
    rng = _rng(seed, epoch, index)
    eta = float(rng.uniform(0.0, 1.0))
    text = _independent(np.asarray(text_eligible, dtype=bool), eta, rng)
    audio = _independent(np.asarray(valid_masks["audio"], dtype=bool), eta, rng)
    vision = _independent(np.asarray(valid_masks["vision"], dtype=bool), eta, rng)
    return MissingnessResult(eta, text, audio, vision)


def evaluation_missingness(valid_masks, text_eligible, pattern, rates, seed, index, block_rate=0.0):
    rng = _rng(seed, 0, index)
    text_rate, audio_rate, vision_rate = map(_validate_rate, rates)
    eligible = {
        "text": np.asarray(text_eligible, dtype=bool),
        "audio": np.asarray(valid_masks["audio"], dtype=bool),
        "vision": np.asarray(valid_masks["vision"], dtype=bool),
    }
    if pattern in ("independent", "continuous"):
        masks = {name: _independent(eligible[name], rate, rng) for name, rate in zip(eligible, rates)}
    elif pattern == "block":
        masks = {name: _block(eligible[name], rate, rng) for name, rate in zip(eligible, rates)}
    elif pattern == "mixed_burst":
        masks = {
            name: _independent(eligible[name], rate, rng) | _block(eligible[name], block_rate, rng)
            for name, rate in zip(eligible, rates)
        }
    else:
        raise ValueError(f"unsupported missingness pattern: {pattern}")
    return MissingnessResult(float(np.mean(rates)), masks["text"], masks["audio"], masks["vision"])


def corrupt_text(input_ids, missing_mask, unk_id=100):
    result = np.array(input_ids, copy=True)
    result[np.asarray(missing_mask, dtype=bool)] = int(unk_id)
    return result


def corrupt_continuous(features, missing_mask):
    result = np.array(features, copy=True)
    result[np.asarray(missing_mask, dtype=bool)] = 0
    return result
```

- [ ] **Step 4: Refactor `MMDataset` to generate corruption per item**

Replace the mutable whole-dataset `generate_m` path with per-item construction. The key public methods and return fields must be:

```python
def set_epoch(self, epoch):
    self.epoch = int(epoch)

def set_evaluation_corruption(self, pattern, rates, seed, block_rate=0.0):
    self.eval_pattern = str(pattern)
    self.eval_rates = tuple(float(value) for value in rates)
    self.missing_seed = int(seed)
    self.eval_block_rate = float(block_rate)

def __getitem__(self, index):
    text_valid = self.text[index, 1, :].astype(bool)
    text_eligible = text_valid.copy()
    text_eligible[0] = False
    valid_positions = np.flatnonzero(text_valid)
    if len(valid_positions):
        text_eligible[valid_positions[-1]] = False
    audio_valid = np.arange(self.audio.shape[1]) < int(self.audio_lengths[index])
    vision_valid = np.arange(self.vision.shape[1]) < int(self.vision_lengths[index])
    valid = {"text": text_valid, "audio": audio_valid, "vision": vision_valid}
    if self.mode == "train":
        missing = training_missingness(valid, text_eligible, self.missing_seed, self.epoch, index)
    else:
        missing = evaluation_missingness(
            valid, text_eligible, self.eval_pattern, self.eval_rates,
            self.missing_seed, index, self.eval_block_rate,
        )
    input_ids = corrupt_text(self.text[index, 0, :], missing.text)
    text_m = np.stack((input_ids, self.text[index, 1, :], self.text[index, 2, :]))
    return {
        "text": torch.from_numpy(self.text[index]).float(),
        "text_m": torch.from_numpy(text_m).float(),
        "text_valid_mask": torch.from_numpy(text_valid.astype(np.float32)),
        "text_missing_mask": torch.from_numpy(missing.text.astype(np.float32)),
        "audio": torch.from_numpy(self.audio[index]).float(),
        "audio_m": torch.from_numpy(corrupt_continuous(self.audio[index], missing.audio)).float(),
        "audio_valid_mask": torch.from_numpy(audio_valid.astype(np.float32)),
        "audio_missing_mask": torch.from_numpy(missing.audio.astype(np.float32)),
        "vision": torch.from_numpy(self.vision[index]).float(),
        "vision_m": torch.from_numpy(corrupt_continuous(self.vision[index], missing.vision)).float(),
        "vision_valid_mask": torch.from_numpy(vision_valid.astype(np.float32)),
        "vision_missing_mask": torch.from_numpy(missing.vision.astype(np.float32)),
        "requested_missing_rate": torch.tensor(missing.eta, dtype=torch.float32),
        "index": index,
        "id": self.ids[index],
        "labels": {"M": torch.from_numpy(self.labels["M"][index].reshape(-1)).float()},
    }
```

- [ ] **Step 5: Run the missingness tests and syntax check**

Run: `python -m unittest tests.test_missingness -v`

Expected: 5 tests pass.

Run: `python -m py_compile core/missingness.py core/dataset.py`

Expected: exit code 0.

- [ ] **Step 6: Commit Task 1**

```bash
git add core/missingness.py core/dataset.py tests/__init__.py tests/test_missingness.py
git commit -m "feat: make missingness explicit and reproducible"
```

---

### Task 2: LMMT substitution and TMM mask alignment

**Files:**
- Create: `models/missingness.py`
- Modify: `models/tmm.py`
- Create: `tests/test_tmm_and_lmmt.py`

**Interfaces:**
- Produces `apply_missing_token(features, missing_mask, valid_mask, token)`.
- Produces `TMMOutput(text, vision, audio, text_missing, vision_missing, audio_missing, vision_alignment, audio_alignment)`.
- `CTCModule.forward(x, valid_mask)` returns `(aligned_features, alignment_matrix)`.
- `EnhanceSubNet.forward(...)` accepts feature, missing, and validity masks explicitly.

- [ ] **Step 1: Write failing LMMT and TMM tests**

```python
# tests/test_tmm_and_lmmt.py
import unittest
import torch

from models.missingness import apply_missing_token
from models.tmm import CTCModule, EnhanceSubNet


class LMMTAndTMMTest(unittest.TestCase):
    def test_lmmt_replaces_missing_valid_frames_only(self):
        features = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]])
        missing = torch.tensor([[1.0, 0.0, 0.0]])
        valid = torch.tensor([[1.0, 1.0, 0.0]])
        token = torch.nn.Parameter(torch.tensor([[[3.0, 4.0]]]))
        result = apply_missing_token(features, missing, valid, token)
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[0, 1], features[0, 1])
        torch.testing.assert_close(result[0, 2], features[0, 2])

    def test_ctc_alignment_rows_are_normalized_and_ignore_padding(self):
        module = CTCModule(in_dim=3, out_seq_len=2)
        x = torch.randn(2, 4, 3)
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
        aligned, matrix = module(x, valid)
        self.assertEqual(aligned.shape, (2, 2, 3))
        torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(2, 2), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(matrix[0, :, 3], torch.zeros(2)))
        self.assertTrue(torch.equal(matrix[1, :, 2:], torch.zeros(2, 2)))

    def test_tmm_uses_same_matrix_for_mask_alignment(self):
        module = EnhanceSubNet([2, 4, 4], [3, 2, 2], 4)
        output = module(
            torch.randn(1, 2, 3), torch.randn(1, 4, 2), torch.randn(1, 4, 2),
            torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]), torch.ones(1, 4), torch.ones(1, 4),
        )
        expected_v = torch.bmm(output.vision_alignment, torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])).squeeze(-1)
        torch.testing.assert_close(output.vision_missing, expected_v)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_tmm_and_lmmt -v`

Expected: import failure for `models.missingness` or missing `CTCModule` mask signature.

- [ ] **Step 3: Implement explicit LMMT substitution**

```python
# models/missingness.py (initial content)
import torch


def apply_missing_token(features, missing_mask, valid_mask, token):
    if features.ndim != 3 or missing_mask.shape != features.shape[:2]:
        raise ValueError("missing mask must match feature batch and sequence dimensions")
    if valid_mask.shape != missing_mask.shape:
        raise ValueError("valid mask must match missing mask shape")
    active = missing_mask.to(dtype=torch.bool) & valid_mask.to(dtype=torch.bool)
    if torch.any(active & ~valid_mask.to(dtype=torch.bool)):
        raise ValueError("missing mask contains padding positions")
    expanded = active.unsqueeze(-1)
    return torch.where(expanded, token.to(dtype=features.dtype, device=features.device), features)
```

- [ ] **Step 4: Return normalized TMM alignments and aligned masks**

Implement these exact public shapes in `models/tmm.py`:

```python
from dataclasses import dataclass


@dataclass
class TMMOutput:
    text: torch.Tensor
    vision: torch.Tensor
    audio: torch.Tensor
    text_missing: torch.Tensor
    vision_missing: torch.Tensor
    audio_missing: torch.Tensor
    vision_alignment: torch.Tensor
    audio_alignment: torch.Tensor


def _masked_alignment(logits, valid_mask):
    if torch.any(valid_mask.sum(dim=-1) == 0):
        raise ValueError("alignment source sequence has no valid positions")
    logits = logits.masked_fill(~valid_mask[:, None, :].bool(), torch.finfo(logits.dtype).min)
    alignment = torch.softmax(logits, dim=-1)
    if not torch.isfinite(alignment).all():
        raise ValueError("alignment matrix contains non-finite values")
    return alignment
```

`CTCModule.forward` must transpose the non-blank logits to `[B, L_text, L_source]`, call `_masked_alignment`, and return `torch.bmm(P, x)` plus `P`. `EnhanceSubNet.forward` must compute aligned masks with the same `P` matrices and return `TMMOutput`.

- [ ] **Step 5: Run Task 2 tests**

Run: `python -m unittest tests.test_tmm_and_lmmt -v`

Expected: 3 tests pass.

Run: `python -m py_compile models/missingness.py models/tmm.py`

Expected: exit code 0.

- [ ] **Step 6: Commit Task 2**

```bash
git add models/missingness.py models/tmm.py tests/test_tmm_and_lmmt.py
git commit -m "feat: align LMMT inputs and missingness masks"
```

---

### Task 3: DTF formula and bidirectional TC selective scans

**Files:**
- Modify: `models/missingness.py`
- Modify: `models/mamba_nets/mm_bimamba.py`
- Create: `tests/test_dtf.py`

**Interfaces:**
- Produces `DTFOutput(delta, alpha, delta_base)`.
- Produces `DynamicTimeFreezing(feature_dim, mask_dim=2, threshold=0.1)`.
- `MMBiMamba.forward(left_states, right_states, missing_indicator, ...)` applies DTF to both streams and both directions.

- [ ] **Step 1: Write failing DTF tests**

```python
# tests/test_dtf.py
import unittest
import torch

from models.missingness import DynamicTimeFreezing


class DTFTest(unittest.TestCase):
    def make_module(self):
        module = DynamicTimeFreezing(2, mask_dim=2, threshold=0.1)
        with torch.no_grad():
            module.feature_gate.weight.copy_(torch.tensor([[1.0, 0.0]]))
            module.feature_gate.bias.zero_()
            module.mask_gate.weight.copy_(torch.tensor([[-2.0, -2.0]]))
        return module

    def test_formula_is_non_negative_and_mask_conditioned(self):
        module = self.make_module().train()
        features = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        projected = torch.zeros(1, 2, 2)
        observed = module(features, projected, torch.zeros(1, 2, 2))
        missing = module(features, projected, torch.ones(1, 2, 2))
        self.assertTrue(torch.all(observed.delta >= 0))
        self.assertTrue(torch.all(missing.delta < observed.delta))

    def test_training_keeps_continuous_gate_and_eval_applies_threshold(self):
        module = self.make_module()
        features = torch.tensor([[[-10.0, 0.0]]])
        projected = torch.zeros(1, 1, 2)
        masks = torch.zeros(1, 1, 2)
        train_out = module.train()(features, projected, masks)
        eval_out = module.eval()(features, projected, masks)
        self.assertGreater(train_out.delta.abs().sum().item(), 0.0)
        self.assertEqual(eval_out.delta.abs().sum().item(), 0.0)

    def test_invalid_shapes_and_threshold_fail_clearly(self):
        with self.assertRaises(ValueError):
            DynamicTimeFreezing(2, threshold=1.0)
        module = DynamicTimeFreezing(2)
        with self.assertRaises(ValueError):
            module(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, 1))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_dtf -v`

Expected: `DynamicTimeFreezing` is not defined.

- [ ] **Step 3: Implement DTF as a pure PyTorch module**

```python
# append to models/missingness.py
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F


@dataclass
class DTFOutput:
    delta: torch.Tensor
    alpha: torch.Tensor
    delta_base: torch.Tensor


class DynamicTimeFreezing(nn.Module):
    def __init__(self, feature_dim, mask_dim=2, threshold=0.1, device=None, dtype=None):
        super().__init__()
        if not 0.0 <= float(threshold) < 1.0:
            raise ValueError(f"DTF threshold must be in [0, 1), got {threshold}")
        kwargs = {"device": device, "dtype": dtype}
        self.mask_dim = int(mask_dim)
        self.threshold = float(threshold)
        self.feature_gate = nn.Linear(feature_dim, 1, bias=True, **kwargs)
        self.mask_gate = nn.Linear(mask_dim, 1, bias=False, **kwargs)
        nn.init.constant_(self.feature_gate.bias, 2.0)

    def forward(self, features, delta_projected, missing_indicator):
        if features.shape != delta_projected.shape:
            raise ValueError("DTF features and projected delta must have identical shapes")
        if missing_indicator.shape != (*features.shape[:2], self.mask_dim):
            raise ValueError(f"DTF missing indicator must end in {self.mask_dim} channels")
        logits = self.feature_gate(features) + self.mask_gate(missing_indicator.to(features.dtype))
        alpha = torch.sigmoid(logits)
        effective_alpha = alpha if self.training else torch.where(alpha > self.threshold, alpha, torch.zeros_like(alpha))
        delta_base = F.softplus(delta_projected)
        return DTFOutput(delta_base * effective_alpha, alpha, delta_base)
```

- [ ] **Step 4: Integrate DTF into forward and backward scans**

In `models/mamba_nets/mm_bimamba.py`:

1. Replace `a_time_gate`/`v_time_gate` with two `DynamicTimeFreezing` modules.
2. Accept `missing_indicator` shaped `[B, L, 2]`.
3. Compute forward projected deltas, call DTF, and pass `delta_softplus=False` with no delta bias.
4. Compute backward convolution/projections from reversed `xz`; reverse `missing_indicator` with the features; call the same stream DTF; run the backward scan with `A_b` and backward parameters; flip outputs back.
5. Sum forward and restored backward outputs before each output projection.
6. Store detached `alpha`, `delta`, and `delta_base` dictionaries for diagnostics only.

The combination must follow:

```python
forward = selective_scan_fn(x_f, dt_f, A, B_f, C_f, D_f, z=z_f, delta_bias=None, delta_softplus=False)
backward_reversed = selective_scan_fn(
    x_b, dt_b, A_b, B_b, C_b, D_b, z=z_b, delta_bias=None, delta_softplus=False
)
combined = forward + backward_reversed.flip(-1)
```

- [ ] **Step 5: Run Task 3 tests and static checks**

Run: `python -m unittest tests.test_dtf -v`

Expected: 3 tests pass.

Run: `python -m py_compile models/missingness.py models/mamba_nets/mm_bimamba.py`

Expected: exit code 0 without importing CUDA kernels.

- [ ] **Step 6: Commit Task 3**

```bash
git add models/missingness.py models/mamba_nets/mm_bimamba.py tests/test_dtf.py
git commit -m "feat: implement mask-aware bidirectional DTF"
```

---

### Task 4: Branch-preserving TC-Mamba, key-only RoPE, and CMS model flow

**Files:**
- Modify: `models/mamba_nets/attention.py`
- Modify: `models/mamba.py`
- Modify: `models/TFMamba.py`
- Create: `tests/test_rotary_attention.py`
- Create: `tests/test_model_contract.py`

**Interfaces:**
- Produces `RotaryKeyCrossAttention(query_dim, modal_dim, heads, dropout=0.0, rope_base=10000.0)`.
- `TCMamba.forward(audio, vision, text, text_missing, audio_missing, vision_missing)` returns `(audio, vision, text_at, text_vt)`.
- `CMSMamba.forward(incomplete_input, missing_masks, valid_masks)` returns at least `{"sentiment_preds": tensor}`.

- [ ] **Step 1: Write failing key-only RoPE tests**

```python
# tests/test_rotary_attention.py
import unittest
import torch

from models.mamba_nets.attention import RotaryKeyCrossAttention


class RotaryAttentionTest(unittest.TestCase):
    def test_rejects_odd_head_dimension(self):
        with self.assertRaises(ValueError):
            RotaryKeyCrossAttention(query_dim=12, modal_dim=24, heads=4, head_dim=3)

    def test_projects_feature_concatenated_modal_input(self):
        module = RotaryKeyCrossAttention(query_dim=8, modal_dim=16, heads=2, head_dim=4)
        query = torch.randn(2, 5, 8)
        modal = torch.randn(2, 5, 16)
        output = module(query, modal)
        self.assertEqual(output.shape, (2, 5, 8))

    def test_rope_changes_keys_but_not_values(self):
        module = RotaryKeyCrossAttention(query_dim=8, modal_dim=16, heads=2, head_dim=4)
        query = torch.randn(1, 4, 8)
        modal = torch.ones(1, 4, 16)
        _, keys, values = module.project_qkv(query, modal, apply_rope=True)
        _, raw_keys, raw_values = module.project_qkv(query, modal, apply_rope=False)
        self.assertFalse(torch.allclose(keys, raw_keys))
        torch.testing.assert_close(values, raw_values)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the RoPE tests and verify RED**

Run: `python -m unittest tests.test_rotary_attention -v`

Expected: `RotaryKeyCrossAttention` is not defined.

- [ ] **Step 3: Implement feature-wise modal attention with key-only RoPE**

`RotaryKeyCrossAttention` must:

```python
def project_qkv(self, query, modal, apply_rope=True):
    q = self._split_heads(self.to_q(query))
    k = self._split_heads(self.to_k(modal))
    v = self._split_heads(self.to_v(modal))
    if apply_rope:
        k = self._apply_rope(k)
    return q, k, v

def forward(self, query, modal):
    q, k, v = self.project_qkv(query, modal, apply_rope=True)
    scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    attended = torch.matmul(torch.softmax(scores, dim=-1), v)
    attended = attended.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], -1)
    return query + self.to_out(attended)
```

The rotary helper constructs position-dependent sine/cosine pairs for `[B, H, L, D_head]` and rotates keys only.

- [ ] **Step 4: Preserve AT and VT text contexts through TC-Mamba**

Refactor `models/mamba.py` so each TC layer receives its branch indicator and the two text states are never averaged inside the stack:

```python
text_at = t_x
text_vt = t_x
at_indicator = torch.stack((text_missing, audio_missing), dim=-1)
vt_indicator = torch.stack((text_missing, vision_missing), dim=-1)
for at_layer, vt_layer in zip(self.at_mamba_layers, self.vt_mamba_layers):
    a_out, text_at = at_layer(a_out, text_at, at_indicator)
    v_out, text_vt = vt_layer(v_out, text_vt, vt_indicator)
return a_out, v_out, text_at, text_vt
```

- [ ] **Step 5: Refactor `TFMamba.py` into the paper flow**

The model class becomes `CMSMamba`. It must:

1. own independent `v_mask_token` and `a_mask_token`;
2. apply explicit LMMT using validity and missing masks;
3. encode corrupted text with BERT;
4. call TMM and receive aligned masks;
5. call TC-Mamba and average `text_at`/`text_vt` only afterward;
6. concatenate `vision`/`audio` as `torch.cat((vision, audio), dim=-1)`;
7. call key-only RoPE cross-attention then TQ-Mamba;
8. pool, apply RNL, and regress;
9. return no reconstruction outputs.

The public signature is:

```python
def forward(self, incomplete_input, missing_masks, valid_masks):
    vision, audio, language = incomplete_input
    text_missing, audio_missing, vision_missing = missing_masks
    text_valid, audio_valid, vision_valid = valid_masks
    # paper-aligned flow
    return {"sentiment_preds": self.output(self.norm_lock(pooled))}
```

- [ ] **Step 6: Add an AST-level model contract test**

```python
# tests/test_model_contract.py
import ast
import pathlib
import unittest


class ModelContractTest(unittest.TestCase):
    def test_model_has_no_zero_heuristic_or_reconstruction_head(self):
        source = pathlib.Path("models/TFMamba.py").read_text(encoding="utf-8")
        self.assertNotIn("== 0).all", source)
        self.assertNotIn("recon_text", source)
        self.assertIn("dim=-1", source)

    def test_model_source_parses(self):
        source = pathlib.Path("models/TFMamba.py").read_text(encoding="utf-8")
        ast.parse(source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 7: Run Task 4 tests and syntax checks**

Run: `python -m unittest tests.test_rotary_attention tests.test_model_contract -v`

Expected: 5 tests pass.

Run: `python -m py_compile models/mamba_nets/attention.py models/mamba.py models/TFMamba.py`

Expected: exit code 0.

- [ ] **Step 8: Commit Task 4**

```bash
git add models/mamba_nets/attention.py models/mamba.py models/TFMamba.py tests/test_rotary_attention.py tests/test_model_contract.py
git commit -m "feat: complete paper-aligned CMS-Mamba fusion"
```

---

### Task 5: MSE-only training and validation-grid checkpoint selection

**Files:**
- Modify: `core/losses.py`
- Create: `core/validation.py`
- Modify: `train.py`
- Create: `tests/test_training_protocol.py`

**Interfaces:**
- `MultimodalLoss.forward(out, label)` returns `{"loss": mse, "l_sp": mse}`.
- `validation_grid(rates, seeds)` returns 15 `(rate, seed)` pairs for manuscript defaults.
- `ValidationCheckpointSelector.update(mean_mae, epoch)` accepts validation MAE only.
- `train.py` saves only the lowest mean validation MAE checkpoint.

- [ ] **Step 1: Write failing training-protocol tests**

```python
# tests/test_training_protocol.py
import unittest
import torch

from core.losses import MultimodalLoss
from core.validation import ValidationCheckpointSelector, validation_grid


class TrainingProtocolTest(unittest.TestCase):
    def test_loss_is_mse_only(self):
        loss_fn = MultimodalLoss({})
        out = {"sentiment_preds": torch.tensor([[1.0], [3.0]])}
        labels = {"sentiment_labels": torch.tensor([[0.0], [1.0]])}
        result = loss_fn(out, labels)
        self.assertEqual(set(result), {"loss", "l_sp"})
        torch.testing.assert_close(result["loss"], torch.tensor(2.5))

    def test_default_validation_grid_has_fifteen_conditions(self):
        grid = validation_grid((0.0, 0.1, 0.5, 0.9, 1.0), (1111, 2222, 3333))
        self.assertEqual(len(grid), 15)
        self.assertEqual(grid[0], (0.0, 1111))
        self.assertEqual(grid[-1], (1.0, 3333))

    def test_selector_uses_only_mean_validation_mae(self):
        selector = ValidationCheckpointSelector()
        self.assertTrue(selector.update(0.8, 1))
        self.assertFalse(selector.update(0.9, 2))
        self.assertTrue(selector.update(0.7, 3))
        self.assertEqual(selector.best_epoch, 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_training_protocol -v`

Expected: missing `core.validation` and old loss signature failure.

- [ ] **Step 3: Implement MSE-only loss and selector**

```python
# core/validation.py
from dataclasses import dataclass
from itertools import product
from math import isfinite


def validation_grid(rates, seeds):
    return tuple((float(rate), int(seed)) for rate, seed in product(rates, seeds))


@dataclass
class ValidationCheckpointSelector:
    best_mae: float = float("inf")
    best_epoch: int = -1

    def update(self, mean_mae, epoch):
        mean_mae = float(mean_mae)
        if not isfinite(mean_mae):
            raise ValueError(f"validation MAE must be finite, got {mean_mae}")
        if mean_mae < self.best_mae:
            self.best_mae = mean_mae
            self.best_epoch = int(epoch)
            return True
        return False
```

`core/losses.py` contains only `nn.MSELoss` and no reconstruction classes.

- [ ] **Step 4: Refactor training to use explicit masks and the validation grid**

`train.py` must build model arguments as:

```python
incomplete_input = (data["vision_m"].to(device), data["audio_m"].to(device), data["text_m"].to(device))
missing_masks = (data["text_missing_mask"].to(device), data["audio_missing_mask"].to(device), data["vision_missing_mask"].to(device))
valid_masks = (data["text_valid_mask"].to(device), data["audio_valid_mask"].to(device), data["vision_valid_mask"].to(device))
out = model(incomplete_input, missing_masks, valid_masks)
loss = loss_fn(out, {"sentiment_labels": data["labels"]["M"].to(device)})
```

Before each epoch call `train_loader.dataset.set_epoch(epoch)`. For validation, iterate `validation_grid`, call `valid_loader.dataset.set_evaluation_corruption("independent", (rate, rate, rate), seed)`, compute MAE, average all 15 values, and save only when `ValidationCheckpointSelector.update(...)` returns true. Remove per-epoch test evaluation and all `best_test_results` selection.

- [ ] **Step 5: Run Task 5 tests and checks**

Run: `python -m unittest tests.test_training_protocol -v`

Expected: 3 tests pass.

Run: `python -m py_compile core/losses.py core/validation.py train.py`

Expected: exit code 0.

- [ ] **Step 6: Commit Task 5**

```bash
git add core/losses.py core/validation.py train.py tests/test_training_protocol.py
git commit -m "fix: select checkpoints on perturbed validation MAE"
```

---

### Task 6: Paper-pattern robustness evaluation and strict checkpoint loading

**Files:**
- Modify: `robust_evaluation.py`
- Create: `tests/test_evaluation_protocol.py`

**Interfaces:**
- CLI accepts `--pattern`, `--missing_rate`, `--text_rate`, `--audio_rate`, `--vision_rate`, `--block_rate`, and `--mask_seed`.
- `resolve_pattern_rates(args)` maps named paper patterns to explicit rate tuples.
- `load_checkpoint` uses `strict=True` and raises an incompatibility error.
- `model_forward` passes explicit feature tensors, missing masks, and validity masks.

- [ ] **Step 1: Write failing evaluation-protocol tests**

```python
# tests/test_evaluation_protocol.py
import argparse
import unittest

from robust_evaluation import resolve_pattern_rates


class EvaluationProtocolTest(unittest.TestCase):
    def make_args(self, pattern, missing_rate=0.5):
        return argparse.Namespace(
            pattern=pattern, missing_rate=missing_rate,
            text_rate=None, audio_rate=None, vision_rate=None, block_rate=0.3,
        )

    def test_complete_modality_patterns(self):
        self.assertEqual(resolve_pattern_rates(self.make_args("text_missing")), (1.0, 0.0, 0.0))
        self.assertEqual(resolve_pattern_rates(self.make_args("av_missing")), (0.0, 1.0, 1.0))

    def test_asymmetric_patterns(self):
        self.assertEqual(resolve_pattern_rates(self.make_args("text_heavy")), (0.7, 0.1, 0.1))
        self.assertEqual(resolve_pattern_rates(self.make_args("av_heavy")), (0.1, 0.7, 0.7))

    def test_continuous_pattern_uses_one_rate(self):
        self.assertEqual(resolve_pattern_rates(self.make_args("continuous", 0.9)), (0.9, 0.9, 0.9))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_evaluation_protocol -v`

Expected: `resolve_pattern_rates` is not defined.

- [ ] **Step 3: Implement explicit named pattern resolution**

```python
def resolve_pattern_rates(opt):
    presets = {
        "text_missing": (1.0, 0.0, 0.0),
        "av_missing": (0.0, 1.0, 1.0),
        "text_heavy": (0.7, 0.1, 0.1),
        "av_heavy": (0.1, 0.7, 0.7),
    }
    if opt.pattern in presets:
        rates = presets[opt.pattern]
    elif None not in (opt.text_rate, opt.audio_rate, opt.vision_rate):
        rates = (opt.text_rate, opt.audio_rate, opt.vision_rate)
    else:
        rates = (opt.missing_rate, opt.missing_rate, opt.missing_rate)
    rates = tuple(float(rate) for rate in rates)
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError(f"missing rates must be in [0, 1], got {rates}")
    return rates
```

Map CLI `block` to dataset pattern `block`, `mixed_burst` to `mixed_burst`, and all independent/preset/continuous cases to `independent` after resolving their rate tuple.

- [ ] **Step 4: Make checkpoint and model forwarding strict**

Use:

```python
try:
    model.load_state_dict(normalize_state_dict(state_dict), strict=True)
except RuntimeError as exc:
    raise RuntimeError(
        "checkpoint is incompatible with the paper-aligned CMS-Mamba architecture; retraining is required"
    ) from exc
```

Remove `missing_policy` and token-zeroing behavior. Pass the three explicit missing masks and validity masks to the model. Report requested rates and realized `missing.sum() / valid.sum()` rates per modality.

- [ ] **Step 5: Run Task 6 tests and checks**

Run: `python -m unittest tests.test_evaluation_protocol -v`

Expected: 3 tests pass.

Run: `python -m py_compile robust_evaluation.py`

Expected: exit code 0.

- [ ] **Step 6: Commit Task 6**

```bash
git add robust_evaluation.py tests/test_evaluation_protocol.py
git commit -m "feat: expose paper missingness evaluation patterns"
```

---

### Task 7: Reconcile configurations and documentation

**Files:**
- Modify: `configs/train_mosi.yaml`
- Modify: `configs/train_mosei.yaml`
- Modify: `configs/train_sims.yaml`
- Modify: `configs/eval_mosi.yaml`
- Modify: `configs/eval_mosei.yaml`
- Modify: `configs/eval_sims.yaml`
- Modify: `README.md`
- Create: `tests/test_configs.py`

**Interfaces:**
- Every training config contains the shared validation grid and `tc_mamba.dtf_threshold: 0.1`.
- No config contains `alpha`, `rec_loss`, or `tmr`.
- MOSEI uses TC/TQ `(2, 2)`, state dimension `16`, expansion `4`, and dropout `0.2`.
- MOSI uses `(1, 1)`, state dimension `12`, expansion `4`, and dropout `0.1`.
- CH-SIMS uses `(1, 2)`, state dimension `16`, expansion `2`, and dropout `0.2`.

- [ ] **Step 1: Write failing config tests**

```python
# tests/test_configs.py
import pathlib
import unittest
import yaml


class ConfigTest(unittest.TestCase):
    def load(self, name):
        return yaml.safe_load(pathlib.Path("configs", name).read_text(encoding="utf-8"))

    def test_training_configs_use_mse_only_and_validation_grid(self):
        for name in ("train_mosi.yaml", "train_mosei.yaml", "train_sims.yaml"):
            config = self.load(name)
            self.assertNotIn("alpha", config["base"])
            self.assertNotIn("rec_loss", config["base"])
            self.assertNotIn("tmr", config["model"])
            self.assertEqual(config["base"]["validation_missing_rates"], [0.0, 0.1, 0.5, 0.9, 1.0])
            self.assertEqual(config["base"]["validation_mask_seeds"], [1111, 2222, 3333])
            self.assertEqual(config["model"]["tc_mamba"]["dtf_threshold"], 0.1)

    def test_mosei_architecture_matches_manuscript(self):
        config = self.load("train_mosei.yaml")
        self.assertEqual(config["model"]["tc_mamba"]["num_layers"], 2)
        self.assertEqual(config["model"]["tq_mamba"]["num_layers"], 2)
        self.assertEqual(config["model"]["tc_mamba"]["mamba_config"]["d_state"], 16)
        self.assertEqual(config["model"]["dropout"], 0.2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_configs -v`

Expected: current reconstruction keys and MOSEI architecture cause failures.

- [ ] **Step 3: Update YAML files**

Each training config must include:

```yaml
base:
  validation_missing_rates: [0.0, 0.1, 0.5, 0.9, 1.0]
  validation_mask_seeds: [1111, 2222, 3333]
model:
  tc_mamba:
    dtf_threshold: 0.1
  tq_mamba:
    rope_base: 10000.0
```

Remove reconstruction and auxiliary-loss keys. Apply the dataset-specific architecture values listed in this task's interfaces to both training and evaluation configurations.

- [ ] **Step 4: Update README**

Document:

- the explicit `1 = missing` contract;
- shared per-sample training `eta`;
- LMMT only on valid missing A/V frames;
- TMM mask propagation;
- DTF formula and inference threshold;
- key-only RoPE and feature-wise A/V concatenation;
- MSE-only training;
- validation-grid checkpoint selection;
- strict old-checkpoint incompatibility;
- named robustness patterns and CLI examples;
- that this code update did not train or reproduce manuscript numbers.

- [ ] **Step 5: Run Task 7 tests and YAML parse checks**

Run: `python -m unittest tests.test_configs -v`

Expected: 2 tests pass.

Run: `python -c "import pathlib, yaml; [yaml.safe_load(p.read_text(encoding='utf-8')) for p in pathlib.Path('configs').glob('*.yaml')]; print('all configs parsed')"`

Expected: `all configs parsed`.

- [ ] **Step 6: Commit Task 7**

```bash
git add configs README.md tests/test_configs.py
git commit -m "docs: align configs and usage with the manuscript"
```

---

### Task 8: Full static verification, self-review, and GitHub delivery

**Files:**
- Modify if required by failures: only files already listed in Tasks 1-7
- Inspect: `docs/superpowers/specs/2026-07-18-paper-aligned-cms-mamba-design.md`
- Inspect: `docs/superpowers/plans/2026-07-18-paper-aligned-cms-mamba.md`

**Interfaces:**
- Produces a clean feature branch with passing lightweight tests and static checks.
- Produces a pushed `agent/paper-aligned-cms-mamba` branch and draft PR targeting `main`.

- [ ] **Step 1: Run all lightweight unit tests**

Run: `python -m unittest discover -s tests -v`

Expected: all non-CUDA tests pass with zero failures and zero errors. CUDA-only integration checks may be explicitly skipped, never silently counted as passing.

- [ ] **Step 2: Run syntax and whitespace verification**

Run: `python -m compileall -q core models train.py robust_evaluation.py tests`

Expected: exit code 0.

Run: `git diff --check origin/main...HEAD`

Expected: no output and exit code 0.

- [ ] **Step 3: Verify the requirements directly**

Run these source scans:

```powershell
Get-ChildItem -Recurse -File -Include *.py,*.yaml | Select-String -Pattern 'recon_text|Rec_Fn|best_test_results|== 0\)\.all'
Get-ChildItem -Recurse -File -Include *.py | Select-String -Pattern 'mask_gate|dtf_threshold|RotaryKeyCrossAttention|validation_grid'
```

Expected: the forbidden scan has no implementation matches; the required scan finds the DTF mask branch, threshold, key-only rotary attention, and validation-grid code.

- [ ] **Step 4: Review the complete diff against the spec**

Run: `git diff --stat origin/main...HEAD`

Run: `git diff origin/main...HEAD -- core models train.py robust_evaluation.py configs README.md tests`

Check every acceptance criterion in the design spec. Fix Critical and Important issues, rerun the affected tests, and create a focused commit for each fix.

- [ ] **Step 5: Confirm publish scope**

Run: `git status -sb`

Expected: only intentional tracked changes; working tree clean after all commits.

- [ ] **Step 6: Push and open a draft pull request**

Push:

```bash
git push -u origin agent/paper-aligned-cms-mamba
```

Open a draft PR through the connected GitHub app with:

- repository: `Indecis1ve/CMS-Mamba`
- base: `main`
- head: `agent/paper-aligned-cms-mamba`
- title: `Align CMS-Mamba implementation with the manuscript`
- body: summarize explicit masks, LMMT/TMM/DTF/RoPE/RNL, MSE-only validation selection, tests, old checkpoint incompatibility, and the fact that no training or paper-result reproduction was performed.

- [ ] **Step 7: Report evidence**

Report branch, final commit SHA, draft PR URL, exact test counts, syntax/config checks, skipped CUDA checks, and the explicit statement that no training, dataset evaluation, benchmark, or performance simulation was run.

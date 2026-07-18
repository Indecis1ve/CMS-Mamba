"""Deterministic missingness policies shared by training and evaluation.

Public masks use one convention throughout the project: ``True``/``1``
means that a valid position is missing. Padding is never marked missing.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "vision")


@dataclass(frozen=True)
class MissingnessResult:
    eta: float
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray


def _rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), int(index)])
    )


def _validate_rate(rate: float) -> float:
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"missing rate must be in [0, 1], got {rate}")
    return rate


def _as_bool_mask(mask: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 1:
        raise ValueError(f"{name} mask must be one-dimensional, got {result.shape}")
    return result


def _independent(
    valid: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random(valid.shape) < _validate_rate(rate)) & valid


def _block(
    valid: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = np.flatnonzero(valid)
    result = np.zeros_like(valid, dtype=bool)
    count = min(len(positions), int(round(_validate_rate(rate) * len(positions))))
    if count:
        start = int(rng.integers(0, len(positions) - count + 1))
        result[positions[start : start + count]] = True
    return result


def _eligible_masks(
    valid_masks: Mapping[str, np.ndarray],
    text_eligible: np.ndarray,
) -> dict[str, np.ndarray]:
    missing = [name for name in MODALITIES if name not in valid_masks]
    if missing:
        raise KeyError(f"valid masks are missing modalities: {missing}")
    valid = {
        name: _as_bool_mask(valid_masks[name], f"{name} valid")
        for name in MODALITIES
    }
    eligible_text = _as_bool_mask(text_eligible, "text eligible")
    if eligible_text.shape != valid["text"].shape:
        raise ValueError("text eligible and valid masks must have identical shapes")
    if np.any(eligible_text & ~valid["text"]):
        raise ValueError("text eligible mask contains padding positions")
    return {
        "text": eligible_text,
        "audio": valid["audio"],
        "vision": valid["vision"],
    }


def training_missingness(
    valid_masks: Mapping[str, np.ndarray],
    text_eligible: np.ndarray,
    seed: int,
    epoch: int,
    index: int,
) -> MissingnessResult:
    """Draw one shared eta and independent masks for a training sample."""

    eligible = _eligible_masks(valid_masks, text_eligible)
    rng = _rng(seed, epoch, index)
    eta = float(rng.uniform(0.0, 1.0))
    masks = {
        name: _independent(eligible[name], eta, rng) for name in MODALITIES
    }
    return MissingnessResult(eta, masks["text"], masks["audio"], masks["vision"])


def evaluation_missingness(
    valid_masks: Mapping[str, np.ndarray],
    text_eligible: np.ndarray,
    pattern: str,
    rates: Sequence[float],
    seed: int,
    index: int,
    block_rate: float = 0.0,
) -> MissingnessResult:
    """Create a deterministic evaluation mask for one sample."""

    if len(rates) != 3:
        raise ValueError("rates must contain text, audio, and vision values")
    rate_tuple = tuple(_validate_rate(value) for value in rates)
    eligible = _eligible_masks(valid_masks, text_eligible)
    rng = _rng(seed, 0, index)
    normalized_pattern = str(pattern).lower().replace("-", "_")

    if normalized_pattern in ("independent", "continuous"):
        masks = {
            name: _independent(eligible[name], rate, rng)
            for name, rate in zip(MODALITIES, rate_tuple)
        }
    elif normalized_pattern == "block":
        masks = {
            name: _block(eligible[name], rate, rng)
            for name, rate in zip(MODALITIES, rate_tuple)
        }
    elif normalized_pattern == "mixed_burst":
        burst_rate = _validate_rate(block_rate)
        masks = {
            name: _independent(eligible[name], rate, rng)
            | _block(eligible[name], burst_rate, rng)
            for name, rate in zip(MODALITIES, rate_tuple)
        }
    else:
        raise ValueError(f"unsupported missingness pattern: {pattern}")

    return MissingnessResult(
        float(np.mean(rate_tuple)),
        masks["text"],
        masks["audio"],
        masks["vision"],
    )


def corrupt_text(
    input_ids: np.ndarray,
    missing_mask: np.ndarray,
    unk_id: int = 100,
) -> np.ndarray:
    result = np.array(input_ids, copy=True)
    missing = _as_bool_mask(missing_mask, "text missing")
    if result.shape != missing.shape:
        raise ValueError("text IDs and missing mask must have identical shapes")
    result[missing] = int(unk_id)
    return result


def corrupt_continuous(
    features: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    result = np.array(features, copy=True)
    missing = _as_bool_mask(missing_mask, "continuous missing")
    if result.ndim != 2 or result.shape[0] != missing.shape[0]:
        raise ValueError(
            "continuous features and missing mask must share the sequence dimension"
        )
    result[missing] = 0
    return result

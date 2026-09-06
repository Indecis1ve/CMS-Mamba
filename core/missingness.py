"""Deterministic corruption and input-derived missingness utilities.

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


@dataclass(frozen=True)
class ContinuousFeatureStatistics:
    """Training-split statistics used by the automatic frame detector."""

    mean: np.ndarray
    variance: np.ndarray


def _valid_frame_mask(lengths: np.ndarray, sequence_length: int) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1:
        raise ValueError("continuous sequence lengths must be one-dimensional")
    if np.any(lengths < 0) or np.any(lengths > sequence_length):
        raise ValueError("continuous sequence lengths are outside the input range")
    return np.arange(sequence_length)[None, :] < lengths[:, None]


def fit_continuous_feature_statistics(
    features: np.ndarray,
    lengths: np.ndarray,
) -> ContinuousFeatureStatistics:
    """Fit per-feature statistics on finite, non-padding training frames only."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("continuous features must have shape [N, L, D]")
    valid = _valid_frame_mask(lengths, values.shape[1])
    finite = np.isfinite(values).all(axis=-1)
    observed = values[valid & finite]
    if observed.size == 0:
        raise ValueError("training split has no finite continuous frames")
    return ContinuousFeatureStatistics(
        mean=observed.mean(axis=0, dtype=np.float64).astype(np.float32),
        variance=observed.var(axis=0, dtype=np.float64).astype(np.float32),
    )


def standardize_received_continuous(
    features: np.ndarray,
    valid_mask: np.ndarray,
    statistics: ContinuousFeatureStatistics,
    epsilon: float = 1e-6,
    zero_variance_threshold: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize received frames while preserving direct missingness evidence.

    Exact-zero and non-finite valid frames are recorded before standardization
    and written back as exact zeros.  Padding stays zero and never becomes a
    missing observation.  Near-constant dimensions are centered and set to
    zero as specified by the manuscript.
    """

    values = np.asarray(features, dtype=np.float32)
    valid = _as_bool_mask(valid_mask, "continuous valid")
    if values.ndim != 2 or values.shape[0] != valid.shape[0]:
        raise ValueError(
            "continuous features and valid mask must share the sequence dimension"
        )
    if statistics.mean.shape != (values.shape[1],) or statistics.variance.shape != (
        values.shape[1],
    ):
        raise ValueError("continuous statistics do not match feature dimensions")
    if epsilon <= 0.0 or zero_variance_threshold < 0.0:
        raise ValueError("standardization constants must be non-negative")

    finite = np.isfinite(values).all(axis=-1)
    exact_zero = finite & np.all(values == 0.0, axis=-1)
    directly_missing = valid & (~finite | exact_zero)
    observed = valid & finite & ~exact_zero

    standardized = np.zeros_like(values, dtype=np.float32)
    denominator = np.sqrt(statistics.variance + float(epsilon))
    standardized[observed] = (
        values[observed] - statistics.mean
    ) / denominator
    constant_dimensions = statistics.variance < float(zero_variance_threshold)
    standardized[:, constant_dimensions] = 0.0
    return standardized, directly_missing


def automatic_continuous_missingness(
    standardized_features: np.ndarray,
    valid_mask: np.ndarray,
    directly_missing: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Estimate continuous-frame missingness from received standardized input.

    Direct exact-zero/non-finite evidence is always retained.  A non-zero
    low-energy frame requires an adjacent valid low-energy frame, preventing a
    single weak but informative frame from being classified as unavailable.
    """

    values = np.asarray(standardized_features, dtype=np.float32)
    valid = _as_bool_mask(valid_mask, "continuous valid")
    direct = _as_bool_mask(directly_missing, "direct continuous missing")
    if values.ndim != 2 or values.shape[0] != valid.shape[0]:
        raise ValueError(
            "standardized features and valid mask must share the sequence dimension"
        )
    if direct.shape != valid.shape:
        raise ValueError("direct continuous missing mask must match valid mask")
    if np.any(direct & ~valid):
        raise ValueError("direct continuous missing mask contains padding")
    threshold = float(threshold)
    if threshold < 0.0:
        raise ValueError("continuous missingness threshold must be non-negative")

    score = np.linalg.norm(values, axis=-1) / np.sqrt(values.shape[-1])
    low_energy = valid & (score < threshold)
    neighbor_support = np.zeros_like(valid, dtype=bool)
    neighbor_support[1:] |= valid[:-1] & low_energy[:-1]
    neighbor_support[:-1] |= valid[1:] & low_energy[1:]
    return (direct | (low_energy & neighbor_support)) & valid


def automatic_text_missingness(
    input_ids: np.ndarray,
    valid_mask: np.ndarray,
    unk_id: int = 100,
) -> np.ndarray:
    """Mark non-special valid ``[UNK]`` tokens as input-derived missingness."""

    ids = np.asarray(input_ids)
    valid = _as_bool_mask(valid_mask, "text valid")
    if ids.ndim != 1 or ids.shape != valid.shape:
        raise ValueError("text IDs and valid mask must have identical shapes")
    eligible = np.array(valid, copy=True)
    positions = np.flatnonzero(valid)
    if positions.size:
        eligible[positions[0]] = False
        eligible[positions[-1]] = False
    return (ids == int(unk_id)) & eligible


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

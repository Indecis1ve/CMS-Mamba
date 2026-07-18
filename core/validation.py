"""Validation-grid construction and MAE-only checkpoint selection."""

from dataclasses import dataclass
from itertools import product
from math import isfinite

import torch


def regression_mae(predictions, targets):
    """Return unrounded sample-level MAE for checkpoint selection."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if predictions.numel() == 0:
        raise ValueError("predictions and targets cannot be empty")
    return float(torch.mean(torch.abs(predictions - targets)).item())


def validation_grid(rates, seeds):
    grid = tuple(
        (float(rate), int(seed)) for rate, seed in product(rates, seeds)
    )
    if not grid:
        raise ValueError("validation grid cannot be empty")
    if any(rate < 0.0 or rate > 1.0 for rate, _ in grid):
        raise ValueError("validation missing rates must be in [0, 1]")
    return grid


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

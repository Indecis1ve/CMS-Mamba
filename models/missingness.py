"""Paper-aligned missing-input and state-update primitives."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def apply_missing_token(features, missing_mask, valid_mask, token):
    """Replace explicitly missing valid frames with a learned token.

    Padding is rejected rather than silently filled, because padding and
    missing observations have different semantics in the manuscript.
    """

    if features.ndim != 3:
        raise ValueError(f"features must have shape [B, L, D], got {features.shape}")
    if missing_mask.shape != features.shape[:2]:
        raise ValueError(
            "missing mask must match the feature batch and sequence dimensions"
        )
    if valid_mask.shape != missing_mask.shape:
        raise ValueError("valid mask must match missing mask shape")
    if token.shape[-1] != features.shape[-1]:
        raise ValueError(
            f"missing token dimension must be {features.shape[-1]}, got {token.shape[-1]}"
        )

    missing = missing_mask.to(device=features.device, dtype=torch.bool)
    valid = valid_mask.to(device=features.device, dtype=torch.bool)
    if torch.any(missing & ~valid):
        raise ValueError("missing mask contains padding positions")

    replacement = token.to(device=features.device, dtype=features.dtype)
    return torch.where(missing.unsqueeze(-1), replacement, features)


@dataclass
class DTFOutput:
    delta: torch.Tensor
    alpha: torch.Tensor
    delta_base: torch.Tensor


class DynamicTimeFreezing(nn.Module):
    """Feature- and mask-conditioned controller for the effective SSM step."""

    def __init__(
        self,
        feature_dim,
        mask_dim=2,
        threshold=0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not 0.0 <= float(threshold) < 1.0:
            raise ValueError(f"DTF threshold must be in [0, 1), got {threshold}")
        if int(mask_dim) <= 0:
            raise ValueError(f"DTF mask dimension must be positive, got {mask_dim}")
        kwargs = {"device": device, "dtype": dtype}
        self.mask_dim = int(mask_dim)
        self.threshold = float(threshold)
        self.feature_gate = nn.Linear(feature_dim, 1, bias=True, **kwargs)
        self.mask_gate = nn.Linear(mask_dim, 1, bias=False, **kwargs)
        nn.init.constant_(self.feature_gate.bias, 2.0)

    def forward(self, features, delta_projected, missing_indicator):
        if features.ndim != 3:
            raise ValueError(
                f"DTF features must have shape [B, L, D], got {features.shape}"
            )
        if features.shape != delta_projected.shape:
            raise ValueError(
                "DTF features and projected delta must have identical shapes"
            )
        expected_mask_shape = (*features.shape[:2], self.mask_dim)
        if missing_indicator.shape != expected_mask_shape:
            raise ValueError(
                f"DTF missing indicator must have {self.mask_dim} channels; "
                f"expected {expected_mask_shape}, got {missing_indicator.shape}"
            )
        if not torch.isfinite(features).all() or not torch.isfinite(
            delta_projected
        ).all():
            raise ValueError("DTF inputs contain non-finite values")

        indicator = missing_indicator.to(
            device=features.device,
            dtype=features.dtype,
        )
        logits = self.feature_gate(features) + self.mask_gate(indicator)
        alpha = torch.sigmoid(logits)
        effective_alpha = (
            alpha
            if self.training
            else torch.where(
                alpha > self.threshold,
                alpha,
                torch.zeros_like(alpha),
            )
        )
        delta_base = F.softplus(delta_projected)
        return DTFOutput(
            delta=delta_base * effective_alpha,
            alpha=alpha,
            delta_base=delta_base,
        )

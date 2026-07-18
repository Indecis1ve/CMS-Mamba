"""Paper-aligned missing-input and state-update primitives."""

import torch


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

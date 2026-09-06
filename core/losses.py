"""Paper-aligned sentiment and masked text-reconstruction objectives."""

import torch
from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    """MSE sentiment loss plus the inherited masked Smooth L1 objective."""

    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss()
        reconstruction = args.get("model", {}).get("reconstruction", {})
        self.reconstruction_weight = float(
            reconstruction.get("loss_weight", 0.0)
        )
        if self.reconstruction_weight < 0.0:
            raise ValueError("reconstruction loss weight must be non-negative")
        self.eps = 1e-6

    def _reconstruction_loss(self, out, label):
        prediction = out.get("reconstructed_text")
        target = out.get("complete_text_features")
        mask = label.get("text_reconstruction_mask")
        if prediction is None or target is None or mask is None:
            return out["sentiment_preds"].sum() * 0.0
        if prediction.shape != target.shape:
            raise ValueError(
                "reconstructed text and complete text features must have identical shapes"
            )
        if mask.shape != prediction.shape[:2]:
            raise ValueError(
                "text reconstruction mask must match batch and token dimensions"
            )
        expanded_mask = mask.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).unsqueeze(-1).expand_as(prediction)
        if torch.any(expanded_mask < 0.0) or torch.any(expanded_mask > 1.0):
            raise ValueError("text reconstruction mask must be in [0, 1]")
        elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
        return (elementwise * expanded_mask).sum() / (
            expanded_mask.sum() + self.eps
        )

    def forward(self, out, label):
        sentiment_loss = self.mse(
            out["sentiment_preds"],
            label["sentiment_labels"],
        )
        reconstruction_loss = self._reconstruction_loss(out, label)
        loss = sentiment_loss + self.reconstruction_weight * reconstruction_loss
        return {
            "loss": loss,
            "l_sp": sentiment_loss,
            "l_rec": reconstruction_loss,
        }

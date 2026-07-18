"""Training objective for the matched CMS-Mamba manuscript protocol."""

from torch import nn


class MultimodalLoss(nn.Module):
    """MSE-only continuous sentiment-regression objective."""

    def __init__(self, args=None):
        super().__init__()
        del args
        self.mse = nn.MSELoss()

    def forward(self, out, label):
        sentiment_loss = self.mse(
            out["sentiment_preds"],
            label["sentiment_labels"],
        )
        return {"loss": sentiment_loss, "l_sp": sentiment_loss}

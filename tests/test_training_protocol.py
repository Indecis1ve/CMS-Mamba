from pathlib import Path
import unittest

import torch

from core.losses import MultimodalLoss
from core.validation import (
    ValidationCheckpointSelector,
    regression_mae,
    validation_grid,
)


class TrainingProtocolTest(unittest.TestCase):
    def test_loss_is_mse_only(self):
        loss_fn = MultimodalLoss({})
        out = {"sentiment_preds": torch.tensor([[1.0], [3.0]])}
        labels = {"sentiment_labels": torch.tensor([[0.0], [1.0]])}

        result = loss_fn(out, labels)

        self.assertEqual(set(result), {"loss", "l_sp"})
        torch.testing.assert_close(result["loss"], torch.tensor(2.5))

    def test_default_validation_grid_has_fifteen_conditions(self):
        grid = validation_grid(
            (0.0, 0.1, 0.5, 0.9, 1.0),
            (1111, 2222, 3333),
        )

        self.assertEqual(len(grid), 15)
        self.assertEqual(grid[0], (0.0, 1111))
        self.assertEqual(grid[-1], (1.0, 3333))

    def test_selector_uses_only_mean_validation_mae(self):
        selector = ValidationCheckpointSelector()

        self.assertTrue(selector.update(0.8, 1))
        self.assertFalse(selector.update(0.9, 2))
        self.assertTrue(selector.update(0.7, 3))
        self.assertEqual(selector.best_epoch, 3)

    def test_selection_mae_is_computed_without_metric_rounding(self):
        predictions = torch.tensor([[0.123456], [0.333333]])
        targets = torch.zeros_like(predictions)

        self.assertAlmostEqual(regression_mae(predictions, targets), 0.2283945)

    def test_training_source_never_selects_on_test_metrics(self):
        source = Path("train.py").read_text(encoding="utf-8")

        self.assertNotIn("best_test_results", source)
        self.assertIn("validation_grid", source)
        self.assertIn("set_epoch", source)
        self.assertIn("set_evaluation_corruption", source)
        self.assertIn("regression_mae", source)

        utils_source = Path("core/utils.py").read_text(encoding="utf-8")
        self.assertNotIn("get_best_results", utils_source)


if __name__ == "__main__":
    unittest.main()

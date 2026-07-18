import unittest
from pathlib import Path

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
        torch.testing.assert_close(
            observed.delta,
            observed.delta_base * observed.alpha,
        )

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
        with self.assertRaisesRegex(ValueError, "2 channels"):
            module(
                torch.zeros(1, 2, 2),
                torch.zeros(1, 2, 2),
                torch.zeros(1, 2, 1),
            )

    def test_multimodal_scan_source_contains_masked_bidirectional_dtf(self):
        source = Path("models/mamba_nets/mm_bimamba.py").read_text(encoding="utf-8")
        self.assertIn("missing_indicator", source)
        self.assertIn("self.a_dtf", source)
        self.assertIn("self.v_dtf", source)
        self.assertIn("self.A_b_log", source)
        self.assertIn(".flip(-1)", source)
        self.assertGreaterEqual(source.count("delta_softplus=False"), 4)


if __name__ == "__main__":
    unittest.main()

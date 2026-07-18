from pathlib import Path
import tempfile
import unittest

import torch

import robust_evaluation


class EvaluationProtocolTest(unittest.TestCase):
    def test_named_robustness_patterns_resolve_to_paper_rates(self):
        self.assertEqual(
            robust_evaluation.resolve_pattern_rates("text_missing", 0.3),
            (1.0, 0.0, 0.0),
        )
        self.assertEqual(
            robust_evaluation.resolve_pattern_rates("av_missing", 0.3),
            (0.0, 1.0, 1.0),
        )
        self.assertEqual(
            robust_evaluation.resolve_pattern_rates("text_heavy", 0.3),
            (0.7, 0.1, 0.1),
        )
        self.assertEqual(
            robust_evaluation.resolve_pattern_rates("av_heavy", 0.3),
            (0.1, 0.7, 0.7),
        )

    def test_continuous_rate_and_overrides_are_supported(self):
        self.assertEqual(
            robust_evaluation.resolve_pattern_rates("continuous", 0.9),
            (0.9, 0.9, 0.9),
        )
        self.assertEqual(
            robust_evaluation.resolve_pattern_rates(
                "independent", 0.5, text_rate=0.2, vision_rate=0.8
            ),
            (0.2, 0.5, 0.8),
        )

    def test_checkpoint_loading_is_strict(self):
        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(3, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            torch.save({"state_dict": source.state_dict()}, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                robust_evaluation.load_checkpoint(target, str(checkpoint_path))

    def test_source_does_not_disable_learned_missing_tokens(self):
        source = Path("robust_evaluation.py").read_text(encoding="utf-8")

        self.assertNotIn("apply_missing_policy", source)
        self.assertNotIn("strict=False", source)
        self.assertIn("text_valid_mask", source)
        self.assertIn("text_missing_mask", source)


if __name__ == "__main__":
    unittest.main()

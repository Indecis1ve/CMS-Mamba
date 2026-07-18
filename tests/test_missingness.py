import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pickle

import numpy as np

from core.dataset import MMDataset

from core.missingness import (
    corrupt_continuous,
    corrupt_text,
    evaluation_missingness,
    training_missingness,
)


class MissingnessTest(unittest.TestCase):
    def setUp(self):
        self.valid = {
            "text": np.array([1, 1, 1, 1, 0], dtype=bool),
            "audio": np.array([1, 1, 1, 0, 0], dtype=bool),
            "vision": np.array([1, 1, 1, 1, 1], dtype=bool),
        }
        self.text_eligible = np.array([0, 1, 1, 0, 0], dtype=bool)

    def test_training_result_is_deterministic_and_preserves_boundaries(self):
        first = training_missingness(self.valid, self.text_eligible, 2024, 3, 17)
        second = training_missingness(self.valid, self.text_eligible, 2024, 3, 17)

        self.assertEqual(first.eta, second.eta)
        np.testing.assert_array_equal(first.text, second.text)
        self.assertFalse(first.text[0])
        self.assertFalse(first.text[3])
        self.assertFalse(first.text[4])

    def test_epoch_changes_training_corruption(self):
        first = training_missingness(self.valid, self.text_eligible, 2024, 3, 17)
        second = training_missingness(self.valid, self.text_eligible, 2024, 4, 17)

        self.assertNotEqual(first.eta, second.eta)

    def test_missing_masks_never_include_padding(self):
        result = evaluation_missingness(
            self.valid,
            self.text_eligible,
            "independent",
            (1.0, 1.0, 1.0),
            1111,
            0,
        )

        for name in ("text", "audio", "vision"):
            mask = getattr(result, name)
            self.assertTrue(np.all(mask <= self.valid[name]))

    def test_corruption_uses_unk_and_zeros_only_at_explicit_positions(self):
        ids = np.array([101, 12, 13, 102, 0])
        text_missing = np.array([0, 1, 0, 0, 0], dtype=bool)
        np.testing.assert_array_equal(
            corrupt_text(ids, text_missing),
            [101, 100, 13, 102, 0],
        )

        features = np.arange(10, dtype=np.float32).reshape(5, 2)
        missing = np.array([0, 1, 0, 0, 0], dtype=bool)
        corrupted = corrupt_continuous(features, missing)
        np.testing.assert_array_equal(corrupted[1], np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(corrupted[0], features[0])

    def test_block_and_mixed_burst_patterns_obey_invariants(self):
        for pattern in ("block", "mixed_burst"):
            with self.subTest(pattern=pattern):
                result = evaluation_missingness(
                    self.valid,
                    self.text_eligible,
                    pattern,
                    (0.5, 0.5, 0.5),
                    2222,
                    9,
                    block_rate=0.3,
                )
                for name in ("text", "audio", "vision"):
                    self.assertTrue(np.all(getattr(result, name) <= self.valid[name]))

    def test_dataset_exposes_explicit_valid_and_missing_masks(self):
        text = np.zeros((1, 3, 5), dtype=np.float32)
        text[0, 0] = [101, 11, 12, 102, 0]
        text[0, 1] = [1, 1, 1, 1, 0]
        payload = {
            "train": {
                "text_bert": text,
                "vision": np.arange(10, dtype=np.float32).reshape(1, 5, 2),
                "audio": np.arange(10, dtype=np.float32).reshape(1, 5, 2),
                "raw_text": np.array(["example"]),
                "id": np.array(["sample-0"]),
                "regression_labels": np.array([[0.25]], dtype=np.float32),
                "audio_lengths": np.array([3]),
                "vision_lengths": np.array([4]),
            }
        }
        with TemporaryDirectory() as directory:
            path = Path(directory, "sample.pkl")
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            args = {
                "base": {
                    "train_mode": "regression",
                    "missing_rate_eval_test": 0.5,
                    "seed": 2024,
                },
                "dataset": {"datasetName": "mosei", "dataPath": str(path)},
            }
            dataset = MMDataset(args, mode="train")
            dataset.set_epoch(2)
            sample = dataset[0]

        for modality in ("text", "audio", "vision"):
            valid = sample[f"{modality}_valid_mask"].numpy().astype(bool)
            missing = sample[f"{modality}_missing_mask"].numpy().astype(bool)
            self.assertTrue(np.all(missing <= valid))
        self.assertEqual(sample["text_m"][0, 0].item(), 101)
        self.assertEqual(sample["text_m"][0, 3].item(), 102)


if __name__ == "__main__":
    unittest.main()

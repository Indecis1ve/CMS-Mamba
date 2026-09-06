from pathlib import Path
import unittest

import yaml


EXPECTED_ARCHITECTURE = {
    "mosi": {"layers": (1, 1), "d_state": 12, "expand": 4, "dropout": 0.1},
    "mosei": {"layers": (2, 2), "d_state": 16, "expand": 4, "dropout": 0.2},
    "sims": {"layers": (1, 2), "d_state": 16, "expand": 2, "dropout": 0.2},
}


class ConfigContractTest(unittest.TestCase):
    def _configs(self):
        for path in sorted(Path("configs").glob("*.yaml")):
            with path.open(encoding="utf-8") as handle:
                yield path, yaml.safe_load(handle)

    def test_all_configs_match_paper_architecture(self):
        for path, config in self._configs():
            with self.subTest(path=path):
                dataset = config["dataset"]["datasetName"]
                expected = EXPECTED_ARCHITECTURE[dataset]
                model = config["model"]
                tc = model["tc_mamba"]
                tq = model["tq_mamba"]

                self.assertIn("reconstruction", model)
                self.assertIn("automatic_missingness", config["base"])
                self.assertEqual(config["base"]["mcssm_indicator_source"], "automatic")
                self.assertEqual(
                    (tc["num_layers"], tq["num_layers"]), expected["layers"]
                )
                self.assertEqual(tc["mamba_config"]["d_state"], expected["d_state"])
                self.assertEqual(tq["mamba_config"]["d_state"], expected["d_state"])
                self.assertEqual(tc["mamba_config"]["expand"], expected["expand"])
                self.assertEqual(tq["mamba_config"]["expand"], expected["expand"])
                self.assertEqual(float(tc["dropout"]), expected["dropout"])
                self.assertEqual(float(tq["dropout"]), expected["dropout"])
                self.assertNotIn("dtf_threshold", tc)
                self.assertEqual(float(tq["rope_base"]), 10000.0)

    def test_english_feature_dimensions_match_manuscript(self):
        for path, config in self._configs():
            dataset = config["dataset"]["datasetName"]
            if dataset in {"mosi", "mosei"}:
                with self.subTest(path=path):
                    self.assertEqual(config["model"]["tmm"]["input_dim"], [768, 35, 74])

    def test_training_configs_define_fixed_validation_grid(self):
        for path in sorted(Path("configs").glob("train_*.yaml")):
            with path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            with self.subTest(path=path):
                self.assertEqual(config["base"]["seed"], 2024)
                self.assertEqual(
                    config["base"]["validation_missing_rates"],
                    [0.0, 0.1, 0.3, 0.5, 0.7],
                )
                self.assertEqual(
                    config["base"]["validation_mask_seeds"],
                    [1111, 2222, 3333],
                )


if __name__ == "__main__":
    unittest.main()

import ast
from pathlib import Path
import unittest


class ModelContractTest(unittest.TestCase):
    def test_model_has_no_zero_heuristic_or_reconstruction_head(self):
        source = Path("models/TFMamba.py").read_text(encoding="utf-8")

        self.assertNotIn("== 0).all", source)
        self.assertNotIn("recon_text", source)
        self.assertIn("dim=-1", source)

    def test_model_preserves_branch_text_contexts_until_fusion(self):
        mamba_source = Path("models/mamba.py").read_text(encoding="utf-8")
        model_source = Path("models/TFMamba.py").read_text(encoding="utf-8")

        self.assertIn("text_at", mamba_source)
        self.assertIn("text_vt", mamba_source)
        self.assertIn("text_at + text_vt", model_source)
        self.assertIn("dim=-1", model_source)

    def test_model_sources_parse(self):
        for path in (Path("models/TFMamba.py"), Path("models/mamba.py")):
            ast.parse(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

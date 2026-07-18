import unittest

import torch

from models.mamba_nets.attention import RotaryKeyCrossAttention


class RotaryAttentionTest(unittest.TestCase):
    def test_rejects_odd_head_dimension(self):
        with self.assertRaises(ValueError):
            RotaryKeyCrossAttention(
                query_dim=12,
                modal_dim=24,
                heads=4,
                head_dim=3,
            )

    def test_projects_feature_concatenated_modal_input(self):
        module = RotaryKeyCrossAttention(
            query_dim=8,
            modal_dim=16,
            heads=2,
            head_dim=4,
        )
        query = torch.randn(2, 5, 8)
        modal = torch.randn(2, 5, 16)

        output = module(query, modal)

        self.assertEqual(output.shape, (2, 5, 8))

    def test_rope_changes_keys_but_not_values(self):
        torch.manual_seed(23)
        module = RotaryKeyCrossAttention(
            query_dim=8,
            modal_dim=16,
            heads=2,
            head_dim=4,
        )
        query = torch.randn(1, 4, 8)
        modal = torch.ones(1, 4, 16)

        _, keys, values = module.project_qkv(query, modal, apply_rope=True)
        _, raw_keys, raw_values = module.project_qkv(
            query,
            modal,
            apply_rope=False,
        )

        self.assertFalse(torch.allclose(keys, raw_keys))
        torch.testing.assert_close(values, raw_values)


if __name__ == "__main__":
    unittest.main()

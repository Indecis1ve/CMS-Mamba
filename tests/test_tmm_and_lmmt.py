import unittest

import torch

from models.missingness import apply_missing_token
from models.tmm import CTCModule, EnhanceSubNet


class LMMTAndTMMTest(unittest.TestCase):
    def test_lmmt_replaces_missing_valid_frames_only(self):
        features = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]])
        missing = torch.tensor([[1.0, 0.0, 0.0]])
        valid = torch.tensor([[1.0, 1.0, 0.0]])
        token = torch.nn.Parameter(torch.tensor([[[3.0, 4.0]]]))

        result = apply_missing_token(features, missing, valid, token)

        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[0, 1], features[0, 1])
        torch.testing.assert_close(result[0, 2], features[0, 2])

    def test_lmmt_rejects_missing_padding(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            apply_missing_token(
                torch.zeros(1, 2, 3),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([[1.0, 0.0]]),
                torch.zeros(1, 1, 3),
            )

    def test_ctc_alignment_rows_are_normalized_and_ignore_padding(self):
        torch.manual_seed(7)
        module = CTCModule(in_dim=3, out_seq_len=2)
        x = torch.randn(2, 4, 3)
        valid = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32
        )

        aligned, matrix = module(x, valid)

        self.assertEqual(aligned.shape, (2, 2, 3))
        torch.testing.assert_close(
            matrix.sum(dim=-1),
            torch.ones(2, 2),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertTrue(torch.equal(matrix[0, :, 3], torch.zeros(2)))
        self.assertTrue(torch.equal(matrix[1, :, 2:], torch.zeros(2, 2)))

    def test_tmm_uses_same_matrix_for_mask_alignment(self):
        torch.manual_seed(11)
        module = EnhanceSubNet([2, 4, 4], [3, 2, 2], 4)
        vision_missing = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        output = module(
            torch.randn(1, 2, 3),
            torch.randn(1, 4, 2),
            torch.randn(1, 4, 2),
            torch.tensor([[0.0, 1.0]]),
            vision_missing,
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            torch.ones(1, 4),
            torch.ones(1, 4),
        )
        expected = torch.bmm(
            output.vision_alignment,
            vision_missing.unsqueeze(-1),
        ).squeeze(-1)

        torch.testing.assert_close(output.vision_missing, expected)


if __name__ == "__main__":
    unittest.main()

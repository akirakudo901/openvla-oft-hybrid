"""ChunkLabelHead output shapes."""

from __future__ import annotations

import unittest

import torch

from hybrid.models.label_head import DEFAULT_NUM_LABEL_CLASSES, ChunkLabelHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class ChunkLabelHeadTest(unittest.TestCase):
    def test_logits_and_argmax_shapes(self) -> None:
        batch_size, hidden_dim = 2, 4
        head = ChunkLabelHead(input_dim=hidden_dim, num_label_classes=DEFAULT_NUM_LABEL_CLASSES)
        hidden = torch.zeros(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
        logits = head(hidden)
        labels = head.predict_labels(hidden)
        self.assertEqual(logits.shape, (batch_size, NUM_ACTIONS_CHUNK, DEFAULT_NUM_LABEL_CLASSES))
        self.assertEqual(labels.shape, (batch_size, NUM_ACTIONS_CHUNK))
        self.assertEqual(labels.dtype, torch.long)

    def test_rejects_wrong_token_count(self) -> None:
        head = ChunkLabelHead(input_dim=4)
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 3, 4))


if __name__ == "__main__":
    unittest.main()

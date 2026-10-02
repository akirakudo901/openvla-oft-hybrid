"""MP/L masks inside the hybrid chunk loss."""

from __future__ import annotations

import unittest

import torch

from hybrid.training.loss import compute_hybrid_loss


class HybridLossMaskTest(unittest.TestCase):
    def test_mp_and_l_l1_are_separate_and_empty_side_is_zero(self) -> None:
        # Step 0 is B-MP with abs error 1. Step 1 is B-L with abs error 0.
        actions = torch.zeros(1, 2, 1)
        actions_hat = torch.tensor([[[1.0], [0.0]]])
        labels = torch.tensor([[0, 2]])
        logits = torch.zeros(1, 2, 5)
        loss, metrics = compute_hybrid_loss(actions_hat, actions, logits, labels)
        self.assertAlmostEqual(metrics["mp_l1_loss"], 1.0)
        self.assertAlmostEqual(metrics["l_l1_loss"], 0.0)
        self.assertAlmostEqual(metrics["weighted_l1_loss"], 1.0)
        self.assertTrue(torch.isfinite(loss))

    def test_pad_steps_are_dropped(self) -> None:
        actions = torch.zeros(1, 2, 1)
        actions_hat = torch.tensor([[[1.0], [100.0]]])
        labels = torch.tensor([[0, 0]])
        is_pad = torch.tensor([[False, True]])
        logits = torch.zeros(1, 2, 5)
        _loss, metrics = compute_hybrid_loss(
            actions_hat, actions, logits, labels, is_pad=is_pad
        )
        self.assertAlmostEqual(metrics["mp_l1_loss"], 1.0)
        self.assertAlmostEqual(metrics["l_l1_loss"], 0.0)

    def test_all_mp_makes_l_term_zero(self) -> None:
        actions = torch.zeros(1, 2, 1)
        actions_hat = torch.ones(1, 2, 1)
        labels = torch.tensor([[0, 4]])
        logits = torch.zeros(1, 2, 5)
        _loss, metrics = compute_hybrid_loss(actions_hat, actions, logits, labels)
        self.assertAlmostEqual(metrics["l_l1_loss"], 0.0)
        self.assertAlmostEqual(metrics["mp_l1_loss"], 1.0)


if __name__ == "__main__":
    unittest.main()

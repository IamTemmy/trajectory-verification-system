from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import torch

from model import FUTURE_STEPS, NUM_MODES, TrajectoryPredictor, multimodal_loss


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TrajectoryPredictor(
            dim=16, heads=4, blocks=1, map_groups=4, dropout=0.0
        ).eval()
        self.target = torch.zeros(2, 11, 7)
        self.target[..., 6] = 1.0
        self.neighbors = torch.zeros(2, 3, 11, 10)
        self.map_points = torch.zeros(2, 8, 5)

    def test_output_matches_external_contract_shape_and_is_finite(self) -> None:
        trajectories, logits = self.model(
            self.target,
            self.neighbors,
            self.map_points,
            torch.tensor([1, 2]),
        )

        self.assertEqual(trajectories.shape, (2, NUM_MODES, FUTURE_STEPS, 2))
        self.assertEqual(logits.shape, (2, NUM_MODES))
        self.assertTrue(torch.isfinite(trajectories).all())
        self.assertTrue(torch.isfinite(logits).all())

    def test_target_object_type_changes_the_prediction(self) -> None:
        vehicle, _ = self.model(
            self.target[:1], self.neighbors[:1], self.map_points[:1],
            torch.tensor([1]),
        )
        pedestrian, _ = self.model(
            self.target[:1], self.neighbors[:1], self.map_points[:1],
            torch.tensor([2]),
        )

        self.assertFalse(torch.allclose(vehicle, pedestrian))

    def test_partial_future_mask_controls_loss_and_gradients(self) -> None:
        trajectories = torch.zeros(2, NUM_MODES, FUTURE_STEPS, 2, requires_grad=True)
        logits = torch.zeros(2, NUM_MODES, requires_grad=True)
        future = torch.ones(2, FUTURE_STEPS, 2)
        mask = torch.zeros(2, FUTURE_STEPS)
        mask[:, :8] = 1.0

        loss, regression, classification = multimodal_loss(
            trajectories, logits, future, mask
        )
        changed_only_after_mask = future.clone()
        changed_only_after_mask[:, 8:] = 10_000.0
        masked_loss, _, _ = multimodal_loss(
            trajectories, logits, changed_only_after_mask, mask
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(regression))
        self.assertTrue(torch.isfinite(classification))
        self.assertAlmostEqual(
            float(loss.detach()), float(masked_loss.detach()), places=6
        )

        loss.backward()
        self.assertTrue(torch.isfinite(trajectories.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from math import cos, sin
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import numpy as np

from dataset import FeatureDataset, consolidate, sampling_weights, split_indices
from features import _rotate
from predict import to_global


class DataPipelineTests(unittest.TestCase):
    def test_agent_frame_round_trip(self) -> None:
        origin = np.array([12.5, -7.0], dtype=np.float32)
        heading = 0.63
        global_points = np.array([[14.0, -2.0], [9.0, -5.5]], dtype=np.float64)
        local = np.array(
            [
                _rotate(x - origin[0], y - origin[1], cos(heading), sin(heading))
                for x, y in global_points
            ]
        )

        np.testing.assert_allclose(
            to_global(local, origin, heading), global_points, rtol=0.0, atol=1e-6
        )

    def test_split_and_class_weights_are_deterministic(self) -> None:
        first = split_indices(20, 0.2, seed=3)
        second = split_indices(20, 0.2, seed=3)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

        codes = np.array([1, 1, 1, 2], dtype=np.int64)
        weights = sampling_weights(codes, damping=1.0)
        self.assertAlmostEqual(weights[0], 4.0 / 3.0)
        self.assertAlmostEqual(weights[-1], 4.0)

    def test_consolidated_arrays_are_readable_by_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = root / "archives"
            flat = root / "flat"
            archives.mkdir()

            for shard, size in (("a", 2), ("b", 3)):
                np.savez_compressed(
                    archives / f"{shard}.npz",
                    target_history=np.zeros((size, 11, 7), dtype=np.float32),
                    neighbors=np.zeros((size, 16, 11, 10), dtype=np.float32),
                    map_points=np.zeros((size, 256, 5), dtype=np.float32),
                    future=np.zeros((size, 16, 2), dtype=np.float32),
                    future_mask=np.ones((size, 16), dtype=np.float32),
                    scenario_id=np.array([f"{shard}-{i}" for i in range(size)]),
                    agent_id=np.array([str(i) for i in range(size)]),
                    object_type=np.array(["vehicle"] * size),
                )

            self.assertEqual(consolidate(archives, flat), 5)
            dataset = FeatureDataset(flat)
            self.assertEqual(len(dataset), 5)
            self.assertEqual(tuple(dataset[0]["target_history"].shape), (11, 7))
            self.assertEqual(int(dataset[0]["object_type"]), 1)
            metadata = json.loads((flat / "meta.json").read_text())
            self.assertEqual(metadata["count"], 5)


if __name__ == "__main__":
    unittest.main()

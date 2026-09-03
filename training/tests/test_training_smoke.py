from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TrainingSmokeTests(unittest.TestCase):
    def test_tiny_cpu_run_saves_and_resumes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = root / "flat"
            checkpoints = root / "checkpoints"
            flat.mkdir()
            count = 12
            generator = np.random.default_rng(4)

            arrays = {
                "target_history": generator.normal(size=(count, 11, 7)).astype(np.float32),
                "neighbors": generator.normal(size=(count, 16, 11, 10)).astype(np.float32),
                "map_points": generator.normal(size=(count, 256, 5)).astype(np.float32),
                "future": generator.normal(size=(count, 16, 2)).astype(np.float32),
                "future_mask": np.ones((count, 16), dtype=np.float32),
            }
            arrays["target_history"][..., 6] = 1.0
            arrays["neighbors"][..., 6] = 1.0
            arrays["map_points"][..., 4] = 1.0
            for name, values in arrays.items():
                np.save(flat / f"{name}.npy", values)

            labels = {
                "scenario_id": [f"scenario-{i}" for i in range(count)],
                "agent_id": [str(i) for i in range(count)],
                "object_type": ["vehicle", "pedestrian", "cyclist"] * 4,
            }
            (flat / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            (flat / "meta.json").write_text(
                json.dumps({"count": count, "shards": ["synthetic"]}),
                encoding="utf-8",
            )

            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "train.py"),
                "--features", str(root / "unused-features"),
                "--flat", str(flat),
                "--checkpoint-dir", str(checkpoints),
                "--batch-size", "4",
                "--workers", "0",
                "--dim", "16",
                "--blocks", "1",
                "--warmup-steps", "1",
                "--holdout", "0.25",
                "--seed", "5",
            ]
            first = subprocess.run(
                [*command, "--epochs", "1"], capture_output=True, text=True
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue((checkpoints / "latest.pt").exists())
            self.assertTrue((checkpoints / "best.pt").exists())

            resumed = subprocess.run(
                [*command, "--epochs", "2"], capture_output=True, text=True
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("resumed from epoch 0", resumed.stdout)
            history = json.loads((checkpoints / "history.json").read_text())
            self.assertEqual([row["epoch"] for row in history], [0, 1])


if __name__ == "__main__":
    unittest.main()

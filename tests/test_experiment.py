import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trajectory_verification.adapters.womd_proto import Scenario
from trajectory_verification.experiment import load_experiment_manifest


def write_shard(path: Path) -> None:
    scenario = Scenario()
    scenario.scenario_id = "experiment-scenario"
    scenario.timestamps_seconds.extend(index * 0.1 for index in range(91))
    scenario.current_time_index = 10
    scenario.sdc_track_index = 0
    track = scenario.tracks.add()
    track.id = 42
    track.object_type = 1
    for index in range(91):
        state = track.states.add()
        state.center_x = float(index)
        state.center_y = 0.0
        state.velocity_x = 10.0
        state.velocity_y = 0.0
        state.valid = True
    required = scenario.tracks_to_predict.add()
    required.track_index = 0
    payload = scenario.SerializeToString()
    path.write_bytes(
        struct.pack("<Q", len(payload)) + b"\0" * 4 + payload + b"\0" * 4
    )


def manifest(shard: str = "scenario.tfrecord") -> dict[str, object]:
    return {
        "experiment_id": "test-experiment",
        "dataset": {
            "name": "fixture",
            "version": "1",
            "shards": [shard],
        },
        "candidates": [
            {"name": "baseline", "model": "constant_velocity"},
            {"name": "candidate", "model": "kinematic_ensemble"},
        ],
        "evaluation": {
            "miss_threshold_m": 2.0,
            "bootstrap_samples": 0,
            "bootstrap_seed": 3,
        },
        "comparison": {
            "baseline": "baseline",
            "candidate": "candidate",
            "bootstrap_samples": 100,
            "bootstrap_seed": 5,
            "policy": {},
        },
        "outputs": {"directory": "artifacts"},
    }


class ExperimentTests(unittest.TestCase):
    def test_rejects_unsafe_candidate_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = manifest()
            payload["candidates"][0]["name"] = "../escape"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "candidate name"):
                load_experiment_manifest(path)

    def test_one_command_writes_indexed_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_shard(root / "scenario.tfrecord")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest()))
            environment = dict(os.environ, TVS_SOURCE_REVISION="verified-test-sha")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "trajectory_verification.experiment_cli",
                    str(manifest_path),
                ],
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            terminal = json.loads(completed.stdout)
            self.assertTrue(terminal["result"]["gate_passed"])
            index_path = root / "artifacts" / "experiment-index.json"
            index = json.loads(index_path.read_text())
            self.assertEqual(index["source_revision"]["commit"], "verified-test-sha")
            self.assertEqual(index["dataset"]["scenario_count"], 1)
            self.assertEqual(len(index["artifacts"]), 17)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in index["artifacts"]))
            self.assertTrue((root / "artifacts" / "comparison.html").exists())
            self.assertTrue(
                (root / "artifacts" / "candidate-evaluation.json").exists()
            )
            risk = json.loads(
                (root / "artifacts" / "candidate-risk.json").read_text()
            )
            self.assertEqual(risk["summary"]["agents"], 1)
            self.assertIn("motion_class", risk["summary"])


if __name__ == "__main__":
    unittest.main()

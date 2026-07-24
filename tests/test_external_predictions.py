import json
import tempfile
import unittest
from pathlib import Path

from trajectory_verification.adapters.external_predictions import (
    load_external_predictions,
)
from trajectory_verification.models import AgentTrack, Scenario, State


def scenario() -> Scenario:
    timestamps = tuple(index * 0.1 for index in range(91))
    return Scenario(
        "external-scenario",
        (AgentTrack("42", tuple(
            State(time_s, float(index), 0.0)
            for index, time_s in enumerate(timestamps)
        ), "vehicle"),),
        current_time_index=10,
        tracks_to_predict=("42",),
        timestamps_s=timestamps,
    )


def artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provenance": {
            "model_name": "Learned Fixture",
            "model_version": "1.0",
            "source_repository": "https://example.test/model",
            "source_revision": "abc123",
            "checkpoint_sha256": "a" * 64,
            "coordinate_frame": "scenario_global",
            "future_data_used": False,
        },
        "predictions": [{
            "scenario_id": "external-scenario",
            "agents": [{
                "agent_id": "42",
                "modes": [
                    {
                        "confidence": 2.0,
                        "xy_m": [[float(step), 0.0] for step in range(16)],
                    },
                    {
                        "confidence": 1.0,
                        "xy_m": [[float(step), 1.0] for step in range(16)],
                    },
                ],
            }],
        }],
    }


class ExternalPredictionTests(unittest.TestCase):
    def test_validates_and_normalizes_external_model_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "external.json"
            source.write_text(json.dumps(artifact()))
            loaded = load_external_predictions(source, (scenario(),))
            modes = loaded.predictions[0].agents[0].trajectories
            self.assertAlmostEqual(modes[0].confidence, 2 / 3)
            self.assertAlmostEqual(modes[1].confidence, 1 / 3)
            self.assertEqual(len(loaded.predictions[0].agents), 1)
            self.assertEqual(len(modes[0].points), 16)

    def test_rejects_future_ground_truth_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.json"
            payload = artifact()
            payload["provenance"]["future_data_used"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not admissible"):
                load_external_predictions(path, (scenario(),))

    def test_rejects_non_string_or_uppercase_checkpoint_provenance(self):
        for checksum in (123, "A" * 64):
            with self.subTest(checksum=checksum):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "external.json"
                    payload = artifact()
                    payload["provenance"]["checkpoint_sha256"] = checksum
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(
                        ValueError, "text fields must be strings|lowercase hex"
                    ):
                        load_external_predictions(path, (scenario(),))

    def test_rejects_missing_target_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.json"
            payload = artifact()
            payload["predictions"][0]["agents"] = []
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "contains no agents"):
                load_external_predictions(path, (scenario(),))

    def test_rejects_non_finite_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.json"
            payload = artifact()
            payload["predictions"][0]["agents"][0]["modes"][0]["xy_m"][0][0] = "NaN"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_external_predictions(path, (scenario(),))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from trajectory_verification.io import load_scenario_manifest


SCENARIO = {"scenario_id": "a", "tracks": [{"agent_id": "ego", "states": [
    {"time_s": 0, "x_m": 0, "y_m": 0},
]}]}


class ManifestTests(unittest.TestCase):
    def test_relative_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenario.json").write_text(json.dumps(SCENARIO))
            (root / "manifest.json").write_text(json.dumps({"scenarios": ["scenario.json"]}))
            self.assertEqual(load_scenario_manifest(root / "manifest.json")[0].scenario_id, "a")

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenario.json").write_text(json.dumps(SCENARIO))
            (root / "manifest.json").write_text(json.dumps(
                {"scenarios": ["scenario.json", "scenario.json"]}
            ))
            with self.assertRaisesRegex(ValueError, "duplicate scenario"):
                load_scenario_manifest(root / "manifest.json")


if __name__ == "__main__":
    unittest.main()

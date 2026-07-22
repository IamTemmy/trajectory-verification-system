import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trajectory_verification.adapters.womd_proto import Scenario
from trajectory_verification.womd_verify_cli import main


class WOMDVerifyCLITests(unittest.TestCase):
    def test_verifies_scenario_with_role_selectors_and_writes_reports(self):
        message = Scenario(
            scenario_id="womd-cli-fixture",
            timestamps_seconds=[0.0, 1.0],
            current_time_index=0,
            sdc_track_index=0,
        )
        ego = message.tracks.add(id=100, object_type=1)
        ego.states.add(center_x=0.0, center_y=0.0, valid=True)
        ego.states.add(center_x=5.0, center_y=0.0, valid=True)
        other = message.tracks.add(id=200, object_type=1)
        other.states.add(center_x=20.0, center_y=0.0, valid=True)
        other.states.add(center_x=24.0, center_y=0.0, valid=True)
        message.tracks_to_predict.add(track_index=1)
        message.map_features.add()
        payload = message.SerializeToString()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "fixture.tfrecord"
            shard.write_bytes(
                struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload + b"\0\0\0\0"
            )
            requirements = root / "requirements.json"
            requirements.write_text(json.dumps({"requirements": [{
                "id": "SPEED", "description": "Limit SDC speed",
                "metric": "speed", "operator": "less_than_or_equal",
                "threshold": 10, "units": "m/s", "subject_agent_id": "@sdc"
            }]}), encoding="utf-8")
            markdown, html, svg = root / "report.md", root / "report.html", root / "scene.svg"
            argv = [
                "verify-womd", str(shard), str(requirements),
                "--markdown-report", str(markdown), "--html-report", str(html),
                "--svg-output", str(svg),
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as output:
                exit_code = main()

            self.assertEqual(0, exit_code)
            self.assertIn('"passed": true', output.getvalue())
            self.assertTrue(markdown.exists())
            self.assertTrue(html.exists())
            self.assertTrue(svg.exists())
            self.assertIn("womd-cli-fixture", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trajectory_verification.adapters.motion_submission import OFFICIAL_PREDICTION_STEPS
from trajectory_verification.adapters.motion_submission_proto import MotionChallengeSubmission
from trajectory_verification.adapters.womd_proto import Scenario


def write_fixture(directory: Path) -> tuple[Path, Path]:
    scenario = Scenario()
    scenario.scenario_id = "scenario-cli"
    scenario.timestamps_seconds.extend(index * 0.1 for index in range(91))
    scenario.current_time_index = 10
    scenario.sdc_track_index = 0
    track = scenario.tracks.add(); track.id = 42; track.object_type = 1
    for index in range(91):
        state = track.states.add()
        state.center_x = float(index); state.center_y = 0.0; state.valid = True
    required = scenario.tracks_to_predict.add(); required.track_index = 0
    payload = scenario.SerializeToString()
    shard = directory / "scenario.tfrecord"
    shard.write_bytes(struct.pack("<Q", len(payload)) + b"\0" * 4 + payload + b"\0" * 4)

    submission = MotionChallengeSubmission()
    item = submission.scenario_predictions.add(); item.scenario_id = "scenario-cli"
    agent = item.single_predictions.predictions.add(); agent.object_id = 42
    mode = agent.trajectories.add(); mode.confidence = 1.0
    mode.trajectory.center_x.extend(float(step) for step in OFFICIAL_PREDICTION_STEPS)
    mode.trajectory.center_y.extend(0.0 for _ in OFFICIAL_PREDICTION_STEPS)
    submission_path = directory / "submission.binproto"
    submission_path.write_bytes(submission.SerializeToString())
    return submission_path, shard


class MotionEvaluateCliTests(unittest.TestCase):
    def test_end_to_end_evaluation_writes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission, shard = write_fixture(root)
            command = [
                sys.executable, "-m", "trajectory_verification.motion_evaluate_cli",
                str(submission), str(shard),
                "--json-report", str(root / "report.json"),
                "--markdown-report", str(root / "report.md"),
                "--html-report", str(root / "report.html"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["summary"]["mean_min_ade_m"], 0.0)
            self.assertEqual(report["summary"]["miss_rate"], 0.0)
            self.assertTrue((root / "report.md").exists())
            self.assertTrue((root / "report.html").exists())


if __name__ == "__main__":
    unittest.main()

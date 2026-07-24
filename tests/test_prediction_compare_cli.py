import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def report(ade):
    return {
        "summary": {
            "mean_min_ade_m": ade,
            "mean_min_fde_m": ade * 2,
            "miss_rate": 0.5,
            "mean_ground_truth_coverage": 1.0,
        },
        "scenarios": [{"scenario_id": "a", "agents": [{"agent_id": "ego"}]}],
    }


class PredictionCompareCliTests(unittest.TestCase):
    def test_writes_passing_comparison_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(report(4.0)))
            candidate.write_text(json.dumps(report(3.0)))
            completed = subprocess.run([
                sys.executable, "-m", "trajectory_verification.prediction_compare_cli",
                str(baseline), str(candidate),
                "--json-report", str(root / "comparison.json"),
                "--markdown-report", str(root / "comparison.md"),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads((root / "comparison.json").read_text())["gate_passed"])
            self.assertIn("Gate result:** PASS", (root / "comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()

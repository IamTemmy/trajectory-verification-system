import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RegressionCliTests(unittest.TestCase):
    def test_demo_fails_and_writes_reports(self):
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = [
                sys.executable, "-m", "trajectory_verification.regression_cli",
                str(root / "examples/regression/baseline_manifest.json"),
                str(root / "examples/regression/candidate_manifest.json"),
                str(root / "examples/regression/requirements.json"),
                "--policy", str(root / "examples/regression/policy.json"),
                "--json-report", str(output / "report.json"),
                "--markdown-report", str(output / "report.md"),
                "--html-report", str(output / "report.html"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertFalse(json.loads((output / "report.json").read_text())["gate_passed"])
            self.assertTrue((output / "report.md").exists())
            self.assertTrue((output / "report.html").exists())


if __name__ == "__main__":
    unittest.main()

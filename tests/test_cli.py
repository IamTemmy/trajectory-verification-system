import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class VerifyTrajectoriesCliTests(unittest.TestCase):
    """Cover the documented intersection example end to end.

    The README presents this scenario as the project's runnable demonstration,
    so it must keep failing on the separation requirement and must keep writing
    every artifact the README links.
    """

    def test_intersection_example_fails_and_writes_all_artifacts(self):
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = [
                sys.executable, "-m", "trajectory_verification.cli",
                str(root / "examples/intersection_scenario.json"),
                str(root / "examples/intersection_requirements.json"),
                "--markdown-report", str(output / "report.md"),
                "--html-report", str(output / "report.html"),
                "--svg-output", str(output / "scenario.svg"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1, completed.stderr)

            markdown = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("INTERSECTION_SEPARATION_001", markdown)
            self.assertIn("5.4 s to 5.8 s", markdown)
            self.assertTrue((output / "report.html").exists())

            svg = (output / "scenario.svg").read_text(encoding="utf-8")
            self.assertTrue(svg.startswith("<svg"))
            for agent_id in ("through_vehicle", "turning_vehicle", "southbound_vehicle",
                             "cyclist", "pedestrian"):
                self.assertIn(agent_id, svg)


if __name__ == "__main__":
    unittest.main()

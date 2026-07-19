"""Regression test for the per-camera automatic table-origin calibration."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "sample.mp4"
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_video import ensure_calibration  # noqa: E402


@unittest.skipUnless(VIDEO.exists(), "sample.mp4 is a local video fixture")
class AutoCalibrationTest(unittest.TestCase):
    def test_analyzer_creates_a_cache_without_spawning_the_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "calibration.json"
            args = SimpleNamespace(
                calibration=None,
                calibration_cache=str(cache),
                video=str(VIDEO),
                start_seconds=0,
            )

            self.assertEqual(ensure_calibration(args, 60), str(cache))
            self.assertTrue(cache.exists())

    def test_first_frame_finds_verified_table_origin(self):
        """The origin stays at the white-center-stripe/net-base intersection."""
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "calibration.json"
            result = subprocess.run(
                [sys.executable, "scripts/auto_calibrate.py", str(VIDEO), "--output", str(cache)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            reported = json.loads(result.stdout)
            calibration = json.loads(cache.read_text())
        # This point was visually approved in artifacts/auto_grid_check.png.
        # Tolerance allows small OpenCV/Hough implementation differences.
        width, height = calibration["image_size"]
        self.assertAlmostEqual(reported["table_center"][0] / width, 1232 / 4096, delta=.01)
        self.assertAlmostEqual(reported["table_center"][1] / height, 1004 / 2160, delta=.01)
        # The far-left lower rail is occluded in this view. The automatic
        # path must leave it unknown instead of extending the table to x=0.
        self.assertEqual(len(calibration["table_polygon"]), 3)

    def test_known_table_contacts_are_not_regressed_to_unknown(self):
        """Keep the three manually confirmed sample contacts detectable."""
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "calibration.json"
            output = Path(directory) / "attempts.jsonl"
            subprocess.run(
                [sys.executable, "scripts/auto_calibrate.py", str(VIDEO), "--output", str(cache)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/analyze_video.py", str(VIDEO), "--calibration", str(cache),
                 "--output", str(output), "--no-annotated", "--end-seconds", "18"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            events = [json.loads(line) for line in output.read_text().splitlines()]
        # User-verified contacts: about 4.5s (middle), 6s (close side), and
        # 17.1s (ball visibly converged with its shadow). The frames are
        # deliberately exact for this checked-in fixture; use a different
        # fixture/cache for a moved camera.
        for frame in (258, 354, 1027):
            event = next(item for item in events if item["frame_number"] == frame)
            self.assertTrue(event["hit_table"], f"frame {frame} should be a table contact")

        event = next(item for item in events if item["frame_number"] == 1027)
        self.assertEqual(event["outcome"], "far_table")
        self.assertTrue(event["is_in"])
        self.assertAlmostEqual(event["posx"], 0.0043, delta=.03)
        self.assertAlmostEqual(event["posz"], 0.7473, delta=.05)


if __name__ == "__main__":
    unittest.main()

"""Regression test for the per-camera automatic table-origin calibration."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "sample.mp4"
SAMPLE2_VIDEO = ROOT / "sample2-trimmed-58s.mp4"
SAMPLE2_CALIBRATION = ROOT / "artifacts" / "sample2-auto.table-calibration.json"
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_video import ensure_calibration  # noqa: E402
from auto_calibrate import detect_geometry  # noqa: E402


class WideViewCalibrationTest(unittest.TestCase):
    def test_room_lines_do_not_replace_table_boundaries(self):
        frame = np.full((540, 1024, 3), 35, dtype=np.uint8)
        for y in (50, 100, 450, 500):
            cv2.line(frame, (0, y), (1023, y), (220, 220, 220), 3)
        table = np.int32([[250, 210], [675, 210], [805, 370], [50, 370]])
        cv2.fillConvexPoly(frame, table, (50, 170, 60))
        cv2.polylines(frame, [table], True, (230, 230, 230), 4)
        cv2.line(frame, (150, 270), (724, 270), (230, 230, 230), 3)
        cv2.line(frame, (460, 210), (410, 370), (15, 15, 15), 10)
        cv2.line(frame, (466, 210), (416, 370), (240, 240, 240), 3)

        polygon, center, _ = detect_geometry(frame)

        self.assertEqual(len(polygon), 4)
        self.assertAlmostEqual(polygon[0][1], 210, delta=5)
        self.assertAlmostEqual(polygon[2][1], 370, delta=5)
        self.assertAlmostEqual(center[1], 270, delta=5)


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
        self.assertEqual(event["outcome"], "hit")
        self.assertTrue(event["is_in"])
        self.assertAlmostEqual(event["posx"], 0.0043, delta=.03)
        self.assertAlmostEqual(event["posz"], 0.7473, delta=.05)

    def test_complete_sample_ordered_hit_sequence(self):
        expected = "out out hit hit miss hit miss miss out out hit miss".split()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.jsonl"
            subprocess.run(
                [sys.executable, "scripts/analyze_video.py", str(VIDEO),
                 "--calibration", str(ROOT / "artifacts" / "sample_auto_calibration.json"),
                 "--output", str(output), "--no-annotated"],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            actual = [json.loads(line)["outcome"] for line in output.read_text().splitlines()]

        self.assertEqual(len(actual), len(expected))
        self.assertEqual(
            [item == "hit" for item in actual],
            [item == "hit" for item in expected],
        )


@unittest.skipUnless(
    SAMPLE2_VIDEO.exists() and SAMPLE2_CALIBRATION.exists(),
    "sample2 video and calibration are local fixtures",
)
class Sample2OrderedRegressionTest(unittest.TestCase):
    def test_every_machine_launch_has_the_labeled_ordered_result(self):
        expected = (
            "hit hit out out hit out hit out hit out hit hit hit hit out "
            "hit hit hit hit out hit hit hit hit hit hit hit hit hit hit "
            "hit hit hit hit out hit hit hit hit hit hit hit hit hit hit "
            "hit hit miss"
        ).split()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample2.jsonl"
            subprocess.run(
                [sys.executable, "scripts/analyze_video.py", str(SAMPLE2_VIDEO),
                 "--calibration", str(SAMPLE2_CALIBRATION),
                 "--output", str(output), "--no-annotated"],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            actual = [json.loads(line)["outcome"] for line in output.read_text().splitlines()]

        self.assertEqual(len(actual), 48, "one result is required for every launch")
        # The user explicitly treats a visible out and a fully occluded miss
        # as equivalent non-hits; table contacts must still match every ball.
        self.assertEqual(
            [item == "hit" for item in actual],
            [item == "hit" for item in expected],
        )


if __name__ == "__main__":
    unittest.main()

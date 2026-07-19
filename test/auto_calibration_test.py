"""Regression test for the per-camera automatic table-origin calibration."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "sample.mp4"
SAMPLE2_VIDEO = ROOT / "sample2-trimmed-58s.mp4"
SAMPLE3_VIDEO = ROOT / "sample3-trimmed-44s.mp4"
sys.path.insert(0, str(ROOT / "scripts"))
from auto_calibrate import (  # noqa: E402
    calibrated_tracking_regions,
    calibration_from_frame,
    detect_geometry,
)


def first_frame_calibration(video: Path):
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise AssertionError(f"could not read {video}")
    return calibration_from_frame(frame)


def normalized(points, width, height):
    return [[x / width, y / height] for x, y in points]


def assert_points_close(test, actual, expected, delta=.01):
    test.assertEqual(len(actual), len(expected))
    for actual_point, expected_point in zip(actual, expected):
        test.assertAlmostEqual(actual_point[0], expected_point[0], delta=delta)
        test.assertAlmostEqual(actual_point[1], expected_point[1], delta=delta)


class WideViewCalibrationTest(unittest.TestCase):
    def test_tracking_regions_follow_camera_orientation_and_exclude_room_edges(self):
        regions = calibrated_tracking_regions(
            [1000, 500],
            [[250, 250], [675, 250], [805, 330], [50, 330]],
            (150, 270),
            (724, 270),
        )

        self.assertLess(regions["return_region"][0], regions["launcher_region"][0])
        corridor = np.float32(regions["tracking_polygon"])
        self.assertGreaterEqual(cv2.pointPolygonTest(corridor, (50, 330), False), 0)
        self.assertLess(cv2.pointPolygonTest(corridor, (990, 20), False), 0)
        self.assertLess(cv2.pointPolygonTest(corridor, (500, 490), False), 0)

        reversed_regions = calibrated_tracking_regions(
            [1000, 500],
            [[195, 250], [950, 330], [325, 250], [50, 330]],
            (850, 270),
            (276, 270),
        )
        self.assertGreater(
            reversed_regions["return_region"][0],
            reversed_regions["launcher_region"][0],
        )

    def test_room_lines_do_not_replace_table_boundaries(self):
        frame = np.full((540, 1024, 3), 35, dtype=np.uint8)
        sky_green = cv2.cvtColor(
            np.uint8([[[90, 90, 235]]]), cv2.COLOR_HSV2BGR,
        )[0, 0]
        for x in range(20, 1000, 90):
            cv2.rectangle(frame, (x, 15), (x + 55, 145), sky_green.tolist(), -1)
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


@unittest.skipUnless(
    VIDEO.exists() and SAMPLE2_VIDEO.exists() and SAMPLE3_VIDEO.exists(),
    "all three sample videos are local fixtures",
)
class AutomaticGeometryRegressionTest(unittest.TestCase):
    def test_first_frame_geometry_for_every_camera_view(self):
        cases = [
            (
                VIDEO, [4096, 2160], [.2993, .4685],
                [[.0504, .2389], [.6406, .2389], [.7595, .8870]],
                [[.3459, .2389], [.2146, .8870]],
            ),
            (
                SAMPLE2_VIDEO, [1024, 540], [.4375, .5000],
                [[.2446, .3981], [.6569, .3981], [.7751, .6796], [.0503, .6796]],
                [[.4507, .3981], [.4138, .6796]],
            ),
            (
                SAMPLE3_VIDEO, [4096, 2160], [.4626, .5481],
                [[.2935, .4565], [.6516, .4565], [.6587, .6981], [.1574, .6981]],
                [[.4684, .4565], [.4534, .6981]],
            ),
        ]
        for video, image_size, expected_center, expected_table, expected_net in cases:
            with self.subTest(video=video.name):
                calibration, center = first_frame_calibration(video)
                width, height = calibration["image_size"]

                self.assertEqual(calibration["image_size"], image_size)
                assert_points_close(
                    self, [[center[0] / width, center[1] / height]], [expected_center],
                )
                assert_points_close(
                    self, normalized(calibration["table_polygon"], width, height),
                    expected_table,
                )
                assert_points_close(
                    self, normalized(calibration["net_line"], width, height),
                    expected_net,
                )


@unittest.skipUnless(VIDEO.exists(), "sample.mp4 is a local video fixture")
class AutoCalibrationTest(unittest.TestCase):
    def test_first_frame_calibration_stays_in_memory(self):
        calibration, _ = first_frame_calibration(VIDEO)

        self.assertTrue(calibration["auto_calibrated"])
        self.assertNotIn("diagnostic", calibration)
        self.assertNotIn("detector_settings", calibration)
        self.assertIn("launcher_region", calibration)
        self.assertIn("return_region", calibration)
        self.assertIn("tracking_polygon", calibration)
        assert_points_close(
            self,
            normalized(calibration["table_contact_polygon"], *calibration["image_size"]),
            normalized(calibration["table_polygon"], *calibration["image_size"]),
            delta=1e-6,
        )

    def test_first_frame_finds_verified_table_origin(self):
        """The origin stays at the white-center-stripe/net-base intersection."""
        calibration, center = first_frame_calibration(VIDEO)
        # This point was visually approved in artifacts/auto_grid_check.png.
        # Tolerance allows small OpenCV/Hough implementation differences.
        width, height = calibration["image_size"]
        self.assertAlmostEqual(center[0] / width, .2993, delta=.01)
        self.assertAlmostEqual(center[1] / height, .4685, delta=.01)
        # The far-left lower rail is occluded in this view. The automatic
        # path must leave it unknown instead of extending the table to x=0.
        self.assertEqual(len(calibration["table_polygon"]), 3)

    def test_known_table_contacts_are_not_regressed_to_unknown(self):
        """Keep the three manually confirmed sample contacts detectable."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempts.jsonl"
            subprocess.run(
                [sys.executable, "scripts/analyze_video.py", str(VIDEO),
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
        # fixture for a moved camera.
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
                 "--output", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            actual = [json.loads(line)["outcome"] for line in output.read_text().splitlines()]
            self.assertEqual({path.name for path in Path(directory).iterdir()}, {"sample.jsonl"})

        self.assertEqual(len(actual), len(expected))
        self.assertEqual(
            [item == "hit" for item in actual],
            [item == "hit" for item in expected],
        )


@unittest.skipUnless(
    SAMPLE2_VIDEO.exists(),
    "sample2 video is a local fixture",
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
                 "--output", str(output), "--no-annotated"],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            actual = [json.loads(line)["outcome"] for line in output.read_text().splitlines()]
            self.assertEqual({path.name for path in Path(directory).iterdir()}, {"sample2.jsonl"})

        self.assertEqual(len(actual), 48, "one result is required for every launch")
        # The user explicitly treats a visible out and a fully occluded miss
        # as equivalent non-hits; table contacts must still match every ball.
        self.assertEqual(
            [item == "hit" for item in actual],
            [item == "hit" for item in expected],
        )


@unittest.skipUnless(SAMPLE3_VIDEO.exists(), "sample3 video is a local fixture")
class Sample3UnhintedRegressionTest(unittest.TestCase):
    def test_auto_calibrated_video_has_every_labeled_hit_in_order(self):
        expected = (
            "miss miss hit hit miss hit hit hit miss hit hit hit hit hit hit miss "
            "hit hit hit hit hit hit hit miss hit miss hit hit hit hit hit hit"
        ).split()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample3.jsonl"
            subprocess.run(
                [sys.executable, "scripts/analyze_video.py", str(SAMPLE3_VIDEO),
                 "--output", str(output), "--no-annotated"],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            actual = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({path.name for path in Path(directory).iterdir()}, {"sample3.jsonl"})

        self.assertEqual(len(actual), 32, "one result is required for every launch")
        self.assertEqual(
            [item["outcome"] == "hit" for item in actual],
            [item == "hit" for item in expected],
        )


if __name__ == "__main__":
    unittest.main()

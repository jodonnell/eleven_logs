"""Regression test for the per-camera automatic table-origin calibration."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from auto_calibrate import (  # noqa: E402
    calibrated_tracking_regions,
    colored_table_extent,
    detect_geometry,
    infer_ball_color,
)


class AutomaticCalibrationPrimitiveTest(unittest.TestCase):
    def test_colored_table_extent_finds_blue_table_among_warm_backgrounds(self):
        frame = np.full((300, 500, 3), (35, 75, 130), dtype=np.uint8)
        blue = cv2.cvtColor(
            np.uint8([[[120, 190, 190]]]), cv2.COLOR_HSV2BGR,
        )[0, 0]
        table = np.int32([[210, 80], [310, 90], [390, 245], [80, 225]])
        cv2.fillConvexPoly(frame, table, blue.tolist())

        top, bottom, mask, profile = colored_table_extent(frame)

        self.assertAlmostEqual(profile["hue_center"], 120, delta=2)
        self.assertLessEqual(top, 90)
        self.assertGreaterEqual(bottom, 220)
        self.assertTrue(mask[160, 250])

    def test_ball_color_comes_from_directed_moving_track(self):
        blue = cv2.cvtColor(
            np.uint8([[[120, 190, 180]]]), cv2.COLOR_HSV2BGR,
        )[0, 0]
        orange = cv2.cvtColor(
            np.uint8([[[15, 235, 245]]]), cv2.COLOR_HSV2BGR,
        )[0, 0]
        frames = []
        for index in range(20):
            frame = np.zeros((120, 200, 3), dtype=np.uint8)
            cv2.fillConvexPoly(
                frame, np.int32([[80, 35], [160, 35], [180, 110], [20, 110]]),
                blue.tolist(),
            )
            cv2.circle(frame, (155 - index * 5, 25 + index * 3), 3, orange.tolist(), -1)
            frames.append(frame)
        calibration = {
            "image_size": [200, 120],
            "tracking_polygon": [[0, 0], [199, 0], [199, 119], [0, 119]],
            "table_polygon": [[80, 35], [160, 35], [180, 110], [20, 110]],
            "table_color": {
                "hue_center": 120, "hue_tolerance": 14,
                "min_saturation": 100, "min_value": 60,
            },
            "control_points": [
                {"name": "x0_player_edge", "image": [60, 90]},
                {"name": "x0_opponent_edge", "image": [155, 25]},
            ],
        }

        profile = infer_ball_color(frames, calibration)

        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile["hue_center"], 15, delta=2)
        self.assertGreater(profile["min_saturation"], 150)

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


if __name__ == "__main__":
    unittest.main()

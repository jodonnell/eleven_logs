"""Focused unit tests for the classical-CV bounce helpers."""
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_video import (  # noqa: E402
    AttemptClassifier,
    DetectorSettings,
    MultiBallTracker,
    find_bounce,
    map_log_coordinate,
    shadow_contact_score,
)


class VideoDetectorUnitTest(unittest.TestCase):
    def classifier(self) -> AttemptClassifier:
        calibration = {
            "table_surface_y": 0.7786086,
            "launcher_region": [580, 0, 950, 300],
        }
        return AttemptClassifier(
            fps=60,
            calibration=calibration,
            table=np.float32([(0, 0), (200, 0), (200, 200), (0, 200)]),
            net_line=np.float32([(500, 0), (500, 500)]),
            occlusion=np.float32([]),
            homography=np.eye(3, dtype=np.float32),
            video_width=1000,
            video_height=500,
            scale=1,
            settings=DetectorSettings(),
        )

    def test_shadow_score_rises_for_dark_table_patch_below_ball(self):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, :] = (65, 165, 160)  # green table
        hsv[55:70, 42:58, 2] = 105      # ball's dark table shadow

        self.assertGreater(shadow_contact_score(hsv, (50, 50)), 30)

    def test_shadow_score_is_zero_on_evenly_lit_table(self):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, :] = (65, 165, 160)

        self.assertEqual(shadow_contact_score(hsv, (50, 50)), 0.0)

    def test_shadow_contact_is_a_bounce_away_from_net(self):
        points = [(frame, 200 + frame * 4, 220, 0.0) for frame in range(9)]
        points[4] = (4, 216, 220, 32.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])

        hit, _, _ = find_bounce(points, table, net_line=np.float32([(0, 0), (0, 500)]))

        self.assertEqual(hit[0], 4)

    def test_net_mesh_darkening_is_not_a_shadow_bounce(self):
        points = [(frame, 45 + frame, 220, 0.0) for frame in range(9)]
        points[4] = (4, 49, 220, 80.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])

        self.assertIsNone(find_bounce(points, table, net_line=np.float32([(50, 0), (50, 500)])))

    def test_classifier_reports_an_off_table_return(self):
        classifier = self.classifier()
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.process_tracks([launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)
        classifier.finish_attempt(draw_frame=40)

        self.assertEqual(len(classifier.events), 1)
        event = classifier.events[0]
        self.assertEqual(event.outcome, "off_table")
        self.assertFalse(event.hit_table)
        self.assertNotIn("pixel", event.to_record())

    def test_classifier_reports_crossed_net_return_that_ends_off_table(self):
        classifier = self.classifier()
        classifier.net_line = np.float32([(150, 0), (150, 500)])
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.process_tracks([launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)
        classifier.finish_attempt(draw_frame=40)

        event = classifier.events[0]
        self.assertTrue(event.return_crossed_net)
        self.assertEqual(event.outcome, "off_table")
        self.assertFalse(event.hit_table)

    def test_default_launcher_region_rejects_table_and_frame_edge_tracks(self):
        classifier = AttemptClassifier(
            fps=60,
            calibration={"table_surface_y": 0.7786086},
            table=np.float32([(250, 200), (675, 200), (805, 370), (50, 370)]),
            net_line=np.float32([(500, 0), (500, 500)]),
            occlusion=np.float32([]),
            homography=np.eye(3, dtype=np.float32),
            video_width=1000,
            video_height=500,
            scale=1,
            settings=DetectorSettings(),
        )
        launch = [(frame, 800 - frame * 10, 205, 0.0) for frame in range(18)]
        lower_table_edge = [(frame, 800 - frame * 10, 300, 0.0) for frame in range(18)]
        outer_frame_edge = [(frame, 980 - frame * 10, 100, 0.0) for frame in range(18)]

        self.assertTrue(classifier.is_reportable_launcher_track(launch))
        self.assertFalse(classifier.is_reportable_launcher_track(lower_table_edge))
        self.assertFalse(classifier.is_reportable_launcher_track(outer_frame_edge))

        classifier.process_tracks([lower_table_edge], draw_frame=18)
        classifier.finish_attempt(draw_frame=19)
        self.assertEqual(classifier.events, [])

    def test_identity_homography_maps_pixel_to_table_coordinate(self):
        self.assertEqual(
            map_log_coordinate(np.eye(3, dtype=np.float32), (2.5, 4.0), 0.7786086),
            (2.5, 0.7786, 4.0),
        )

    def test_calibration_can_override_detector_settings(self):
        settings = DetectorSettings.from_calibration({"detector_settings": {"motion_threshold": 9}})

        self.assertEqual(settings.motion_threshold, 9)
        self.assertEqual(settings.max_gap, 3)

    def test_tracker_completes_a_path_after_the_allowed_gap(self):
        tracker = MultiBallTracker(DetectorSettings(max_gap=1))

        self.assertEqual(tracker.update(0, [(10, 20, 0)]), [])
        self.assertEqual(tracker.update(1, []), [])

        self.assertEqual(tracker.update(2, []), [[(0, 10, 20, 0)]])


if __name__ == "__main__":
    unittest.main()

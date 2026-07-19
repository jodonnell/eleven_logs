"""Focused unit tests for the classical-CV bounce helpers."""
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_video import (  # noqa: E402
    Attempt,
    AttemptClassifier,
    DetectorSettings,
    MultiBallTracker,
    find_bounce,
    map_log_coordinate,
    normalize_attempt_events,
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

    def test_terminal_shadow_contact_can_finish_a_return_track(self):
        points = [(frame, 100 + frame * 20, 220, 0.0) for frame in range(9)]
        points[-1] = (8, 260, 220, 32.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])

        hit, _, departure = find_bounce(
            points, table, net_line=np.float32([(0, 0), (0, 500)])
        )

        self.assertEqual(hit[0], 8)
        self.assertEqual(departure, [])

    def test_ball_radius_allows_a_contact_just_beyond_the_table_edge(self):
        points = [(frame, 180 + frame * 20, 100 + abs(4 - frame) * -10, 0.0) for frame in range(9)]
        table = np.float32([(0, 0), (255, 0), (255, 500), (0, 500)])

        hit, _, _ = find_bounce(
            points, table, net_line=np.float32([(0, 0), (0, 500)])
        )

        self.assertEqual(hit[0], 4)

    def test_shadow_plateau_after_ball_disappears_is_not_a_contact(self):
        points = [(frame, 100 + frame * 20, 100 + frame * 5, 0.0) for frame in range(9)]
        points[5] = (5, 200, 125, 33.0)
        points[6] = (6, 198, 125, 33.0)
        points[7] = (7, 196, 125, 34.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])

        self.assertIsNone(
            find_bounce(points, table, net_line=np.float32([(0, 0), (0, 500)]))
        )

    def test_backward_tracker_handoff_is_not_a_trajectory_bounce(self):
        points = [(frame, 100 + frame * 20, 100 + frame * 10, 0.0) for frame in range(9)]
        points[4] = (4, 140, 150, 0.0)
        points[5] = (5, 138, 145, 0.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])

        self.assertIsNone(
            find_bounce(points, table, net_line=np.float32([(0, 0), (0, 500)]))
        )

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

    def test_attempt_emits_a_later_miss_after_an_earlier_bounce(self):
        classifier = self.classifier()
        classifier.net_line = np.float32([(150, 0), (150, 500)])
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        bounced = [
            (30, 100, 100, 0.0),
            (31, 120, 110, 0.0),
            (32, 140, 120, 0.0),
            (33, 160, 130, 0.0),
            (34, 180, 120, 0.0),
            (35, 200, 110, 0.0),
            (36, 220, 100, 0.0),
            (37, 240, 90, 0.0),
            (38, 260, 80, 0.0),
        ]
        missed = [(50 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.process_tracks([launch], draw_frame=18)
        classifier.process_tracks([bounced], draw_frame=39)
        classifier.process_tracks([missed], draw_frame=59)
        classifier.finish_attempt(draw_frame=60)

        self.assertEqual([event.outcome for event in classifier.events], ["far_table", "off_table"])

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

    def test_decisive_return_reports_for_unreportable_launcher(self):
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
        outer_frame_launch = [(frame, 980 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 100, 250, 0.0) for frame in range(9)]

        classifier.process_tracks([outer_frame_launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)
        classifier.finish_attempt(draw_frame=40)

        self.assertFalse(classifier.is_reportable_launcher_track(outer_frame_launch))
        self.assertEqual(len(classifier.events), 1)
        self.assertEqual(classifier.events[0].outcome, "off_table")

    def test_return_recovers_after_a_stale_bright_object_prefix(self):
        classifier = self.classifier()
        path = [
            (0, 700, 100, 0.0),
            (1, 705, 100, 0.0),
            (2, 710, 100, 0.0),
            (3, 200, 120, 0.0),
            (4, 275, 125, 0.0),
            (5, 350, 130, 0.0),
            (6, 425, 135, 0.0),
        ]

        returned = classifier.return_segment(path)

        self.assertIsNotNone(returned)
        self.assertEqual(returned[0][0], 3)
        self.assertTrue(classifier.is_return_track(path))

    def test_identity_homography_maps_pixel_to_table_coordinate(self):
        self.assertEqual(
            map_log_coordinate(np.eye(3, dtype=np.float32), (2.5, 4.0), 0.7786086),
            (2.5, 0.7786, 4.0),
        )

    def test_calibration_can_override_detector_settings(self):
        settings = DetectorSettings.from_calibration({"detector_settings": {"motion_threshold": 9}})

        self.assertEqual(settings.motion_threshold, 9)
        self.assertEqual(settings.max_gap, 5)

    def test_tracker_completes_a_path_after_the_allowed_gap(self):
        tracker = MultiBallTracker(DetectorSettings(max_gap=1))

        self.assertEqual(tracker.update(0, [(10, 20, 0)]), [])
        self.assertEqual(tracker.update(1, []), [])

        self.assertEqual(tracker.update(2, []), [[(0, 10, 20, 0)]])

    def test_cadence_fills_an_unseen_launch_with_one_miss(self):
        classifier = self.classifier()
        events = []
        for frame in (70, 190, 250):
            event = classifier.no_bounce_event(Attempt(frame, (0, 0)), frame)
            event.hit_table = True
            event.outcome = "far_table"
            event.confidence = .8
            events.append(event)

        normalized = normalize_attempt_events(events, total_frames=300, fps=60)

        self.assertEqual([event.outcome for event in normalized], ["hit", "miss", "hit", "hit"])


if __name__ == "__main__":
    unittest.main()

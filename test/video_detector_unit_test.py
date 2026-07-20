"""Focused unit tests for the classical-CV bounce helpers."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_video import (  # noqa: E402
    Attempt,
    AttemptClassifier,
    BounceEvent,
    DetectorDiagnostics,
    DetectorSettings,
    LiveAttemptNormalizer,
    MultiBallTracker,
    TelemetryReading,
    TelemetryReader,
    attach_missing_machine_telemetry,
    candidates_for_frame,
    find_bounce,
    map_log_coordinate,
    normalize_attempt_events,
    read_telemetry,
    shadow_contact_score,
)


class VideoDetectorUnitTest(unittest.TestCase):
    def classifier(self, calibration=None) -> AttemptClassifier:
        calibration = calibration or {
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

    def cadence_event(self, frame, outcome="far_table", confidence=.8):
        event = self.classifier().no_bounce_event(Attempt(frame, (0, 0)), frame)
        event.hit_table = outcome == "far_table"
        event.outcome = outcome
        event.confidence = confidence
        return event

    def test_shadow_score_rises_for_dark_table_patch_below_ball(self):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, :] = (65, 165, 160)  # green table
        hsv[55:70, 42:58, 2] = 105      # ball's dark table shadow

        self.assertGreater(shadow_contact_score(hsv, (50, 50)), 30)

    def test_shadow_score_is_zero_on_evenly_lit_table(self):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, :] = (65, 165, 160)

        self.assertEqual(shadow_contact_score(hsv, (50, 50)), 0.0)

    def test_candidate_diagnostics_separate_raw_and_rejected_blobs(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:23, 20:23] = 255
        frame[70:73, 70:73] = 255
        previous_gray = np.zeros((100, 100), dtype=np.uint8)
        tracking = np.float32([(0, 0), (50, 0), (50, 50), (0, 50)])
        diagnostics = DetectorDiagnostics()
        diagnostics.begin_frame()

        _, candidates = candidates_for_frame(
            frame, previous_gray, tracking, diagnostics=diagnostics,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            [item.kind for item in diagnostics.candidates],
            ["raw", "rejected"],
        )
        self.assertEqual(
            diagnostics.candidates[1].reason, "outside tracking region",
        )

    def candidate_frame(self, shape=(100, 100), color=(255, 255, 255)):
        frame = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        previous_gray = np.zeros(shape, dtype=np.uint8)
        tracking = np.float32([
            (0, 0), (shape[1] - 1, 0),
            (shape[1] - 1, shape[0] - 1), (0, shape[0] - 1),
        ])
        return frame, previous_gray, tracking, color

    def test_candidate_appearance_rejects_single_pixel_codec_shimmer(self):
        frame, previous_gray, tracking, color = self.candidate_frame()
        frame[10, 10] = color
        frame[10:12, 20:21] = color

        _, candidates = candidates_for_frame(frame, previous_gray, tracking)

        self.assertEqual([(round(x), round(y)) for x, y, _ in candidates], [(20, 10)])

    def test_candidate_appearance_rejects_elongated_and_sparse_blobs(self):
        frame, previous_gray, tracking, color = self.candidate_frame()
        frame[10:12, 10:16] = color
        for offset in range(4):
            frame[30 + offset, 30 + offset] = color
        diagnostics = DetectorDiagnostics()
        diagnostics.begin_frame()

        _, candidates = candidates_for_frame(
            frame, previous_gray, tracking, diagnostics=diagnostics,
        )

        self.assertEqual(candidates, [])
        self.assertEqual(
            [item.reason.split()[0] for item in diagnostics.candidates],
            ["aspect", "compactness"],
        )

    def test_candidate_appearance_checks_brightness_and_saturation(self):
        frame, previous_gray, tracking, _ = self.candidate_frame()
        frame[20:23, 20:23] = (200, 200, 200)
        frame[40:43, 40:43] = (0, 255, 255)
        settings = DetectorSettings(
            bright_ball_lower=(0, 0, 150),
            bright_ball_upper=(180, 255, 255),
        )
        diagnostics = DetectorDiagnostics()
        diagnostics.begin_frame()

        _, candidates = candidates_for_frame(
            frame, previous_gray, tracking, settings, diagnostics,
        )

        self.assertEqual(candidates, [])
        self.assertEqual(
            [item.reason.split()[0] for item in diagnostics.candidates],
            ["brightness", "saturation"],
        )

    def test_candidate_area_limit_grows_with_perspective(self):
        frame, previous_gray, tracking, color = self.candidate_frame()
        frame[8:18, 10:20] = color
        frame[78:88, 70:80] = color

        _, candidates = candidates_for_frame(
            frame, previous_gray, tracking,
            DetectorSettings(max_candidate_area=200, far_max_candidate_area_ratio=.1),
        )

        self.assertEqual([(round(x), round(y)) for x, y, _ in candidates], [(74, 82)])

    def test_track_diagnostics_report_classifier_decisions_and_reasons(self):
        reported = []
        classifier = self.classifier()
        classifier.on_track_diagnostic = lambda item, frame: reported.append(
            (item.kind, item.reason, frame)
        )
        short = [(frame, 50 + frame, 100, 0.0) for frame in range(3)]
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [
            (30 + frame, 100 + frame * 20, 100 + abs(4 - frame) * -10, 0.0)
            for frame in range(9)
        ]

        classifier.process_tracks([short, launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)

        self.assertEqual(
            [kind for kind, _, _ in reported],
            ["rejected", "launcher", "return", "confirmed_bounce"],
        )
        self.assertEqual(reported[0][1], "too short (3/9)")

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

    def test_classifier_reports_each_event_to_the_live_callback(self):
        reported = []
        classifier = self.classifier()
        classifier.on_event = reported.append
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.process_tracks([launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)
        classifier.finish_attempt(draw_frame=40)

        self.assertEqual(reported, classifier.events)
        self.assertEqual(len(reported), 1)

    def test_classifier_signals_live_settlement_at_the_next_launch(self):
        settled = []
        classifier = self.classifier()
        classifier.on_attempt_finished = lambda: settled.append(list(classifier.events))
        first_launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]
        next_launch = [(60 + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]

        classifier.process_tracks([first_launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)
        classifier.process_tracks([next_launch], draw_frame=78)

        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0], classifier.events)

    def test_confirmed_hit_is_reported_before_the_attempt_finishes(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        classifier = self.classifier()
        classifier.on_event = normalizer.observe
        classifier.on_confirmed_hit = normalizer.observe_confirmed_hit
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.start_attempt(launch, 18)
        classifier.add_bounce(
            returned, returned[4], returned[1:4], returned[5:8], 39,
        )

        self.assertEqual([event.outcome for event in reported], ["hit"])
        self.assertEqual(classifier.events, [])

        classifier.finish_attempt(40)
        normalizer.settle_attempt()
        self.assertEqual([event.outcome for event in reported], ["hit"])

    def test_active_return_reports_hit_before_track_completion(self):
        reported = []
        classifier = self.classifier()
        classifier.on_confirmed_hit = reported.append
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [
            (30, 20, 70, 0.0),
            (31, 40, 80, 0.0),
            (32, 60, 90, 0.0),
            (33, 80, 100, 0.0),
            (34, 100, 110, 0.0),
            (35, 120, 100, 0.0),
            (36, 140, 90, 0.0),
            (37, 160, 80, 0.0),
            (38, 180, 70, 0.0),
        ]

        classifier.start_attempt(launch, 18)
        classifier.process_active_tracks([returned], draw_frame=38)

        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0].frame_number, 34)
        self.assertEqual(classifier.events, [])

    def test_active_return_waits_to_confirm_terminal_shadow_peak(self):
        reported = []
        classifier = self.classifier()
        classifier.on_confirmed_hit = reported.append
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 20 + frame * 20, 100, 0.0) for frame in range(9)]
        returned[-1] = (*returned[-1][:3], 32.0)

        classifier.start_attempt(launch, 18)
        classifier.process_active_tracks([returned], draw_frame=38)

        self.assertEqual(reported, [])

        returned.append((39, 200, 100, 0.0))
        classifier.process_active_tracks([returned], draw_frame=39)
        self.assertEqual(reported, [])

        returned.append((40, 220, 100, 0.0))
        classifier.process_active_tracks([returned], draw_frame=40)
        self.assertEqual(len(reported), 1)

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

    def test_launcher_track_follows_calibrated_direction_in_mirrored_view(self):
        classifier = self.classifier({
            "table_surface_y": 0.7786086,
            "launcher_region": [0, 0, 420, 300],
            "return_region": [720, 0, 1000, 300],
        })
        launch = [(frame, 200 + frame * 10, 100, 0.0) for frame in range(18)]
        wrong_way = [(frame, 400 - frame * 10, 100, 0.0) for frame in range(18)]

        self.assertTrue(classifier.is_launcher_track(launch))
        self.assertFalse(classifier.is_launcher_track(wrong_way))
        self.assertEqual(
            classifier.launcher_rejection_reason(wrong_way),
            "insufficient travel toward player",
        )

    def test_launcher_track_requires_sustained_progress_toward_player(self):
        classifier = self.classifier()
        shimmer = [(frame, 800 + (frame % 2) * 2, 100, 0.0) for frame in range(18)]
        unrelated_motion = [
            (frame, x, 100, 0.0)
            for frame, x in enumerate(
                [800, 760, 720, 680, 640, 600, 560, 600, 640,
                 680, 640, 600, 560, 520, 560, 520, 500, 480]
            )
        ]

        self.assertFalse(classifier.is_launcher_track(shimmer))
        self.assertEqual(
            classifier.launcher_rejection_reason(shimmer),
            "insufficient travel toward player",
        )
        self.assertFalse(classifier.is_launcher_track(unrelated_motion))
        self.assertEqual(
            classifier.launcher_rejection_reason(unrelated_motion),
            "inconsistent travel toward player",
        )

    def test_launcher_validation_accepts_supported_speed_and_spin_shapes(self):
        classifier = self.classifier()
        cases = {
            "slow topspin": [
                (frame, 850 - frame * 5, 80 + frame * .4, 0.0)
                for frame in range(30)
            ],
            "fast backspin": [
                (frame, 850 - frame * 20, 120 - frame * 1.5, 0.0)
                for frame in range(18)
            ],
            "sidespin arc": [
                (frame, 850 - frame * 10, 100 + ((frame - 9) ** 2) * .25, 0.0)
                for frame in range(18)
            ],
        }

        for label, path in cases.items():
            with self.subTest(label):
                self.assertTrue(classifier.is_launcher_track(path))

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
            (7, 500, 140, 0.0),
        ]

        returned = classifier.return_segment(path)

        self.assertIsNotNone(returned)
        self.assertEqual(returned[0][0], 3)
        self.assertTrue(classifier.is_return_track(path))

    def test_return_follows_calibrated_direction_in_mirrored_view(self):
        classifier = self.classifier({
            "table_surface_y": 0.7786086,
            "launcher_region": [0, 0, 420, 300],
            "return_region": [720, 0, 1000, 300],
        })
        returned = [(30 + frame, 900 - frame * 20, 100, 0.0) for frame in range(9)]
        wrong_way = [(30 + frame, 750 + frame * 20, 100, 0.0) for frame in range(9)]

        self.assertTrue(classifier.is_return_track(returned))
        self.assertFalse(classifier.is_return_track(wrong_way))
        self.assertEqual(
            classifier.return_rejection_reason(wrong_way),
            "insufficient travel toward opponent",
        )

    def test_return_requires_observations_after_its_active_launch(self):
        classifier = self.classifier()
        launch = [
            (30 + frame, 800 - frame * 10, 100, 0.0)
            for frame in range(18)
        ]
        classifier.process_tracks([launch], draw_frame=48)
        assert classifier.active_attempt is not None
        ended_before_launch = [
            (frame, 100 + frame * 20, 100, 0.0) for frame in range(9)
        ]
        stale_prefix_then_return = [
            (20, 100, 100, 0.0),
            (21, 105, 100, 0.0),
            (31, 120, 100, 0.0),
            (32, 160, 100, 0.0),
            (33, 200, 100, 0.0),
            (34, 240, 100, 0.0),
        ]

        self.assertEqual(
            classifier.return_rejection_reason(
                ended_before_launch, classifier.active_attempt,
            ),
            "too few return observations after launch (0/3)",
        )
        self.assertIsNone(
            classifier.return_rejection_reason(
                stale_prefix_then_return, classifier.active_attempt,
            )
        )

        classifier.process_tracks(
            [ended_before_launch], draw_frame=50,
        )
        self.assertEqual(classifier.active_attempt.returns, [])

    def test_partially_occluded_return_needs_several_post_launch_observations(self):
        classifier = self.classifier()
        attempt = Attempt(10, (800, 100))
        mostly_pre_launch = [
            (5, 100, 120, 0.0),
            (6, 130, 125, 0.0),
            (7, 160, 130, 0.0),
            (8, 190, 135, 0.0),
            (9, 220, 140, 0.0),
            (10, 250, 145, 0.0),
            (11, 280, 150, 0.0),
            (12, 310, 155, 0.0),
        ]

        self.assertFalse(classifier.is_return_track(mostly_pre_launch, attempt))
        self.assertEqual(
            classifier.return_rejection_reason(mostly_pre_launch, attempt),
            "too few return observations after launch (2/3)",
        )

    def test_identity_homography_maps_pixel_to_table_coordinate(self):
        self.assertEqual(
            map_log_coordinate(np.eye(3, dtype=np.float32), (2.5, 4.0), 0.7786086),
            (2.5, 0.7786, 4.0),
        )

    def test_calibration_can_override_detector_settings(self):
        settings = DetectorSettings.from_calibration({"detector_settings": {"motion_threshold": 9}})

        self.assertEqual(settings.motion_threshold, 9)
        self.assertEqual(settings.max_gap, 5)

    def test_high_resolution_tv_telemetry_is_read(self):
        cap = cv2.VideoCapture(str(ROOT / "sample3-trimmed-44s.mp4"))
        cap.set(cv2.CAP_PROP_POS_MSEC, 100)
        ok, frame = cap.read()
        cap.release()

        self.assertTrue(ok)
        reading = read_telemetry(frame, 6)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.speed_mps, 11.4)
        self.assertEqual(reading.spin_revolutions_per_second, 64)
        self.assertEqual(reading.spin_direction["label"], "up")

    def test_low_resolution_tv_telemetry_is_read(self):
        cap = cv2.VideoCapture(str(ROOT / "sample2-trimmed-58s.mp4"))
        cap.set(cv2.CAP_PROP_POS_MSEC, 5_000)
        ok, frame = cap.read()
        cap.release()

        self.assertTrue(ok)
        reading = read_telemetry(frame, 300)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.speed_mps, 9.6)
        self.assertEqual(reading.spin_revolutions_per_second, 77)
        self.assertEqual(reading.spin_direction["label"], "up-right")

    def test_low_resolution_three_digit_spin_is_read(self):
        cap = cv2.VideoCapture(str(ROOT / "sample2-trimmed-58s.mp4"))
        cap.set(cv2.CAP_PROP_POS_MSEC, 3_400)
        ok, frame = cap.read()
        cap.release()

        self.assertTrue(ok)
        reading = read_telemetry(frame, 204)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.speed_mps, 10.4)
        self.assertEqual(reading.spin_revolutions_per_second, 109)
        self.assertEqual(reading.spin_direction["label"], "up")

    def test_implausible_low_resolution_spin_ocr_is_rejected(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        direction = {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"}

        with (
            patch("analyze_video.read_hud_number", side_effect=[19.5, 517]),
            patch("analyze_video.read_spin_direction", return_value=direction),
        ):
            reading = read_telemetry(frame, 1, (0, 0, 1, 1))

        self.assertIsNone(reading)

    def test_low_resolution_telemetry_jitter_is_one_state(self):
        original = TelemetryReading(
            1, 10.5, 51,
            {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"},
        )
        jittered = TelemetryReading(
            2, 10.6, 50,
            {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"},
        )

        self.assertTrue(TelemetryReader.same_values(original, jittered))

    def test_return_out_uses_telemetry_before_event_as_player_hit(self):
        direction = {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"}
        machine = TelemetryReading(180, 10.6, 50, direction)
        player = TelemetryReading(201, 10.4, 109, direction)
        next_machine = TelemetryReading(246, 10.5, 51, direction)
        event = BounceEvent(
            4.067, "00:04.067", False, False, "out",
            None, None, None, .58, 244, (0, 0), 244,
            return_crossed_net=True,
            machine=next_machine.to_record(60),
        )

        attached = attach_missing_machine_telemetry(
            [event], [machine, player, next_machine], 60,
        )[0]

        self.assertEqual(attached.hit["speed_mps"], 10.4)
        self.assertEqual(attached.hit["spin_revolutions_per_second"], 109)
        self.assertEqual(attached.machine["speed_mps"], 10.6)

    def test_hit_and_machine_telemetry_attach_to_the_landing(self):
        classifier = self.classifier()
        machine = TelemetryReading(2, 10.5, 51, {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"})
        returned = TelemetryReading(25, 15.0, 80, {"x": -.7, "y": .7, "angle_degrees": 135, "label": "up-left"})
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        path = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.observe_telemetry(machine)
        classifier.start_attempt(launch, 18)
        classifier.observe_telemetry(returned)
        classifier.add_bounce(path, path[4], path[1:4], path[5:8], 39)
        classifier.finish_attempt(40)

        record = classifier.events[0].to_record()
        self.assertEqual(record["machine"]["speed_mps"], 10.5)
        self.assertEqual(record["hit"]["speed_mps"], 15.0)
        self.assertIsNotNone(record["posx"])

    def test_tracker_completes_a_path_after_the_allowed_gap(self):
        tracker = MultiBallTracker(DetectorSettings(max_gap=1, min_track_observations=1))

        self.assertEqual(tracker.update(0, [(10, 20, 0)]), [])
        self.assertEqual(tracker.update(1, []), [])

        self.assertEqual(tracker.update(2, []), [[(0, 10, 20, 0)]])

    def test_tracker_requires_consistent_observations_before_completion(self):
        tracker = MultiBallTracker(DetectorSettings(max_gap=0, min_track_observations=3))

        tracker.update(0, [(10, 20, 0)])
        tracker.update(1, [(20, 20, 0)])

        self.assertEqual(tracker.update(2, []), [])

        tracker.update(3, [(10, 20, 0)])
        tracker.update(4, [(20, 20, 0)])
        tracker.update(5, [(30, 20, 0)])
        self.assertEqual(
            tracker.update(6, []),
            [[(3, 10, 20, 0), (4, 20, 20, 0), (5, 30, 20, 0)]],
        )

    def test_tracker_allows_prediction_uncertainty_across_a_gap(self):
        tracker = MultiBallTracker(DetectorSettings(max_gap=2))
        tracker.update(0, [(10, 20, 0)])
        tracker.update(1, [(20, 20, 0)])
        tracker.update(2, [])
        tracker.update(3, [(40, 20, 0), (30, 20, 0)])

        self.assertEqual(
            tracker.tracks[0].points[-1],
            (3, 30, 20, 0),
        )

    def test_tracker_rejects_implausible_jump_acceleration_and_reversal(self):
        settings = DetectorSettings(
            min_track_observations=3,
            max_track_speed=50,
            max_track_acceleration=12,
            max_direction_change_degrees=100,
        )
        cases = {
            "too fast": (80, 20, 0),
            "excessive acceleration": (45, 20, 0),
            "direction reversal": (10, 20, 0),
        }
        for label, candidate in cases.items():
            with self.subTest(label):
                tracker = MultiBallTracker(settings)
                tracker.update(0, [(10, 20, 0)])
                tracker.update(1, [(20, 20, 0)])
                tracker.update(2, [candidate])

                self.assertEqual(len(tracker.tracks[0].points), 2)
                self.assertFalse(tracker.tracks[0].confirmed)

    def test_cadence_fills_an_unseen_launch_with_one_miss(self):
        events = [self.cadence_event(frame) for frame in (70, 190, 250)]

        normalized = normalize_attempt_events(events, total_frames=300, fps=60)

        self.assertEqual([event.outcome for event in normalized], ["hit", "miss", "hit", "hit"])

    def test_live_normalizer_emits_settled_hits_after_cadence_warmup(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            normalizer.observe(self.cadence_event(frame))
            normalizer.settle_attempt()

        self.assertEqual([event.outcome for event in reported], ["hit", "hit", "hit"])

    def test_live_normalizer_emits_a_settled_out(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            normalizer.observe(self.cadence_event(frame))
            normalizer.settle_attempt()
        normalizer.observe(self.cadence_event(250, "off_table", .58))
        normalizer.settle_attempt()

        self.assertEqual([event.outcome for event in reported], ["hit", "hit", "hit", "out"])

    def test_live_normalizer_emits_a_miss_when_next_launch_closes_attempt(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        normalizer.observe(self.cadence_event(70, "unknown", .2))

        self.assertEqual(reported, [])
        normalizer.settle_attempt()

        self.assertEqual([event.outcome for event in reported], ["miss"])

    def test_live_normalizer_does_not_miss_an_attempt_with_a_confirmed_hit(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        hit = self.cadence_event(80)
        normalizer.observe_confirmed_hit(hit)
        normalizer.observe(self.cadence_event(70, "near_table", .5))
        normalizer.observe(hit)
        normalizer.settle_attempt()

        self.assertEqual([event.outcome for event in reported], ["hit"])

    def test_immediate_non_hit_is_not_repeated_when_cadence_warms(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for event in (
            self.cadence_event(76),
            self.cadence_event(143),
            self.cadence_event(136, "off_table", .58),
            self.cadence_event(244, "off_table", .58),
            self.cadence_event(289, "unknown", .2),
            self.cadence_event(368),
        ):
            if event.outcome == "far_table":
                normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)
            normalizer.settle_attempt()

        self.assertEqual(
            [event.frame_number for event in reported].count(136), 1,
        )

    def test_live_normalizer_infers_only_settled_misses(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 190):
            normalizer.observe(self.cadence_event(frame))
            normalizer.settle_attempt()
        self.assertEqual(reported, [])

        normalizer.observe(self.cadence_event(250))
        normalizer.settle_attempt()
        self.assertEqual(
            [event.outcome for event in reported],
            ["hit", "miss", "hit", "hit"],
        )

    def test_live_normalizer_never_emits_a_slot_twice(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            normalizer.observe(self.cadence_event(frame))
            normalizer.settle_attempt()
        reported_frames = [event.frame_number for event in reported]
        normalizer.observe(self.cadence_event(190, confidence=.9))
        normalizer.settle_attempt()

        self.assertEqual(reported_frames.count(190), 1)
        self.assertEqual(
            [event.frame_number for event in reported].count(190), 1,
        )

    def test_immediate_hits_are_not_repeated_after_cadence_warms(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 190, 250):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)
            normalizer.settle_attempt()

        self.assertEqual(
            [event.frame_number for event in reported if event.outcome == "hit"],
            [70, 190, 250],
        )
        self.assertEqual(
            [event.outcome for event in reported].count("miss"), 1,
        )

    def test_live_normalizer_final_output_matches_batch_normalization(self):
        events = [self.cadence_event(frame) for frame in (70, 190, 250)]
        events.append(self.cadence_event(310, "off_table", .58))
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for event in events:
            normalizer.observe(event)
            normalizer.settle_attempt()

        finalized = normalizer.finalize(380)
        self.assertEqual(finalized, normalize_attempt_events(events, 380, 60))
        self.assertEqual(reported, finalized[:len(reported)])


if __name__ == "__main__":
    unittest.main()

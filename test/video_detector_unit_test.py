"""Focused unit tests for the classical-CV bounce helpers."""
import sys
import tempfile
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
    ContactCandidate,
    DirectLiveAttemptPublisher,
    LiveCounterHealthMonitor,
    LivePipelineLogger,
    bounce_signal,
    DetectorDiagnostics,
    DetectorSettings,
    LiveAttemptNormalizer,
    MultiBallTracker,
    TelemetryReading,
    TelemetryReader,
    attach_missing_machine_telemetry,
    candidates_for_frame,
    classify_digit,
    find_bounce,
    find_bounces,
    map_log_coordinate,
    normalize_attempt_events,
    read_telemetry,
    reset_output_file,
    shadow_contact_score,
    split_wide_component,
    telemetry_title_bounds,
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

    def test_reset_output_file_clears_previous_session(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "video_bounces.jsonl"
            output.parent.mkdir()
            output.write_text('{"stale": true}\n', encoding="utf-8")

            reset_output_file(output)

            self.assertEqual(output.read_text(encoding="utf-8"), "")

    def test_live_pipeline_logger_emits_bounded_progress_and_end_records(self):
        now = [100.0]
        wall = [1_700_000_000.0]
        records = []
        logger = LivePipelineLogger(
            60,
            records.append,
            interval_seconds=1,
            monotonic=lambda: now[0],
            wall_time=lambda: wall[0],
        )

        logger.observe(120, {"candidate_count": 3})
        now[0] += 0.5
        wall[0] += 0.5
        logger.observe(150, {"candidate_count": 4})
        now[0] += 0.5
        wall[0] += 0.5
        logger.observe(180, {"candidate_count": 5})
        logger.finish(180, "processing_ended")

        self.assertEqual(
            [record["type"] for record in records],
            ["pipeline_heartbeat", "pipeline_heartbeat", "pipeline_end"],
        )
        self.assertEqual(records[0]["frame_number"], 120)
        self.assertEqual(records[1]["frame_number"], 180)
        self.assertEqual(records[1]["processing_fps"], 60)
        self.assertEqual(records[1]["estimated_lag_seconds"], 0)
        self.assertEqual(records[1]["candidate_count"], 5)
        self.assertEqual(records[2]["reason"], "processing_ended")

    def test_live_health_warns_when_detector_events_are_not_published(self):
        records = []
        monitor = LiveCounterHealthMonitor(records.append)

        monitor.observe_pipeline({
            "detected_event_count": 3,
            "attempt_upsert_count": 0,
        })
        monitor.observe_pipeline({
            "detected_event_count": 4,
            "attempt_upsert_count": 0,
        })
        monitor.observe_pipeline({
            "detected_event_count": 4,
            "attempt_upsert_count": 1,
        })

        self.assertEqual(
            [(item["status"], item["code"]) for item in records],
            [
                ("warning", "publisher_stalled"),
                ("recovered", "publisher_stalled"),
            ],
        )

    def test_live_health_warns_after_repeated_attempts_without_a_hit(self):
        records = []
        monitor = LiveCounterHealthMonitor(
            records.append, no_hit_attempt_threshold=3,
        )
        for sequence in range(1, 4):
            monitor.observe_attempt({
                "attempt_id": f"launch-{sequence}",
                "state": "finalized",
                "outcome": "miss",
            })
        monitor.observe_attempt({
            "attempt_id": "launch-4",
            "state": "finalized",
            "outcome": "hit",
        })

        self.assertEqual(
            [(item["status"], item["code"]) for item in records],
            [
                ("warning", "no_confirmed_contacts"),
                ("recovered", "no_confirmed_contacts"),
            ],
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

    def test_bounce_signal_describes_observation_shape(self):
        hit = (4, 180, 120, 0.0)
        maximum_approach = [(1, 120, 90, 0.0), (2, 140, 100, 0.0), (3, 160, 110, 0.0)]
        maximum_departure = [(5, 200, 105, 0.0), (6, 220, 95, 0.0)]
        minimum_approach = [(1, 120, 150, 0.0), (2, 140, 140, 0.0), (3, 160, 130, 0.0)]
        minimum_departure = [(5, 200, 135, 0.0), (6, 220, 145, 0.0)]

        self.assertEqual(
            bounce_signal(hit, maximum_approach, maximum_departure),
            "vertical_maximum",
        )
        self.assertEqual(
            bounce_signal(hit, minimum_approach, minimum_departure),
            "vertical_minimum",
        )
        self.assertEqual(
            bounce_signal(hit, maximum_approach[-2:], maximum_departure),
            "shadow",
        )
        self.assertEqual(
            bounce_signal(hit, maximum_approach[-2:], []),
            "terminal",
        )

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
            [
                "rejected", "launcher", "association", "return",
                "contact_candidate", "confirmed_bounce",
            ],
        )
        self.assertEqual(reported[0][1], "too short (3/9)")

    def test_shadow_contact_is_a_bounce_away_from_net(self):
        points = [(frame, 200 + frame * 4, 220, 0.0) for frame in range(9)]
        points[4] = (4, 216, 220, 32.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])

        hit, _, _ = find_bounce(points, table, net_line=np.float32([(0, 0), (0, 500)]))

        self.assertEqual(hit[0], 4)

    def test_find_bounces_returns_all_contacts_in_chronological_order(self):
        points = [(frame, 100 + frame * 20, 220, 0.0) for frame in range(14)]
        points[4] = (*points[4][:3], 32.0)
        points[10] = (*points[10][:3], 38.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])
        net = np.float32([(0, 0), (0, 500)])

        contacts = find_bounces(points, table, net)

        self.assertEqual([contact[0][0] for contact in contacts], [4, 10])
        self.assertEqual(find_bounce(points, table, net)[0][0], 4)

    def test_attempt_keeps_ordered_contacts_and_deduplicates_track_handoffs(self):
        classifier = self.classifier()
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        later = [
            (36 + frame, 100 + frame * 20, 100 - abs(4 - frame) * 10, 0.0)
            for frame in range(9)
        ]
        earlier = [
            (30 + frame, 100 + frame * 20, 90 - abs(4 - frame) * 10, 0.0)
            for frame in range(9)
        ]
        handoff = [(32 + frame, 140 + frame * 10, 90, 0.0) for frame in range(7)]
        classifier.start_attempt(launch, 18)

        classifier.add_bounce(later, later[4], later[1:4], later[5:7], 50)
        classifier.add_bounce(earlier, earlier[4], earlier[1:4], earlier[5:7], 50)
        classifier.add_bounce(handoff, earlier[4], earlier[1:4], earlier[5:7], 50)

        attempt = classifier.active_attempt
        self.assertIsNotNone(attempt)
        self.assertEqual(
            [candidate.frame_number for candidate in attempt.contact_candidates],
            [34, 40],
        )
        self.assertEqual(len(attempt.bounces), 2)
        record = attempt.contact_candidates[0].to_record()
        self.assertEqual(record["table_side"], "opponent")
        self.assertEqual(record["signal_type"], "vertical_maximum")
        self.assertEqual(record["source_track_key"], [30, 100, 50])

    def ordered_contact_candidate(
        self, frame, z, side, signal="vertical_minimum", source=(30, 100, 50),
    ):
        y = z + 100
        approach = (
            (frame - 3, 100, y + 6, 0.0),
            (frame - 2, 120, y + 4, 0.0),
            (frame - 1, 140, y + 2, 0.0),
        )
        departure = (
            (frame + 1, 180, y + 2, 0.0),
            (frame + 2, 200, y + 4, 0.0),
        )
        return ContactCandidate(
            frame, (160, y), (0.0, 0.7786, z), side, signal,
            3.0, 0.7, source, approach, departure,
        )

    def test_player_contact_then_shallower_opponent_contact_is_a_miss(self):
        reported = []
        classifier = self.classifier()
        classifier.on_confirmed_hit = reported.append
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        path = [(30 + frame, 100 + frame * 20, 150, 0.0) for frame in range(13)]
        classifier.start_attempt(launch, 18)
        attempt = classifier.active_attempt
        source = (30, 100, 150)
        near = self.ordered_contact_candidate(
            34, -80, "player", source=source,
        )
        far = self.ordered_contact_candidate(
            39, 50, "opponent", source=source,
        )
        classifier.record_contact_candidate(near, path, 42)
        classifier.record_contact_candidate(far, path, 42)

        classifier.add_bounce(path, path[10], path[7:10], path[11:13], 42)
        classifier.finish_attempt(43)

        self.assertEqual(reported, [])
        self.assertEqual([event.outcome for event in classifier.events], ["near_table"])
        self.assertEqual(classifier.events[0].frame_number, 34)
        self.assertIsNotNone(attempt)

    def test_weak_in_flight_turn_before_far_contact_remains_a_hit(self):
        classifier = self.classifier()
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        path = [(30 + frame, 100 + frame * 20, 150, 0.0) for frame in range(13)]
        classifier.start_attempt(launch, 18)
        near = self.ordered_contact_candidate(34, -20, "player")
        far = self.ordered_contact_candidate(40, 70, "opponent")
        classifier.record_contact_candidate(near, path, 42)
        classifier.record_contact_candidate(far, path, 42)

        classifier.add_bounce(path, path[10], path[7:10], path[11:13], 42)
        classifier.finish_attempt(43)

        self.assertEqual([event.outcome for event in classifier.events], ["far_table"])

    def test_ordered_miss_does_not_suppress_an_unrelated_far_contact(self):
        near = self.ordered_contact_candidate(
            34, -80, "player", source=(30, 100, 50),
        )
        far = self.ordered_contact_candidate(
            39, 50, "opponent", source=(30, 100, 50),
        )
        unrelated = self.ordered_contact_candidate(
            45, 70, "opponent", source=(43, 200, 80),
        )

        self.assertTrue(
            AttemptClassifier.contact_belongs_to_ordered_miss(
                far, (near, far),
            )
        )
        self.assertFalse(
            AttemptClassifier.contact_belongs_to_ordered_miss(
                unrelated, (near, far),
            )
        )

    def test_velocity_flattening_cannot_establish_player_contact(self):
        attempt = Attempt(0, (0, 0))
        attempt.contact_candidates = [
            self.ordered_contact_candidate(
                34, -80, "player", signal="velocity_flattening",
            ),
            self.ordered_contact_candidate(40, 50, "opponent"),
        ]

        self.assertIsNone(
            AttemptClassifier.confirmed_player_then_opponent(attempt)
        )

    def test_contacts_beyond_bounded_evidence_horizon_remain_a_hit(self):
        classifier = self.classifier()
        attempt = Attempt(0, (0, 0))
        attempt.contact_candidates = [
            self.ordered_contact_candidate(34, -80, "player"),
            self.ordered_contact_candidate(40, 50, "opponent"),
        ]

        self.assertIsNone(
            AttemptClassifier.confirmed_player_then_opponent(attempt)
        )
        classifier.record_contact_history_rejections(attempt, 42)
        self.assertEqual(len(attempt.rejected_contact_candidates), 1)
        rejected = attempt.rejected_contact_candidates[0]
        self.assertFalse(rejected.accepted)
        self.assertEqual(
            rejected.rejection_reason,
            "opponent contact is outside bounded approach/departure evidence",
        )

    def test_net_contact_then_opponent_contact_is_a_hit_history(self):
        attempt = Attempt(0, (0, 0))
        net = self.ordered_contact_candidate(
            34, 0, "net", signal="net",
        )
        far = self.ordered_contact_candidate(40, 70, "opponent")
        attempt.contact_candidates = [net, far]

        self.assertEqual(
            AttemptClassifier.confirmed_net_then_opponent(attempt),
            (net, far),
        )
        self.assertIsNone(
            AttemptClassifier.confirmed_player_then_opponent(attempt)
        )

    def test_net_only_contact_remains_a_miss_when_attempt_settles(self):
        classifier = self.classifier()
        classifier.net_line = np.float32([(150, 0), (150, 500)])
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 20 + frame * 15, 100, 0.0) for frame in range(9)]
        classifier.start_attempt(launch, 18)

        classifier.process_tracks([returned], draw_frame=39)
        attempt = classifier.active_attempt
        classifier.finish_attempt(40)

        self.assertEqual([event.outcome for event in classifier.events], ["net"])
        self.assertEqual(
            [candidate.signal_type for candidate in attempt.contact_candidates],
            ["net"],
        )

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

    def test_post_contact_tracker_reversal_is_not_a_trajectory_bounce(self):
        points = [
            (frame, 100 + frame * 20, 100 + abs(4 - frame) * -10, 0.0)
            for frame in range(9)
        ]
        points[5] = (5, 175, 90, 0.0)
        points[6] = (6, 172, 80, 0.0)
        table = np.float32([(0, 0), (500, 0), (500, 500), (0, 500)])

        self.assertIsNone(
            find_bounce(points, table, net_line=np.float32([(0, 0), (0, 500)]))
        )

    def test_pre_contact_tracker_handoff_is_not_a_trajectory_bounce(self):
        points = [
            (0, 100, 80, 0.0),
            (1, 120, 85, 0.0),
            (2, 118, 90, 0.0),
            (3, 116, 95, 0.0),
            (4, 140, 110, 0.0),
            (5, 160, 95, 0.0),
            (6, 180, 90, 0.0),
            (7, 200, 85, 0.0),
            (8, 220, 80, 0.0),
        ]
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
        classifier.on_attempt_finished = (
            lambda _frame: settled.append(list(classifier.events))
        )
        first_launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]
        next_launch = [(60 + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]

        classifier.process_tracks([first_launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)
        classifier.process_tracks([next_launch], draw_frame=78)

        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0], classifier.events)

    def test_active_launcher_closes_previous_attempt_before_track_completion(self):
        settled = []
        classifier = self.classifier()
        classifier.on_attempt_finished = (
            lambda _frame: settled.append(list(classifier.events))
        )
        first_launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        active_next_launch = [
            (60 + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)
        ]

        classifier.process_tracks([first_launch], draw_frame=18)
        classifier.process_active_tracks([active_next_launch], draw_frame=77)

        self.assertEqual(len(settled), 1)
        self.assertEqual([event.outcome for event in settled[0]], ["unknown"])

        classifier.process_tracks([active_next_launch], draw_frame=82)
        self.assertEqual(len(settled), 1)

    def test_classifier_signals_when_the_first_machine_launch_is_detected(self):
        started = []
        classifier = self.classifier()
        classifier.on_attempt_started = started.append
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]

        classifier.process_tracks([launch], draw_frame=18)

        self.assertEqual(started, [0])

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

        # The first three direct hits establish cadence before stable attempt
        # IDs can be published.
        self.assertEqual(reported, [])
        self.assertEqual(classifier.events, [])

        classifier.finish_attempt(40)
        normalizer.settle_attempt()
        self.assertEqual(reported, [])

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

    def test_completed_visible_out_is_reported_before_next_launch(self):
        reported = []
        classifier = self.classifier()
        classifier.net_line = np.float32([(150, 0), (150, 500)])
        classifier.on_confirmed_non_hit = reported.append
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        returned = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.process_tracks([launch], draw_frame=18)
        classifier.process_tracks([returned], draw_frame=39)

        self.assertEqual([event.outcome for event in reported], ["off_table"])
        self.assertEqual(classifier.events, [])

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

    def test_elevated_view_splits_continuous_delivery_and_return_track(self):
        classifier = self.classifier({
            "table_surface_y": 0.7786086,
            "camera_geometry": "elevated_end_view",
            "launcher_region": [580, 0, 950, 300],
            "return_region": [0, 0, 500, 500],
        })
        delivery = [
            (frame, 800 - frame * 20, 100, 0.0)
            for frame in range(18)
        ]
        outbound = [
            (18, 500, 100, 0.0),
            (19, 550, 100, 0.0),
            (20, 620, 100, 0.0),
        ]

        returned = classifier.perspective_round_trip_return(
            delivery + outbound,
        )

        self.assertIsNotNone(returned)
        self.assertEqual(returned[0][0], 17)
        self.assertEqual(returned[-1][0], 20)

    def test_side_view_does_not_split_continuous_round_trip_track(self):
        classifier = self.classifier()
        path = [
            (frame, 800 - frame * 20, 100, 0.0)
            for frame in range(18)
        ] + [
            (18, 500, 100, 0.0),
            (19, 550, 100, 0.0),
            (20, 620, 100, 0.0),
        ]

        self.assertIsNone(classifier.perspective_round_trip_return(path))

    def test_profile_view_classifies_turn_side_without_fabricating_position(self):
        classifier = self.classifier({
            "table_surface_y": 0.7786086,
            "camera_geometry": "profile_side_view",
            "launcher_region": [580, 0, 950, 300],
            "control_points": [
                {"name": "x0_player_edge", "image": [100, 100], "log": [0, -1.37]},
                {"name": "x0_opponent_edge", "image": [900, 100], "log": [0, 1.37]},
            ],
        })
        hit = (4, 600, 120, 0.0)
        approach = [
            (1, 540, 90, 0.0), (2, 560, 100, 0.0), (3, 580, 110, 0.0),
        ]
        departure = [(5, 620, 105, 0.0), (6, 640, 95, 0.0)]

        candidate = classifier.contact_candidate(
            approach + [hit] + departure, hit, approach, departure,
        )

        self.assertEqual(candidate.table_side, "opponent")
        self.assertIsNone(candidate.log_position)
        self.assertEqual(candidate.signal_type, "vertical_maximum")

    def test_profile_view_ignores_a_non_upward_turn(self):
        classifier = self.classifier({
            "table_surface_y": 0.7786086,
            "camera_geometry": "profile_side_view",
            "launcher_region": [580, 0, 950, 300],
            "control_points": [
                {"name": "x0_player_edge", "image": [100, 100], "log": [0, -1.37]},
                {"name": "x0_opponent_edge", "image": [900, 100], "log": [0, 1.37]},
            ],
        })
        attempt = Attempt(0, (800, 100))
        classifier.active_attempts.append(attempt)
        hit = (4, 600, 90, 0.0)
        approach = [
            (1, 540, 120, 0.0), (2, 560, 110, 0.0), (3, 580, 100, 0.0),
        ]
        departure = [(5, 620, 105, 0.0), (6, 640, 115, 0.0)]
        path = approach + [hit] + departure

        classifier.add_bounce(path, hit, approach, departure, draw_frame=6)

        self.assertEqual(attempt.bounces, [])

    def test_slow_rolling_ball_is_not_a_return(self):
        classifier = self.classifier()
        rolling = [
            (30 + frame, 100 + frame * 2, 100, 0.0)
            for frame in range(70)
        ]

        self.assertFalse(classifier.is_return_track(rolling))
        self.assertEqual(
            classifier.return_rejection_reason(rolling),
            "return moved too slowly",
        )

    def test_return_fragments_reconnect_across_a_short_occlusion(self):
        classifier = self.classifier()
        attempt = Attempt(10, (800, 100))
        first = [
            (30 + frame, 100 + frame * 20, 100, 0.0)
            for frame in range(8)
        ]
        continuation = [
            (45 + frame, 260 + frame * 20, 100, 0.0)
            for frame in range(8)
        ]
        attempt.returns.append(first)

        reconnected = classifier.reconnected_return(continuation, attempt)

        self.assertIsNotNone(reconnected)
        self.assertEqual(reconnected, first + continuation)

    def profile_classifier(self) -> AttemptClassifier:
        classifier = self.classifier({
            "table_surface_y": 0.7786086,
            "camera_geometry": "profile_side_view",
            "launcher_region": [700, 0, 950, 500],
            "return_region": [0, 0, 300, 500],
            "control_points": [
                {"name": "x0_player_edge", "image": [100, 350], "log": [0, -1.37]},
                {"name": "x0_opponent_edge", "image": [850, 350], "log": [0, 1.37]},
            ],
        })
        classifier.table = np.float32([
            (400, 330), (850, 330), (850, 360), (400, 360),
        ])
        classifier.net_line = np.float32([(400, 300), (400, 400)])
        return classifier

    def test_profile_return_stitches_overlapping_contact_fragments(self):
        classifier = self.profile_classifier()
        attempt = Attempt(10, (800, 100))
        descent = [
            (30, 100, 280, 0.0),
            (31, 180, 290, 0.0),
            (32, 260, 300, 0.0),
            (33, 340, 315, 0.0),
            (34, 420, 310, 0.0),
            (35, 500, 320, 0.0),
            (36, 580, 350, 0.0),
        ]
        rising = [
            (34, 421, 311, 0.0),
            (35, 501, 321, 0.0),
            (37, 660, 355, 0.0),
            (38, 740, 340, 0.0),
            (39, 820, 325, 0.0),
        ]
        older_fragment = [
            (15, 100, 290, 0.0),
            (16, 150, 295, 0.0),
            (17, 200, 300, 0.0),
            (18, 250, 305, 0.0),
            (19, 300, 310, 0.0),
            (20, 350, 315, 0.0),
        ]
        attempt.returns.extend([older_fragment, descent])

        stitched = classifier.reconnected_return(rising, attempt)

        self.assertIsNotNone(stitched)
        self.assertEqual(
            [point[0] for point in stitched],
            [30, 31, 32, 33, 34, 35, 37, 38, 39],
        )
        bounce = find_bounce(
            stitched, classifier.table, classifier.net_line,
            classifier.settings,
        )
        self.assertIsNotNone(bounce)
        self.assertEqual(bounce_signal(*bounce), "vertical_maximum")

    def test_profile_return_stitches_same_frame_contact_fragments(self):
        classifier = self.profile_classifier()
        attempt = Attempt(10, (800, 100))
        descent = [
            (30, 100, 280, 0.0),
            (31, 200, 295, 0.0),
            (32, 300, 310, 0.0),
            (33, 400, 330, 0.0),
            (34, 500, 350, 0.0),
        ]
        rising = [
            (34, 502, 351, 0.0),
            (35, 600, 355, 0.0),
            (36, 700, 340, 0.0),
            (37, 800, 325, 0.0),
        ]
        attempt.returns.append(descent)

        stitched = classifier.reconnected_return(rising, attempt)

        self.assertIsNotNone(stitched)
        self.assertEqual(
            [point[0] for point in stitched],
            [30, 31, 32, 33, 34, 35, 36, 37],
        )

    def test_profile_return_rejects_spatially_unrelated_overlap(self):
        classifier = self.profile_classifier()
        attempt = Attempt(10, (800, 100))
        descent = [
            (30 + frame, 100 + frame * 80, 300 + frame * 8, 0.0)
            for frame in range(6)
        ]
        unrelated = [
            (34 + frame, 700 + frame * 30, 340 - frame * 5, 0.0)
            for frame in range(5)
        ]
        attempt.returns.append(descent)

        self.assertIsNone(
            classifier.reconnected_return(unrelated, attempt),
        )

    def test_profile_return_rejects_overlapping_backward_motion(self):
        classifier = self.profile_classifier()
        attempt = Attempt(10, (800, 100))
        descent = [
            (30 + frame, 100 + frame * 80, 300 + frame * 8, 0.0)
            for frame in range(6)
        ]
        backward = [
            (34, 421, 333, 0.0),
            (35, 501, 341, 0.0),
            (36, 450, 345, 0.0),
            (37, 400, 335, 0.0),
            (38, 350, 325, 0.0),
        ]
        attempt.returns.append(descent)

        self.assertIsNone(
            classifier.reconnected_return(backward, attempt),
        )

    def test_perspective_return_does_not_stitch_overlapping_fragments(self):
        classifier = self.classifier()
        attempt = Attempt(10, (800, 100))
        descent = [
            (30 + frame, 100 + frame * 20, 100, 0.0)
            for frame in range(8)
        ]
        overlap = [
            (36 + frame, 220 + frame * 20, 100, 0.0)
            for frame in range(8)
        ]
        attempt.returns.append(descent)

        self.assertIsNone(
            classifier.reconnected_return(overlap, attempt),
        )

    def test_return_does_not_reconnect_a_slow_old_ball(self):
        classifier = self.classifier()
        attempt = Attempt(10, (800, 100))
        first = [
            (30 + frame, 100 + frame * 20, 100, 0.0)
            for frame in range(8)
        ]
        rolling = [
            (45 + frame, 260 + frame * 2, 100, 0.0)
            for frame in range(8)
        ]
        attempt.returns.append(first)

        self.assertIsNone(classifier.reconnected_return(rolling, attempt))

    def test_launch_during_unresolved_return_keeps_two_bounded_attempts(self):
        classifier = self.classifier()
        classifier.table = np.float32([(0, 0), (600, 0), (600, 200), (0, 200)])
        first_launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        unresolved = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(8)]
        next_launch = [(60 + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]

        classifier.process_tracks([first_launch], 18)
        classifier.process_tracks([unresolved], 39)
        classifier.process_tracks([next_launch], 78)

        self.assertEqual([item.frame for item in classifier.active_attempts], [0, 60])
        self.assertEqual(classifier.active_attempts[0].state, "return_seen")
        self.assertLessEqual(len(classifier.active_attempts), 3)

    def test_delayed_fragment_retains_old_owner_after_next_launch(self):
        classifier = self.classifier()
        classifier.table = np.float32([(0, 0), (600, 0), (600, 200), (0, 200)])
        first_launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        first = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(8)]
        next_launch = [(60 + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        continuation = [(45 + frame, 250 + frame * 20, 100, 0.0) for frame in range(8)]

        classifier.process_tracks([first_launch], 18)
        classifier.process_tracks([first], 39)
        classifier.process_tracks([next_launch], 78)
        owner, combined = classifier.associate_return(continuation, 80)

        self.assertIsNotNone(owner)
        self.assertEqual(owner.frame, 0)
        self.assertEqual(combined, first + continuation)
        self.assertEqual(
            classifier.track_owners[classifier.track_key(continuation)], 0,
        )

    def test_track_ownership_is_exclusive_and_ties_are_deterministic(self):
        classifier = self.classifier()
        classifier.table = np.float32([(0, 0), (600, 0), (600, 200), (0, 200)])
        launches = [
            [(offset + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
            for offset in (0, 60)
        ]
        classifier.process_tracks([launches[0]], 18)
        classifier.process_tracks([launches[1]], 78)
        fragment = [(90 + frame, 100 + frame * 20, 100, 0.0) for frame in range(8)]

        owner, _ = classifier.associate_return(fragment, 100)
        same_owner, _ = classifier.associate_return(fragment, 101)

        self.assertIsNotNone(owner)
        self.assertIs(owner, same_owner)
        key = classifier.track_key(fragment)
        self.assertEqual(
            sum(key in item.owned_track_keys for item in classifier.active_attempts), 1,
        )

    def test_delayed_contact_settles_in_launch_order(self):
        settled = []
        classifier = self.classifier()
        classifier.table = np.float32([(0, 0), (600, 0), (600, 200), (0, 200)])
        classifier.net_line = np.float32([(300, 0), (300, 500)])
        classifier.on_attempt_finished = settled.append
        first_launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        first = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(8)]
        next_launch = [(60 + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        continuation = [(45 + frame, 250 + frame * 20, 100, 0.0) for frame in range(8)]

        classifier.process_tracks([first_launch], 18)
        classifier.process_tracks([first], 39)
        self.assertEqual(
            classifier.active_attempt.contact_candidates[-1].signal_type, "net",
        )
        classifier.process_tracks([next_launch], 78)
        owner, combined = classifier.associate_return(continuation, 80)
        assert owner is not None and combined is not None
        classifier.add_bounce(
            combined, combined[-4], combined[-6:-4], combined[-3:-1], 80,
            attempt=owner,
        )
        owner.state = "settled"
        classifier.drain_settled_attempts(80)

        self.assertEqual(settled, [None])
        self.assertEqual(classifier.events[0].attempt_frame_number, 0)
        self.assertEqual([item.frame for item in classifier.active_attempts], [60])

    def test_later_hit_streams_despite_an_unresolved_older_return(self):
        reported = []
        classifier = self.classifier()
        classifier.table = np.float32([(0, 0), (600, 0), (600, 200), (0, 200)])
        classifier.on_confirmed_hit = reported.append
        first_launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        unresolved = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(8)]
        next_launch = [(60 + frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        next_return = [(90 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.process_tracks([first_launch], 18)
        classifier.process_tracks([unresolved], 39)
        classifier.process_tracks([next_launch], 78)
        newer = classifier.active_attempt
        classifier.add_bounce(
            next_return, next_return[4], next_return[1:4], next_return[5:8],
            99, attempt=newer,
        )

        self.assertEqual(len(reported), 1)

    def test_hit_notification_is_not_duplicated_when_launch_order_settles(self):
        reported = []
        classifier = self.classifier()
        classifier.on_confirmed_hit = reported.append
        older = Attempt(10, (800, 100), state="return_seen")
        newer = Attempt(70, (800, 100))
        classifier.active_attempts.extend((older, newer))
        path = [
            (90 + frame, 100 + frame * 20, 100, 0.0)
            for frame in range(9)
        ]

        classifier.add_bounce(
            path, path[4], path[1:4], path[5:8], 99, attempt=newer,
        )
        self.assertEqual(len(reported), 1)

        older.state = "settled"
        newer.state = "settled"
        classifier.drain_settled_attempts(100)

        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0].outcome, "far_table")

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

    def test_profile_return_projects_contact_hidden_by_center_scoreboard(self):
        calibration = {
            "table_surface_y": 0.7786086,
            "camera_geometry": "profile_side_view",
            "control_points": [
                {"name": "x0_opponent_edge", "image": [800, 340]},
            ],
            "launcher_region": [600, 0, 900, 500],
            "return_region": [0, 0, 300, 500],
        }
        classifier = self.classifier(calibration)
        classifier.table = np.float32([
            (120, 340), (800, 340), (800, 380), (120, 380),
        ])
        classifier.net_line = np.float32([(474, 340), (474, 380)])
        path = [
            (
                200 + index,
                float(x),
                float(.000978 * x * x - .768 * x + 429.4),
                0.0,
            )
            for index, x in enumerate(np.linspace(165, 456, 24))
        ]

        hit = classifier.projected_profile_contact(path)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit[1], 650, delta=15)
        self.assertEqual(hit[2], 340)

    def test_short_pre_net_profile_fragment_is_not_projected_as_a_hit(self):
        calibration = {
            "table_surface_y": 0.7786086,
            "camera_geometry": "profile_side_view",
            "control_points": [
                {"name": "x0_opponent_edge", "image": [800, 340]},
            ],
            "launcher_region": [600, 0, 900, 500],
            "return_region": [0, 0, 300, 500],
        }
        classifier = self.classifier(calibration)
        classifier.table = np.float32([
            (120, 340), (800, 340), (800, 380), (120, 380),
        ])
        classifier.net_line = np.float32([(474, 340), (474, 380)])
        path = [
            (
                200 + index,
                float(x),
                float(.000978 * x * x - .768 * x + 429.4),
                0.0,
            )
            for index, x in enumerate(np.linspace(300, 455, 12))
        ]

        self.assertIsNone(classifier.projected_profile_contact(path))

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

    def test_telemetry_title_ignores_stronger_blue_table_edge(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[577:586, 815:988] = (255, 0, 0)
        frame[660:670, 230:1502] = (255, 0, 0)

        self.assertEqual(
            telemetry_title_bounds(frame),
            (815, 577, 988, 586),
        )

    def test_compressed_four_pixel_hud_digits_are_read(self):
        zero = np.uint8([
            [0, 255, 255, 255, 255, 0],
            [255, 255, 0, 0, 255, 255],
            [255, 255, 0, 0, 255, 255],
            [0, 255, 255, 255, 255, 0],
        ])
        five = np.uint8([
            [255, 255, 255, 255, 0],
            [255, 255, 255, 255, 0],
            [0, 0, 255, 255, 255],
            [255, 0, 255, 255, 255],
        ])

        self.assertEqual(classify_digit(zero)[0], "0")
        self.assertEqual(classify_digit(five)[0], "5")

    def test_net_obscured_spin_digits_are_read(self):
        nine = np.uint8([
            [0, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ]) * 255
        four = np.uint8([
            [0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0],
        ]) * 255

        self.assertEqual(classify_digit(nine)[0], "9")
        self.assertEqual(classify_digit(four)[0], "4")

    def test_black_arena_compression_variants_are_read(self):
        five = np.uint8([
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 0],
        ]) * 255
        six = np.uint8([
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1],
        ]) * 255
        seven = np.uint8([
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]) * 255
        eight = np.uint8([
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ]) * 255

        self.assertEqual(classify_digit(five)[0], "5")
        self.assertEqual(classify_digit(six)[0], "6")
        self.assertEqual(classify_digit(seven)[0], "7")
        self.assertEqual(classify_digit(eight)[0], "8")

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

    def test_two_pixel_wide_hud_component_can_be_split(self):
        mask = np.full((1, 2), 255, dtype=np.uint8)

        parts = split_wide_component(mask, (0, 0, 2, 1, 2))

        self.assertEqual([part.shape for part in parts], [(1, 1), (1, 1)])

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

    def test_telemetry_combines_repeated_fields_across_imperfect_frames(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        direction = {
            "x": 0,
            "y": 1,
            "angle_degrees": 90,
            "label": "up",
        }
        reader = TelemetryReader(stable_samples=1)
        reader.bounds = (0, 0, 1, 1)
        components = [
            (12.3, 87, None),
            (12.3, 87, None),
            (None, None, direction),
            (None, None, direction),
        ]

        with patch(
            "analyze_video.read_telemetry_components",
            side_effect=components,
        ):
            readings = [
                reader.update(frame, frame_number)
                for frame_number in range(len(components))
            ]

        self.assertIsNone(readings[0])
        self.assertIsNone(readings[1])
        self.assertIsNone(readings[2])
        self.assertIsNotNone(readings[3])
        self.assertEqual(readings[3].speed_mps, 12.3)
        self.assertEqual(readings[3].spin_revolutions_per_second, 87)
        self.assertEqual(readings[3].spin_direction["label"], "up")

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

    def test_machine_screen_is_not_substituted_for_missing_hit_screen(self):
        classifier = self.classifier()
        direction = {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"}
        machine = TelemetryReading(2, 10.5, 51, direction)
        launch = [(frame, 800 - frame * 10, 100, 0.0) for frame in range(18)]
        path = [(30 + frame, 100 + frame * 20, 100, 0.0) for frame in range(9)]

        classifier.observe_telemetry(machine)
        classifier.start_attempt(launch, 18)
        classifier.add_bounce(path, path[4], path[1:4], path[5:8], 39)
        classifier.finish_attempt(40)

        record = classifier.events[0].to_record()
        self.assertEqual(record["machine"]["spin_revolutions_per_second"], 51)
        self.assertNotIn("hit", record)

    def test_stale_post_launch_screen_is_not_used_as_hit_telemetry(self):
        classifier = self.classifier()
        direction = {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"}
        machine = TelemetryReading(2, 10.5, 51, direction)
        stale = TelemetryReading(20, 15.0, 90, direction)
        attempt = Attempt(
            0, (0, 0),
            machine_telemetry=machine,
            telemetry_after_launch=[stale],
        )

        hit, attached_machine = classifier.telemetry_pair_for_attempt(
            attempt, 100,
        )

        self.assertIsNone(hit)
        self.assertEqual(attached_machine, machine)

    def test_obscured_player_spin_infers_leading_one(self):
        reading = TelemetryReading(
            25, 15.0, 9,
            {"x": 0, "y": 1, "angle_degrees": 90, "label": "up"},
        )

        record = reading.to_player_record(60)

        self.assertEqual(record["spin_revolutions_per_second"], 109)
        self.assertTrue(record["spin_leading_digit_inferred"])

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

    def test_cadence_does_not_fill_long_idle_setup_or_tail_periods(self):
        events = [self.cadence_event(frame) for frame in (7382, 7461, 7539)]
        events[-1].draw_frame = 12882

        normalized = normalize_attempt_events(events, total_frames=12882, fps=60)

        self.assertEqual(len(normalized), 3)
        self.assertEqual(
            [event.outcome for event in normalized],
            ["hit", "hit", "hit"],
        )
        self.assertGreaterEqual(normalized[0].frame_number, 7300)

    def test_live_normalizer_publishes_one_pending_and_final_upsert_per_slot(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            event = self.cadence_event(frame)
            normalizer.observe(event)
            normalizer.settle_attempt()
        normalizer.finish_session(250)

        self.assertEqual(
            [(item["attempt_id"], item["state"]) for item in reported],
            [
                ("attempt-0001", "pending"),
                ("attempt-0002", "pending"),
                ("attempt-0003", "pending"),
                ("attempt-0001", "finalized"),
                ("attempt-0002", "finalized"),
                ("attempt-0003", "finalized"),
                ("attempt-0004", "pending"),
            ],
        )
        self.assertEqual(
            [item["outcome"] for item in reported if item["state"] == "finalized"],
            ["hit", "hit", "hit"],
        )

    def test_live_normalizer_deduplicates_adjacent_contact_tracks_before_cadence(self):
        reported = []
        normalizer = LiveAttemptNormalizer(30, reported.append)
        for frame in (180, 225, 226):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)

        self.assertIsNone(normalizer.period)
        self.assertEqual(reported, [])

        event = self.cadence_event(342)
        normalizer.observe_confirmed_hit(event)
        normalizer.observe(event)

        self.assertAlmostEqual(normalizer.period, 40.5)
        anchors = [
            item["anchor_frame_number"]
            for item in reported if item["state"] == "pending"
        ]
        self.assertEqual(len(anchors), len(set(anchors)))

    def test_live_normalizer_reports_warmup_until_configured_hit_count(self):
        reported = []
        statuses = []
        normalizer = LiveAttemptNormalizer(
            30,
            reported.append,
            minimum_cadence_hits=6,
            on_status=statuses.append,
        )
        for frame in (100, 140, 180, 220, 260):
            normalizer.observe_attempt_started(frame - 20)
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)

        self.assertEqual(reported, [])
        self.assertEqual(statuses[-1]["status"], "warming_up")

        event = self.cadence_event(300)
        normalizer.observe_confirmed_hit(event)
        normalizer.observe(event)

        self.assertTrue(reported)
        self.assertAlmostEqual(normalizer.period, 40.0)

    def test_direct_live_publisher_streams_detector_hit_once(self):
        reported = []
        publisher = DirectLiveAttemptPublisher(60, reported.append)
        publisher.observe_attempt_started(100)
        hit = self.cadence_event(130)
        hit.attempt_frame_number = 100

        publisher.observe_confirmed_hit(hit)
        publisher.observe(hit)

        self.assertEqual(
            [(item["state"], item.get("outcome")) for item in reported],
            [("pending", None), ("finalized", "hit")],
        )

    def test_direct_live_publisher_corrects_launch_miss_with_delayed_hit(self):
        reported = []
        publisher = DirectLiveAttemptPublisher(60, reported.append)
        publisher.observe_attempt_started(100)
        publisher.observe_attempt_started(200)
        hit = self.cadence_event(130)
        hit.attempt_frame_number = 100

        publisher.observe_confirmed_hit(hit)

        first = [
            item for item in reported
            if item["attempt_id"] == "launch-100"
            and item["state"] == "finalized"
        ]
        self.assertEqual(
            [(item["outcome"], item.get("revision", 0)) for item in first],
            [("miss", 0), ("hit", 1)],
        )

    def test_direct_live_publisher_finalizes_last_pending_launch_at_session_end(self):
        reported = []
        publisher = DirectLiveAttemptPublisher(60, reported.append)
        publisher.observe_attempt_started(100)

        publisher.finish_session(220)

        self.assertEqual(reported[-1]["state"], "finalized")
        self.assertEqual(reported[-1]["outcome"], "miss")
        self.assertEqual(reported[-1]["attempt_frame_number"], 100)

    def test_live_normalizer_finalized_attempts_are_monotonic(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190, 250):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)
            normalizer.settle_attempt()
        normalizer.observe_confirmed_non_hit(
            self.cadence_event(250, "off_table", .58)
        )
        normalizer.finish_session(310)

        finalized = [
            item for item in reported if item["state"] == "finalized"
        ]
        by_id = {}
        for item in finalized:
            by_id.setdefault(item["attempt_id"], set()).add(item["outcome"])
        self.assertTrue(all(len(outcomes) == 1 for outcomes in by_id.values()))
        self.assertEqual(len(finalized), len(by_id))

    def test_live_normalizer_publishes_emitted_hit_before_session_end(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            event = self.cadence_event(frame)
            normalizer.observe(event)
            normalizer.settle_attempt()

        fourth = self.cadence_event(250)
        normalizer.observe(fourth)
        normalizer.settle_attempt()

        finalized = [
            item for item in reported
            if item.get("sequence") == 4 and item["state"] == "finalized"
        ]
        self.assertEqual(
            [(item["outcome"], item.get("revision", 0)) for item in finalized],
            [("hit", 0)],
        )

    def test_live_normalizer_refines_future_cadence_without_moving_latest_id(self):
        normalizer = LiveAttemptNormalizer(
            60, lambda _attempt: None, minimum_cadence_hits=6,
        )
        normalizer.period = 82.0
        normalizer.phase = 70.0
        normalizer.ledger = [
            normalizer.attempt_record(
                index, round(70 + index * 82), "pending",
            )
            for index in range(8)
        ]
        normalizer.events = [
            self.cadence_event(round(70 + index * 81.4))
            for index in range(8)
        ]
        latest_anchor = normalizer.ledger[-1]["anchor_frame_number"]

        normalizer.refine_cadence()

        self.assertAlmostEqual(normalizer.period, 81.4)
        self.assertAlmostEqual(
            normalizer.phase + 7 * normalizer.period,
            latest_anchor,
        )

    def test_live_normalizer_does_not_finalize_a_provisional_out_before_hit(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)
            normalizer.settle_attempt()

        hit = self.cadence_event(250)
        provisional_out = self.cadence_event(250, "off_table", .58)
        normalizer.observe_confirmed_non_hit(provisional_out)

        self.assertFalse(any(
            item.get("sequence") == 4
            and item["state"] == "finalized"
            and item.get("outcome") == "out"
            for item in reported
        ))

        normalizer.observe_confirmed_hit(hit)
        normalizer.observe(hit)
        normalizer.settle_attempt()
        normalizer.finish_session(310)

        finalized = [
            item for item in reported if item["state"] == "finalized"
        ]
        self.assertEqual(
            [item["outcome"] for item in finalized],
            ["hit", "hit", "hit", "hit"],
        )

    def test_live_normalizer_accepts_previous_hit_after_newer_hit_arrives(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)

        newer = self.cadence_event(310)
        normalizer.observe_confirmed_hit(newer)
        normalizer.observe(newer)
        previous = self.cadence_event(250)
        normalizer.observe_confirmed_hit(previous)
        normalizer.observe(previous)
        normalizer.finish_session(370)

        by_id = {}
        for item in reported:
            if item["state"] != "finalized":
                continue
            current = by_id.get(item["attempt_id"])
            if (
                current is None
                or item.get("revision", 0) > current.get("revision", 0)
            ):
                by_id[item["attempt_id"]] = item
        finalized = sorted(by_id.values(), key=lambda item: item["sequence"])
        self.assertEqual(
            [item["outcome"] for item in finalized],
            ["hit", "hit", "hit", "hit", "hit"],
        )

    def test_confirmed_hit_corrects_an_inferred_finalized_miss(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)
        normalizer.advance(382)

        delayed = self.cadence_event(250)
        normalizer.observe_confirmed_hit(delayed)

        fourth = [
            item for item in reported
            if item.get("sequence") == 4
            and item["state"] == "finalized"
        ]
        self.assertEqual(
            [(item["outcome"], item.get("revision", 0)) for item in fourth],
            [("miss", 0), ("hit", 1)],
        )

    def test_batch_normalizer_ignores_late_provisional_out_evidence(self):
        events = [self.cadence_event(frame) for frame in (70, 130, 190, 250, 310)]
        events.append(self.cadence_event(235, "off_table", .58))

        normalized = normalize_attempt_events(events, total_frames=370, fps=60)

        self.assertEqual(
            [event.outcome for event in normalized],
            ["hit", "hit", "hit", "hit", "hit"],
        )

    def test_live_normalizer_finalizes_unseen_miss_at_next_credible_slot(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 190, 250):
            event = self.cadence_event(frame)
            normalizer.observe(event)
            normalizer.settle_attempt()
        normalizer.finish_session(310)

        finalized = [
            item for item in reported if item["state"] == "finalized"
        ]
        self.assertEqual(
            [item["outcome"] for item in finalized],
            ["hit", "miss", "hit", "hit"],
        )
        self.assertEqual(
            len({item["attempt_id"] for item in finalized}), len(finalized),
        )

    def test_live_normalizer_finalizes_unseen_miss_at_cadence_deadline(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 130, 190):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)
            normalizer.settle_attempt()

        normalizer.advance(381)
        self.assertFalse(any(
            item.get("sequence") == 4 and item["state"] == "finalized"
            for item in reported
        ))
        normalizer.advance(382)

        deadline = next(
            item for item in reported
            if item.get("sequence") == 4 and item["state"] == "finalized"
        )
        self.assertEqual(deadline["outcome"], "miss")
        self.assertEqual(deadline["decision_frame_number"], 382)

    def test_later_confirmed_hit_is_the_decision_frame_for_inferred_misses(self):
        reported = []
        normalizer = LiveAttemptNormalizer(60, reported.append)
        for frame in (70, 190, 250):
            event = self.cadence_event(frame)
            normalizer.observe_confirmed_hit(event)
            normalizer.observe(event)

        normalizer.settle_attempt(next_launch_frame=370)
        missed = next(
            item for item in reported
            if item.get("outcome") == "miss"
        )

        self.assertEqual(missed["decision_frame_number"], 250)

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

if __name__ == "__main__":
    unittest.main()

"""End-to-end replay regressions for the user-visible live counter."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "test" / "fixtures" / "side-view-live-counter.json"
VIDEO = ROOT / "side-view-regression.mkv"
CALIBRATION = ROOT / "artifacts" / "live-2026-07-24-side-calibration.json"
QUEST_FIXTURE = ROOT / "test" / "fixtures" / "quest-2026-07-20-live-counter.json"
QUEST_VIDEO = ROOT / "artifacts" / "quest-2026-07-20-live-counter.mkv"
sys.path.insert(0, str(ROOT / "scripts"))
from live_counter_replay import (  # noqa: E402
    expected_streaks,
    read_jsonl,
    reconcile_live_messages,
    run_replay,
    streak_transitions,
    verify_records,
)
from analyze_video import BounceEvent, LiveAttemptNormalizer  # noqa: E402


@unittest.skipUnless(VIDEO.exists(), "side-view video is a local fixture")
class SideViewLiveCounterReplayTest(unittest.TestCase):
    def test_labeled_side_view_replays_through_live_normalizer(self):
        mismatches = run_replay(FIXTURE)

        self.assertEqual(mismatches, [], "\n" + "\n".join(mismatches))

    def test_expected_ledger_is_derived_from_confirmed_human_labels(self):
        fixture = json.loads(FIXTURE.read_text())
        labels_path = ROOT / fixture["ground_truth"]
        labels = json.loads(labels_path.read_text())["labels"]

        self.assertEqual(
            fixture["outcomes"],
            [label["outcome"] for label in labels],
        )


@unittest.skipUnless(QUEST_VIDEO.exists(), "Quest failure capture is local")
class QuestLiveCounterReplayTest(unittest.TestCase):
    def test_user_labeled_capture_publishes_exact_attempt_ledger(self):
        fixture = json.loads(QUEST_FIXTURE.read_text())
        with tempfile.TemporaryDirectory() as directory:
            canonical = Path(directory) / "canonical.jsonl"
            live = Path(directory) / "live.jsonl"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "analyze_video.py"),
                str(QUEST_VIDEO),
                "--output", str(canonical),
                "--live-events", str(live),
                "--no-annotated",
            ], cwd=ROOT, capture_output=True, text=True, check=True)

            finalized = reconcile_live_messages(read_jsonl(live))

        self.assertEqual(
            [item["outcome"] for item in finalized], fixture["outcomes"],
        )
        self.assertEqual(
            [item["sequence"] for item in finalized],
            list(range(1, len(fixture["outcomes"]) + 1)),
        )
        self.assertEqual(
            len({item["attempt_id"] for item in finalized}), len(finalized),
        )
        for item in finalized:
            limit = fixture[
                f"max_{item['outcome']}_publication_delay_seconds"
            ]
            self.assertLessEqual(
                item["attempt_publication_delay_seconds"], limit, item,
            )


class StructuredLiveNormalizerTest(unittest.TestCase):
    def test_labeled_hit_no_swing_and_out_sequence(self):
        fixture = json.loads(
            (ROOT / "test" / "fixtures" / "structured-live-counter.json").read_text()
        )
        processing_frame = 0
        publications = []

        def publish(attempt):
            publications.append({
                **attempt,
                "publication_frame_number": processing_frame,
            })

        def detected_event(launch, outcome):
            hit = outcome == "hit"
            frame = launch + 30 if hit or outcome == "out" else launch
            return BounceEvent(
                video_time_seconds=frame / 60,
                video_timestamp=f"frame {frame}",
                hit_table=hit,
                is_in=hit,
                outcome="far_table" if hit else (
                    "off_table" if outcome == "out" else "unknown"
                ),
                posx=0.0 if hit else None,
                posy=0.0 if hit else None,
                posz=0.5 if hit else None,
                confidence=.9 if hit else .5,
                frame_number=frame,
                pixel=(0, 0),
                draw_frame=frame + 2,
            )

        normalizer = LiveAttemptNormalizer(60, publish)
        pending = None
        for item in fixture["attempts"]:
            launch = item["launch_frame_number"]
            processing_frame = launch + fixture["launch_detection_delay_frames"]
            if pending is not None:
                normalizer.observe(pending)
                normalizer.settle_attempt(launch)
            pending = detected_event(launch, item["outcome"])
            if item["outcome"] == "hit":
                processing_frame = pending.draw_frame
                normalizer.observe_confirmed_hit(pending)
        processing_frame = fixture["attempts"][-1]["launch_frame_number"] + 60
        normalizer.observe(pending)
        normalizer.finish_session(processing_frame)

        expected = [item["outcome"] for item in fixture["attempts"]]
        finalized = reconcile_live_messages([
            {"type": "attempt_upsert", **item}
            for item in publications
        ])
        self.assertEqual(
            [item["outcome"] for item in finalized],
            expected,
            finalized,
        )
        self.assertEqual(len(finalized), len(fixture["attempts"]))
        self.assertEqual(
            streak_transitions(finalized), expected_streaks(expected),
        )
        attempt_ids = [item["attempt_id"] for item in finalized]
        self.assertEqual(len(attempt_ids), len(set(attempt_ids)))
        for index in fixture["no_swing_indexes"]:
            publication = finalized[index]
            # One launcher-like track can be a fragment of the current ball.
            # Wait for corroborating cadence evidence, bounded by the same
            # conservative 2.2-period deadline used by the live pipeline.
            deadline = publication["anchor_frame_number"] + round(2.2 * 60)
            self.assertLessEqual(
                publication["publication_frame_number"], deadline, finalized,
            )

    def test_mismatch_report_includes_expected_actual_timestamp_and_delay(self):
        fixture = {
            "outcomes": ["hit"],
            "max_no_swing_publication_delay_seconds": 2,
        }
        record = {
            "outcome": "miss",
            "frame_number": 60,
            "attempt_frame_number": 60,
            "video_timestamp": "00:01.000",
            "publication_frame_number": 120,
            "publication_delay_seconds": 1.0,
        }

        mismatches = verify_records(fixture, [record], [record])

        self.assertIn(
            "#1 expected=hit actual=miss shot=00:01.000 delay=1.0s",
            mismatches,
        )


@unittest.skipUnless(VIDEO.exists(), "side-view video is a fixture")
class CleanRecordingTest(unittest.TestCase):
    def test_clean_recording_losslessly_preserves_bounded_processed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = Path(directory) / "clean.mkv"
            output = Path(directory) / "events.jsonl"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "analyze_video.py"),
                str(VIDEO),
                "--calibration", str(CALIBRATION),
                "--output", str(output),
                "--clean-recording", str(clean),
                "--clean-recording-seconds", ".1",
                "--clean-recording-start", "immediate",
                "--end-seconds", ".5",
                "--no-annotated",
            ], cwd=ROOT, capture_output=True, text=True, check=True)

            source = cv2.VideoCapture(str(VIDEO))
            capture = cv2.VideoCapture(str(clean))
            try:
                self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 6)
                self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1024)
                self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 576)
                source_ok, source_frame = source.read()
                clean_ok, clean_frame = capture.read()
                self.assertTrue(source_ok)
                self.assertTrue(clean_ok)
                expected = cv2.resize(
                    source_frame, (1024, 576), interpolation=cv2.INTER_AREA,
                )
                self.assertTrue((clean_frame == expected).all())
            finally:
                source.release()
                capture.release()


if __name__ == "__main__":
    unittest.main()

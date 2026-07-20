"""Tests for live shot event delivery."""

import sys
import unittest
from pathlib import Path
from argparse import Namespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_counter_server import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    ShotEventBroker,
    analyzer_command,
)


class ShotEventBrokerTest(unittest.TestCase):
    def test_subscriber_receives_unchanged_shot_data(self):
        events = ShotEventBroker()
        updates = events.subscribe()
        shot = {"outcome": "hit", "frame_number": 42}

        events.publish(shot)

        event_id, received = updates.get_nowait()
        self.assertEqual(event_id, 1)
        self.assertEqual(received, shot)

    def test_subscription_replays_only_events_after_given_id(self):
        events = ShotEventBroker()
        events.publish({"outcome": "hit"})
        events.publish({"outcome": "miss"})
        events.publish({"outcome": "out"})

        updates = events.subscribe(after_event_id=1)

        self.assertEqual(updates.get_nowait(), (2, {"outcome": "miss"}))
        self.assertEqual(updates.get_nowait(), (3, {"outcome": "out"}))

    def test_browser_only_resumes_an_event_id_from_the_same_server_session(self):
        events = ShotEventBroker()

        self.assertEqual(events.resume_index(events.stream_id(12)), 12)
        self.assertEqual(events.resume_index("previous-session:12"), 0)
        self.assertEqual(events.resume_index("12"), 0)

    def test_reconnect_after_restart_replays_the_new_session_from_the_start(self):
        previous = ShotEventBroker()
        previous.publish({"outcome": "hit", "frame_number": 10})
        stale_id = previous.stream_id(1)
        restarted = ShotEventBroker()
        shot = {"outcome": "miss", "frame_number": 20}
        restarted.publish(shot)

        updates = restarted.subscribe(restarted.resume_index(stale_id))

        self.assertEqual(updates.get_nowait(), (1, shot))

    def test_analyzer_command_forwards_annotated_video_path(self):
        args = Namespace(
            video="srt://camera:9000",
            output="shots.jsonl",
            calibration=None,
            annotated="artifacts/live-debug.mp4",
            clean_recording=None,
            clean_recording_seconds=120,
            clean_recording_start="launch",
            live_events=None,
        )

        command = analyzer_command(args)

        self.assertEqual(command[-2:], ["--annotated", "artifacts/live-debug.mp4"])

    def test_analyzer_command_forwards_bounded_clean_capture_and_live_log(self):
        args = Namespace(
            video="srt://camera:9000",
            output="shots.jsonl",
            calibration=None,
            annotated=None,
            clean_recording="artifacts/live-clean.mkv",
            clean_recording_seconds=90,
            clean_recording_start="launch",
            live_events="artifacts/live-events.jsonl",
        )

        command = analyzer_command(args)

        self.assertEqual(command[-8:], [
            "--clean-recording", "artifacts/live-clean.mkv",
            "--clean-recording-seconds", "90",
            "--clean-recording-start", "launch",
            "--live-events", "artifacts/live-events.jsonl",
        ])


if __name__ == "__main__":
    unittest.main()

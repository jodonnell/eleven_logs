"""Tests for live shot event delivery."""

import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_counter_server import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    ShotEventBroker,
    analyzer_command,
    counter_urls,
    run_analyzer,
    stop_analyzer,
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

    def test_status_reports_message_count_and_completed_source(self):
        events = ShotEventBroker()
        events.publish({"outcome": "hit"})

        self.assertEqual(events.status(), {"done": False, "messages": 1})
        events.mark_source_done()

        self.assertEqual(events.status(), {"done": True, "messages": 1})

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
            clean_recording_codec="ffv1",
            live_events=None,
            realtime=False,
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
            clean_recording_codec="mjpeg",
            live_events="artifacts/live-events.jsonl",
            realtime=False,
        )

        command = analyzer_command(args)

        self.assertEqual(command[-10:], [
            "--clean-recording", "artifacts/live-clean.mkv",
            "--clean-recording-seconds", "90",
            "--clean-recording-start", "launch",
            "--clean-recording-codec", "mjpeg",
            "--live-events", "artifacts/live-events.jsonl",
        ])

    def test_analyzer_command_forwards_realtime_file_pacing(self):
        args = Namespace(
            video="recording.mp4",
            output="shots.jsonl",
            calibration=None,
            annotated=None,
            clean_recording=None,
            clean_recording_seconds=120,
            clean_recording_start="launch",
            clean_recording_codec="ffv1",
            live_events=None,
            realtime=True,
        )

        self.assertEqual(analyzer_command(args)[-1], "--realtime")

    def test_wildcard_bind_advertises_loopback_and_routed_lan_urls(self):
        with patch("live_counter_server.lan_address_for", return_value="192.168.1.42"):
            urls = counter_urls(
                "0.0.0.0", 8000, "srt://192.168.1.197:9000",
            )

        self.assertEqual(urls, [
            "http://127.0.0.1:8000",
            "http://192.168.1.42:8000",
        ])

    def test_stop_analyzer_interrupts_and_waits_for_clean_exit(self):
        process = MagicMock()
        process.poll.return_value = None

        stop_analyzer(process, timeout_seconds=3)

        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.wait.assert_called_once_with(timeout=3)
        process.terminate.assert_not_called()

    def test_stop_analyzer_terminates_after_interrupt_timeout(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("analyzer", 3),
            0,
        ]

        stop_analyzer(process, timeout_seconds=3)

        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.terminate.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def test_analyzer_exit_is_published_and_appended_to_live_log(self):
        with tempfile.TemporaryDirectory() as directory:
            live_events = Path(directory) / "events.jsonl"
            args = Namespace(
                video="srt://camera:9000",
                output="shots.jsonl",
                calibration=None,
                annotated=None,
                clean_recording=None,
                clean_recording_seconds=120,
                clean_recording_start="launch",
                clean_recording_codec="ffv1",
                live_events=str(live_events),
                wait_for_subscriber=False,
                realtime=False,
            )
            process = MagicMock()
            process.stdout = [
                json.dumps({"type": "attempt_upsert", "outcome": "hit"}) + "\n",
            ]
            process.wait.return_value = 7
            events = ShotEventBroker()
            holder = []

            with patch(
                "live_counter_server.subprocess.Popen",
                return_value=process,
            ):
                run_analyzer(args, events, holder)

            updates = events.subscribe()
            self.assertEqual(updates.get_nowait()[1]["type"], "attempt_upsert")
            exit_event = updates.get_nowait()[1]
            self.assertEqual(exit_event["type"], "analyzer_exit")
            self.assertEqual(exit_event["returncode"], 7)
            self.assertEqual(events.status(), {"done": True, "messages": 2})
            self.assertEqual(
                json.loads(live_events.read_text(encoding="utf-8"))["type"],
                "analyzer_exit",
            )


if __name__ == "__main__":
    unittest.main()

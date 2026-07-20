"""Tests for live shot event delivery."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_counter_server import ShotEventBroker  # pyright: ignore[reportMissingImports]  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

"""Unit tests for evaluation label storage and video range requests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from label_evaluation import LabelStore, parse_byte_range  # noqa: E402


class LabelEvaluationTest(unittest.TestCase):
    def test_range_parser_supports_browser_video_requests(self):
        self.assertEqual(parse_byte_range("bytes=100-199", 1000), (100, 199))
        self.assertEqual(parse_byte_range("bytes=900-", 1000), (900, 999))
        self.assertEqual(parse_byte_range("bytes=-100", 1000), (900, 999))
        with self.assertRaises(ValueError):
            parse_byte_range("bytes=1000-", 1000)

    def test_labels_are_validated_sorted_and_saved_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LabelStore(root / "labels.json", root / "video.mkv")

            saved = store.write({"labels": [
                {"time_seconds": 2.12349, "outcome": "miss"},
                {"time_seconds": 1, "outcome": "hit"},
            ]})

            self.assertEqual(
                saved["labels"],
                [
                    {"time_seconds": 1.0, "outcome": "hit"},
                    {"time_seconds": 2.123, "outcome": "miss"},
                ],
            )
            self.assertEqual(json.loads(store.path.read_text()), saved)
            self.assertFalse(store.path.with_suffix(".json.tmp").exists())

    def test_invalid_outcomes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LabelStore(root / "labels.json", root / "video.mkv")
            with self.assertRaises(ValueError):
                store.write({"labels": [
                    {"time_seconds": 1, "outcome": "out"},
                ]})


if __name__ == "__main__":
    unittest.main()

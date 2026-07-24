"""Tests for human-label versus browser-SSE comparison."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_sse_labels import compare  # noqa: E402


class CompareSseLabelsTest(unittest.TestCase):
    def test_browser_reconciliation_exposes_missing_attempt(self):
        truth = {"labels": [
            {"time_seconds": 1.0, "outcome": "hit"},
            {"time_seconds": 2.0, "outcome": "miss"},
            {"time_seconds": 3.0, "outcome": "hit"},
        ]}
        messages = [
            {
                "type": "attempt_upsert", "attempt_id": "launch-1",
                "sequence": 1, "anchor_frame_number": 30, "state": "pending",
            },
            {
                "type": "attempt_upsert", "attempt_id": "launch-1",
                "sequence": 1, "anchor_frame_number": 30, "state": "finalized",
                "frame_number": 60, "outcome": "hit",
                "attempt_publication_delay_seconds": 0.5,
            },
            {
                "type": "attempt_upsert", "attempt_id": "launch-3",
                "sequence": 2, "anchor_frame_number": 150, "state": "finalized",
                "frame_number": 180, "outcome": "hit",
                "attempt_publication_delay_seconds": 0.5,
            },
        ]

        report = compare(truth, messages)

        sequence = report["browser_reconciled"]["evaluation"]["sequence"]
        self.assertEqual(sequence["match"], 2)
        self.assertEqual(sequence["missing"], 1)
        self.assertEqual(report["raw_sse"]["left_pending"], 0)
        self.assertEqual(
            report["browser_reconciled"]["streak"]["exact_after_attempt"], 1,
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for ordered detector evaluation alignment and metrics."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_detector import align_outcomes, evaluate  # noqa: E402


class EvaluateDetectorTest(unittest.TestCase):
    def truth(self, outcomes):
        return {"labels": [
            {"time_seconds": index * 1.3, "outcome": outcome}
            for index, outcome in enumerate(outcomes)
        ]}

    def predictions(self, outcomes):
        return [
            {"frame_number": index * 78, "outcome": outcome}
            for index, outcome in enumerate(outcomes)
        ]

    def test_missing_prediction_does_not_shift_later_results(self):
        report = evaluate(
            self.truth(["hit", "miss", "miss", "hit"]),
            self.predictions(["hit", "miss", "hit"]),
        )

        self.assertEqual(report["sequence"]["match"], 3)
        self.assertEqual(report["sequence"]["missing"], 1)
        self.assertEqual(report["sequence"]["wrong_outcome"], 0)

    def test_extra_prediction_is_reported_separately(self):
        report = evaluate(
            self.truth(["hit", "miss", "hit"]),
            self.predictions(["hit", "miss", "miss", "hit"]),
        )

        self.assertEqual(report["sequence"]["match"], 3)
        self.assertEqual(report["sequence"]["extra"], 1)

    def test_wrong_hit_miss_updates_confusion_matrix(self):
        report = evaluate(
            self.truth(["hit", "miss"]),
            self.predictions(["miss", "hit"]),
        )

        self.assertEqual(report["sequence"]["wrong_outcome"], 2)
        self.assertEqual(report["hit_classification"]["false_hit"], 1)
        self.assertEqual(report["hit_classification"]["false_miss"], 1)


if __name__ == "__main__":
    unittest.main()

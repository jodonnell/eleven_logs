"""Tests for report-driven mismatch review manifests."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from review_mismatches import build_manifest  # noqa: E402


class ReviewMismatchesTest(unittest.TestCase):
    def test_manifest_uses_prediction_launch_and_links_nearby_diagnostics(self):
        report = {
            "expected_attempts": 2,
            "predicted_attempts": 2,
            "sequence": {"match": 1, "wrong_outcome": 1, "missing": 0,
                         "extra": 0, "accuracy": 0.5},
            "hit_classification": {"true_hit": 1, "false_hit": 1,
                                   "true_miss": 0, "false_miss": 0,
                                   "precision": 0.5, "recall": 1, "f1": 2 / 3},
            "errors": [{
                "kind": "wrong_outcome", "expected_number": 2,
                "expected_outcome": "miss", "label_time_seconds": 12.0,
                "predicted_number": 2, "predicted_outcome": "hit",
                "prediction_time_seconds": 10.0,
            }],
        }
        diagnostics = [{
            "reason": "shadow hit_frame=630", "points": [[600, 1, 2, 3], [640, 4, 5, 6]],
        }]

        manifest = build_manifest(report, diagnostics, 60, 2.5, 1.5)

        review = manifest["reviews"][0]
        self.assertEqual(review["anchor_time_seconds"], 10.0)
        self.assertEqual(review["clip_start_seconds"], 7.5)
        self.assertEqual(review["nearby_bounce_diagnostics"][0]["signal"], "shadow")
        self.assertIsNone(review["category"])

    def test_missing_launch_falls_back_to_ground_truth_time(self):
        report = {
            "expected_attempts": 1, "predicted_attempts": 0,
            "sequence": {"match": 0, "wrong_outcome": 0, "missing": 1,
                         "extra": 0, "accuracy": 0},
            "hit_classification": {"true_hit": 0, "false_hit": 0,
                                   "true_miss": 0, "false_miss": 1,
                                   "precision": 0, "recall": 0, "f1": 0},
            "errors": [{
                "kind": "missing", "expected_number": 1,
                "expected_outcome": "hit", "label_time_seconds": 3.0,
                "predicted_number": None, "predicted_outcome": None,
                "prediction_time_seconds": None,
            }],
        }

        review = build_manifest(report, [], 60, 2.5, 1.5)["reviews"][0]

        self.assertEqual(review["anchor_time_seconds"], 3.0)
        self.assertEqual(review["clip_start_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()

"""Focused unit tests for the classical-CV bounce helpers."""
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_video import find_bounce, shadow_contact_score  # noqa: E402


class VideoDetectorUnitTest(unittest.TestCase):
    def test_shadow_score_rises_for_dark_table_patch_below_ball(self):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, :] = (65, 165, 160)  # green table
        hsv[55:70, 42:58, 2] = 105      # ball's dark table shadow

        self.assertGreater(shadow_contact_score(hsv, (50, 50)), 30)

    def test_shadow_score_is_zero_on_evenly_lit_table(self):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[:, :, :] = (65, 165, 160)

        self.assertEqual(shadow_contact_score(hsv, (50, 50)), 0.0)

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


if __name__ == "__main__":
    unittest.main()

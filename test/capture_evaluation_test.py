"""Tests for the classifier-free evaluation capture command."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_evaluation import capture_command  # noqa: E402


class CaptureEvaluationTest(unittest.TestCase):
    def test_stream_is_remuxed_without_video_reencoding(self):
        command = capture_command("srt://example", Path("capture.mkv"), 1200)

        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertEqual(command[command.index("-t") + 1], "1200")
        self.assertNotIn("FFV1", command)


if __name__ == "__main__":
    unittest.main()

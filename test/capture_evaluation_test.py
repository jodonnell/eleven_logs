"""Tests for the classifier-free evaluation capture command."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_evaluation import capture_command, capture_manifest, default_output  # noqa: E402


class CaptureEvaluationTest(unittest.TestCase):
    def test_default_capture_uses_its_own_timestamped_run_directory(self):
        output = default_output()

        self.assertEqual(output.parent.parent, ROOT / "artifacts" / "runs")
        self.assertRegex(output.parent.name, r"^evaluation-\d{4}-\d{2}-\d{2}-\d{6}$")
        self.assertEqual(output.name, "capture.mkv")

    def test_manifest_records_a_workspace_relative_recording(self):
        output = ROOT / "artifacts" / "runs" / "evaluation-test" / "capture.mkv"

        manifest = capture_manifest("srt://example", output, 1200)

        self.assertEqual(
            manifest["recording"],
            "artifacts/runs/evaluation-test/capture.mkv",
        )
        self.assertEqual(manifest["maximum_duration_seconds"], 1200)

    def test_stream_is_remuxed_without_video_reencoding(self):
        command = capture_command("srt://example", Path("capture.mkv"), 1200)

        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertEqual(command[command.index("-t") + 1], "1200")
        self.assertNotIn("FFV1", command)


if __name__ == "__main__":
    unittest.main()

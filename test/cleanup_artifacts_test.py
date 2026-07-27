"""Tests for safe artifact retention."""

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cleanup_artifacts import cleanup_candidates  # noqa: E402


class CleanupArtifactsTest(unittest.TestCase):
    def test_selects_only_old_unpinned_disposable_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            old_run = artifacts / "runs" / "old-run"
            fresh_run = artifacts / "runs" / "fresh-run"
            pinned_run = artifacts / "runs" / "pinned-run"
            archive = artifacts / "runs" / "archive"
            old_scratch = artifacts / "scratch" / "old-scratch"
            for path in (
                old_run, fresh_run, pinned_run, archive, old_scratch,
            ):
                path.mkdir(parents=True)
            (pinned_run / "PINNED").touch()

            old_time = 1_000_000
            fresh_time = old_time + 20 * 86400
            for path in (old_run, pinned_run, archive, old_scratch):
                os.utime(path, (old_time, old_time))
            os.utime(fresh_run, (fresh_time, fresh_time))

            candidates = list(cleanup_candidates(
                artifacts, older_than_days=14, now=fresh_time,
            ))

            self.assertEqual(candidates, [old_run, old_scratch])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for file and live-stream video source selection."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from video_source import (  # noqa: E402
    FileVideoSource,
    SrtVideoSource,
    VideoSourceError,
    open_video_source,
)


class FakeCapture:
    def __init__(self, _path, *_args):
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        return {
            cv2.CAP_PROP_FPS: 60,
            cv2.CAP_PROP_FRAME_WIDTH: 16,
            cv2.CAP_PROP_FRAME_HEIGHT: 12,
            cv2.CAP_PROP_POS_FRAMES: self.position,
        }.get(prop, 0)

    def set(self, prop, value):
        import cv2

        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        elif prop == cv2.CAP_PROP_POS_MSEC:
            self.position = round(value * 60 / 1000)
        return True

    def read(self):
        image = np.full((12, 16, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, image

    def release(self):
        self.released = True


class VideoSourceTest(unittest.TestCase):
    def test_local_input_uses_seekable_opencv_source(self):
        capture = FakeCapture("sample.mp4")
        with patch("video_source.cv2.VideoCapture", return_value=capture):
            source = open_video_source("sample.mp4")
            source.seek_seconds(2)
            frame = source.read()
            source.close()

        self.assertIsInstance(source, FileVideoSource)
        self.assertEqual(frame.number, 120)
        self.assertEqual(frame.time_seconds, 2)
        self.assertTrue(capture.released)

    def test_srt_input_uses_opencv_ffmpeg_backend(self):
        capture = FakeCapture("srt://127.0.0.1:9000")
        with patch("video_source.cv2.VideoCapture", return_value=capture) as constructor:
            source = open_video_source("srt://127.0.0.1:9000?mode=listener")
            frame = source.read()
            source.close()

        self.assertIsInstance(source, SrtVideoSource)
        self.assertEqual((source.width, source.height), (16, 12))
        self.assertEqual(source.fps, 60)
        self.assertEqual(frame.number, 0)
        self.assertEqual(frame.image.shape, (12, 16, 3))
        constructor.assert_called_once_with(
            "srt://127.0.0.1:9000?mode=listener", cv2.CAP_FFMPEG,
        )
        self.assertTrue(capture.released)

    def test_live_source_rejects_file_seek(self):
        capture = FakeCapture("srt://127.0.0.1:9000")
        with patch("video_source.cv2.VideoCapture", return_value=capture):
            source = SrtVideoSource("srt://127.0.0.1:9000")
            with self.assertRaisesRegex(VideoSourceError, "cannot seek"):
                source.seek_seconds(1)
            source.close()


if __name__ == "__main__":
    unittest.main()

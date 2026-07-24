"""Sequential OpenCV video sources for local files and live SRT streams."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


PathLike = Union[str, Path]


class VideoSourceError(ValueError):
    """Raised when a video source cannot be opened or decoded."""


class RealtimeLagMonitor:
    """Report when processing a nominally real-time stream falls behind."""

    def __init__(
        self,
        fps: float,
        threshold_seconds: float = 2.0,
    ):
        self.fps = fps
        self.threshold_seconds = threshold_seconds
        self.lag_seconds = 0.0
        self.behind = False

    def observe_processing_time(self, seconds: float) -> Optional[str]:
        frame_budget = 1 / self.fps
        self.lag_seconds = max(
            0.0, self.lag_seconds + seconds - frame_budget,
        )
        if not self.behind and self.lag_seconds >= self.threshold_seconds:
            self.behind = True
            return (
                "WARNING: live video processing is "
                f"{self.lag_seconds:.1f}s behind real time"
            )
        if self.behind and self.lag_seconds < self.threshold_seconds / 2:
            self.behind = False
            return (
                "Live video processing caught up "
                f"({self.lag_seconds:.1f}s behind real time)"
            )
        return None


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame and its source-relative position."""

    number: int
    time_seconds: float
    image: np.ndarray


class VideoSource:
    """Small interface shared by seekable files and forward-only streams."""

    fps: float
    width: int
    height: int
    seekable: bool = False
    live: bool = False

    def read(self) -> Optional[VideoFrame]:
        raise NotImplementedError

    def seek_seconds(self, seconds: float) -> None:
        if seconds:
            raise VideoSourceError("This video source cannot seek by timestamp")

    def seek_frame(self, frame: int) -> None:
        if frame:
            raise VideoSourceError("This video source cannot seek by frame number")

    def close(self) -> None:
        raise NotImplementedError


class FileVideoSource(VideoSource):
    """A seekable file decoded by OpenCV."""

    seekable = True

    def __init__(self, path: PathLike):
        self.path = str(path)
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoSourceError(f"Could not open {path}")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 60.0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            self.close()
            raise VideoSourceError(f"Could not determine the video size for {path}")

    def read(self) -> Optional[VideoFrame]:
        number = round(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        ok, image = self._capture.read()
        if not ok:
            return None
        return VideoFrame(number, number / self.fps, image)

    def seek_seconds(self, seconds: float) -> None:
        if seconds < 0:
            raise VideoSourceError("Seek time cannot be negative")
        self._capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

    def seek_frame(self, frame: int) -> None:
        if frame < 0:
            raise VideoSourceError("Frame number cannot be negative")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def close(self) -> None:
        self._capture.release()


class SrtVideoSource(VideoSource):
    """A forward-only SRT stream decoded by OpenCV's FFmpeg backend."""

    live = True

    def __init__(self, url: str):
        self._capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self._capture.isOpened():
            self._capture.release()
            self.close()
            raise VideoSourceError("Could not open the SRT video stream")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 60.0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            self.close()
            raise VideoSourceError("Could not determine the SRT video size")
        self._next_frame = 0
        self._lag_monitor = RealtimeLagMonitor(self.fps)
        self._last_frame_delivered_at: Optional[float] = None
        print(
            f"SRT video connected: {self.width}x{self.height} at {self.fps:g} FPS",
            file=sys.stderr,
            flush=True,
        )

    def read(self) -> Optional[VideoFrame]:
        read_started_at = time.monotonic()
        if self._last_frame_delivered_at is not None:
            lag_message = self._lag_monitor.observe_processing_time(
                read_started_at - self._last_frame_delivered_at,
            )
            if lag_message is not None:
                print(lag_message, file=sys.stderr, flush=True)
        ok, image = self._capture.read()
        if not ok:
            return None
        number = self._next_frame
        self._next_frame += 1
        self._last_frame_delivered_at = time.monotonic()
        return VideoFrame(number, number / self.fps, image)

    def close(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()


def open_video_source(location: PathLike) -> VideoSource:
    """Select a source implementation from a path or URL."""
    value = str(location)
    if value.lower().startswith("srt://"):
        return SrtVideoSource(value)
    return FileVideoSource(value)

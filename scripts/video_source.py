"""Sequential OpenCV video sources for local files and live SRT streams."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

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

    def set_event_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Receive live-source lifecycle records when the source supports them."""
        return


class FileVideoSource(VideoSource):
    """A seekable file decoded by OpenCV."""

    seekable = True

    def __init__(self, path: PathLike, realtime: bool = False):
        self.path = str(path)
        self.realtime = realtime
        self._realtime_started_at: Optional[float] = None
        self._media_started_at: Optional[float] = None
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
        media_time = number / self.fps
        if self.realtime:
            now = time.monotonic()
            if self._realtime_started_at is None or self._media_started_at is None:
                self._realtime_started_at = now
                self._media_started_at = media_time
            target = (
                self._realtime_started_at
                + media_time
                - self._media_started_at
            )
            delay = target - now
            if delay > 0:
                time.sleep(delay)
        return VideoFrame(number, media_time, image)

    def _reset_realtime_clock(self) -> None:
        self._realtime_started_at = None
        self._media_started_at = None

    def seek_seconds(self, seconds: float) -> None:
        if seconds < 0:
            raise VideoSourceError("Seek time cannot be negative")
        self._capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        self._reset_realtime_clock()

    def seek_frame(self, frame: int) -> None:
        if frame < 0:
            raise VideoSourceError("Frame number cannot be negative")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self._reset_realtime_clock()

    def close(self) -> None:
        self._capture.release()


class SrtVideoSource(VideoSource):
    """A forward-only SRT stream decoded by OpenCV's FFmpeg backend."""

    live = True

    def __init__(
        self,
        url: str,
        open_timeout_msec: int = 5000,
        read_timeout_msec: int = 3000,
        reconnect_delay_seconds: float = 1.0,
    ):
        self.url = url
        self.open_timeout_msec = open_timeout_msec
        self.read_timeout_msec = read_timeout_msec
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._event_callback: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None
        self._capture = self._open_capture()
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
        self._reconnect_count = 0
        self._lag_monitor = RealtimeLagMonitor(self.fps)
        self._last_frame_delivered_at: Optional[float] = None
        print(
            f"SRT video connected: {self.width}x{self.height} at {self.fps:g} FPS",
            file=sys.stderr,
            flush=True,
        )

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_msec,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_msec,
            ],
        )

    def _source_record(self, event_type: str, **values: Any) -> Dict[str, Any]:
        return {
            "type": event_type,
            "logged_at_unix_seconds": round(time.time(), 3),
            "frame_number": self._next_frame,
            "reconnect_count": self._reconnect_count,
            **values,
        }

    def _emit(self, event_type: str, **values: Any) -> None:
        record = self._source_record(event_type, **values)
        if self._event_callback is not None:
            self._event_callback(record)

    def set_event_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        self._event_callback = callback
        self._emit(
            "source_connected",
            width=self.width,
            height=self.height,
            fps=self.fps,
        )

    def _reconnect(self) -> None:
        self._capture.release()
        attempt = 0
        while True:
            attempt += 1
            self._emit("source_reconnecting", attempt=attempt)
            time.sleep(self.reconnect_delay_seconds)
            capture = self._open_capture()
            if not capture.isOpened():
                capture.release()
                continue
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (width, height) != (self.width, self.height):
                self._emit(
                    "source_reconnect_rejected",
                    attempt=attempt,
                    reason="resolution_changed",
                    width=width,
                    height=height,
                    expected_width=self.width,
                    expected_height=self.height,
                )
                capture.release()
                continue
            self._capture = capture
            self._reconnect_count += 1
            self._last_frame_delivered_at = None
            self._lag_monitor = RealtimeLagMonitor(self.fps)
            self._emit(
                "source_reconnected",
                attempt=attempt,
                width=width,
                height=height,
                fps=capture.get(cv2.CAP_PROP_FPS) or self.fps,
            )
            print(
                f"SRT video reconnected after {attempt} attempt(s)",
                file=sys.stderr,
                flush=True,
            )
            return

    def read(self) -> Optional[VideoFrame]:
        while True:
            read_started_at = time.monotonic()
            if self._last_frame_delivered_at is not None:
                lag_message = self._lag_monitor.observe_processing_time(
                    read_started_at - self._last_frame_delivered_at,
                )
                if lag_message is not None:
                    print(lag_message, file=sys.stderr, flush=True)
            ok, image = self._capture.read()
            if ok:
                number = self._next_frame
                self._next_frame += 1
                self._last_frame_delivered_at = time.monotonic()
                return VideoFrame(number, number / self.fps, image)
            self._emit(
                "source_stalled",
                read_timeout_msec=self.read_timeout_msec,
            )
            print(
                "WARNING: SRT frame read timed out; reconnecting",
                file=sys.stderr,
                flush=True,
            )
            self._reconnect()

    def close(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()


def open_video_source(location: PathLike, realtime: bool = False) -> VideoSource:
    """Select a source implementation from a path or URL."""
    value = str(location)
    if value.lower().startswith("srt://"):
        return SrtVideoSource(value)
    return FileVideoSource(value, realtime=realtime)

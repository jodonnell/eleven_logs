"""Live publishing, progress logging, health checks, and output helpers."""

import argparse
import base64
import json
import math
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from auto_calibrate import calibration_from_frame, hue_distance, infer_ball_color
from video_source import VideoFrame, VideoSource, VideoSourceError, open_video_source

from .models import BounceEvent, PathLike, TelemetryReading
from .detection import fmt_timestamp

class DirectLiveAttemptPublisher:
    """Publish detector-native launch results without cadence reconstruction."""

    def __init__(
        self,
        fps: float,
        on_attempt: Callable[[Dict[str, Any]], None],
    ) -> None:
        self.fps = fps
        self.on_attempt = on_attempt
        self.attempts: Dict[int, Dict[str, Any]] = {}
        self.launch_order: List[int] = []

    def attempt_record(
        self,
        anchor: int,
        state: str,
        event: Optional[BounceEvent] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "attempt_id": f"launch-{anchor}",
            "sequence": self.launch_order.index(anchor) + 1,
            "anchor_frame_number": anchor,
            "state": state,
        }
        if event is not None:
            record.update(event.to_record())
            record["outcome"] = (
                "hit"
                if event.hit_table and event.outcome == "far_table"
                else "miss"
            )
            record["attempt_frame_number"] = anchor
            record["decision_frame_number"] = event.draw_frame
        return record

    def observe_attempt_started(self, anchor: int) -> None:
        if anchor in self.attempts:
            return
        # A newly detected machine launch closes any older launch that still
        # has no detector result. A delayed confirmed hit can revise it later.
        for previous_anchor in self.launch_order:
            previous = self.attempts[previous_anchor]
            if previous["state"] != "pending":
                continue
            missed = BounceEvent(
                video_time_seconds=round(previous_anchor / self.fps, 3),
                video_timestamp=fmt_timestamp(previous_anchor / self.fps),
                hit_table=False,
                is_in=False,
                outcome="unknown",
                posx=None,
                posy=None,
                posz=None,
                confidence=0.3,
                frame_number=previous_anchor,
                pixel=(0, 0),
                draw_frame=anchor,
                attempt_frame_number=previous_anchor,
            )
            finalized = self.attempt_record(
                previous_anchor, "finalized", missed,
            )
            self.attempts[previous_anchor] = finalized
            self.on_attempt(finalized)
        self.launch_order.append(anchor)
        pending = self.attempt_record(anchor, "pending")
        self.attempts[anchor] = pending
        self.on_attempt(pending)

    def publish_event(self, event: BounceEvent) -> None:
        anchor = (
            event.attempt_frame_number
            if event.attempt_frame_number is not None
            else event.frame_number
        )
        if anchor not in self.attempts:
            self.observe_attempt_started(anchor)
        existing = self.attempts[anchor]
        finalized = self.attempt_record(anchor, "finalized", event)
        if existing["state"] == "finalized":
            if (
                existing.get("outcome") == finalized["outcome"]
                or finalized["outcome"] != "hit"
            ):
                return
            finalized["revision"] = existing.get("revision", 0) + 1
        self.attempts[anchor] = finalized
        self.on_attempt(finalized)

    def observe(self, event: BounceEvent) -> None:
        self.publish_event(event)

    def observe_confirmed_hit(self, event: BounceEvent) -> None:
        self.publish_event(event)

    def observe_confirmed_non_hit(self, _event: BounceEvent) -> None:
        # Completed detector events arrive through ``observe``. Do not let a
        # provisional track ending race a subsequently confirmed contact.
        return

    def settle_attempt(self, _next_launch_frame: Optional[int] = None) -> None:
        return

    def advance(self, _frame_number: int) -> None:
        return

    def finish_session(self, total_frames: Optional[int] = None) -> None:
        """Close the final visible launch when the source session ends."""
        if total_frames is None:
            return
        for anchor in self.launch_order:
            current = self.attempts[anchor]
            if current["state"] != "pending":
                continue
            missed = BounceEvent(
                video_time_seconds=round(anchor / self.fps, 3),
                video_timestamp=fmt_timestamp(anchor / self.fps),
                hit_table=False,
                is_in=False,
                outcome="unknown",
                posx=None,
                posy=None,
                posz=None,
                confidence=0.3,
                frame_number=anchor,
                pixel=(0, 0),
                draw_frame=total_frames,
                attempt_frame_number=anchor,
            )
            finalized = self.attempt_record(anchor, "finalized", missed)
            self.attempts[anchor] = finalized
            self.on_attempt(finalized)


class LivePipelineLogger:
    """Emit bounded, continuous evidence that the live detector is advancing."""

    def __init__(
        self,
        fps: float,
        write_record: Callable[[Dict[str, Any]], None],
        interval_seconds: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self.fps = fps
        self.write_record = write_record
        self.interval_seconds = interval_seconds
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.started_at: Optional[float] = None
        self.started_frame: Optional[int] = None
        self.last_logged_at: Optional[float] = None
        self.last_metrics: Dict[str, Any] = {}

    def timing(self, frame_number: int, now: float) -> Dict[str, Any]:
        if self.started_at is None or self.started_frame is None:
            self.started_at = now
            self.started_frame = frame_number
        wall_elapsed = now - self.started_at
        source_elapsed = (frame_number - self.started_frame) / self.fps
        return {
            "frame_number": frame_number,
            "video_time_seconds": round(frame_number / self.fps, 3),
            "wall_elapsed_seconds": round(wall_elapsed, 3),
            "source_elapsed_seconds": round(source_elapsed, 3),
            "estimated_lag_seconds": round(
                max(0.0, wall_elapsed - source_elapsed), 3,
            ),
            "processing_fps": round(
                (frame_number - self.started_frame) / wall_elapsed, 2,
            ) if wall_elapsed > 0 else None,
        }

    def observe(
        self,
        frame_number: int,
        metrics: Dict[str, Any],
    ) -> None:
        now = self.monotonic()
        self.last_metrics = metrics
        if (
            self.last_logged_at is not None
            and now - self.last_logged_at < self.interval_seconds
        ):
            return
        self.last_logged_at = now
        self.write_record({
            "type": "pipeline_heartbeat",
            "logged_at_unix_seconds": round(self.wall_time(), 3),
            **self.timing(frame_number, now),
            **metrics,
        })

    def finish(self, frame_number: int, reason: str) -> None:
        now = self.monotonic()
        self.write_record({
            "type": "pipeline_end",
            "reason": reason,
            "logged_at_unix_seconds": round(self.wall_time(), 3),
            **self.timing(frame_number, now),
            **self.last_metrics,
        })


class LiveCounterHealthMonitor:
    """Surface live semantic failures that do not crash the pipeline."""

    def __init__(
        self,
        write_record: Callable[[Dict[str, Any]], None],
        publisher_event_threshold: int = 3,
        no_hit_attempt_threshold: int = 8,
    ) -> None:
        self.write_record = write_record
        self.publisher_event_threshold = publisher_event_threshold
        self.no_hit_attempt_threshold = no_hit_attempt_threshold
        self.active: Set[str] = set()
        self.finalized: Dict[str, str] = {}

    def set_warning(self, code: str, active: bool, message: str) -> None:
        if active and code not in self.active:
            self.active.add(code)
            self.write_record({
                "type": "counter_health",
                "status": "warning",
                "code": code,
                "message": message,
            })
        elif not active and code in self.active:
            self.active.remove(code)
            self.write_record({
                "type": "counter_health",
                "status": "recovered",
                "code": code,
                "message": "Detector health recovered",
            })

    def observe_pipeline(self, metrics: Dict[str, Any]) -> None:
        stalled = (
            metrics.get("detected_event_count", 0)
            >= self.publisher_event_threshold
            and metrics.get("attempt_upsert_count", 0) == 0
        )
        self.set_warning(
            "publisher_stalled",
            stalled,
            "Detector is active but the counter is not publishing attempts",
        )

    def observe_attempt(self, attempt: Dict[str, Any]) -> None:
        if attempt.get("state") != "finalized":
            return
        self.finalized[attempt["attempt_id"]] = attempt.get("outcome", "miss")
        no_hits = (
            len(self.finalized) >= self.no_hit_attempt_threshold
            and not any(outcome == "hit" for outcome in self.finalized.values())
        )
        self.set_warning(
            "no_confirmed_contacts",
            no_hits,
            (
                f"No table contacts confirmed after {len(self.finalized)} attempts; "
                "check camera framing and calibration"
            ),
        )


def attach_missing_machine_telemetry(
    events: Sequence[BounceEvent],
    readings: Sequence[TelemetryReading],
    fps: float,
) -> List[BounceEvent]:
    """Attach nearby HUD states to events inferred during normalization."""
    attached = []
    for event in events:
        # A tracked return that crossed the net necessarily followed a player
        # contact. Use HUD states preceding its terminal event; choosing the
        # nearest state can incorrectly grab the next machine launch.
        if event.hit is None and event.return_crossed_net:
            preceding = [
                reading for reading in readings
                if reading.frame_number <= event.frame_number
            ]
            if preceding:
                hit = preceding[-1]
                machine = preceding[-2] if len(preceding) >= 2 else None
                event = replace(
                    event,
                    hit=hit.to_record(fps),
                    machine=(
                        machine.to_record(fps)
                        if machine else event.machine
                    ),
                )
                attached.append(event)
                continue
        if event.machine is not None or event.hit is not None or not readings:
            attached.append(event)
            continue
        nearest = min(readings, key=lambda item: abs(item.frame_number - event.frame_number))
        if abs(nearest.frame_number - event.frame_number) <= fps * .6:
            event = replace(event, machine=nearest.to_record(fps))
        attached.append(event)
    return attached


def create_video_writer(
    path: PathLike,
    fps: float,
    size: Tuple[int, int],
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """Create an annotated-video writer or fail before processing begins."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        raise SystemExit(f"Could not create annotated video at {path}")
    return writer


def reset_output_file(path: PathLike) -> None:
    """Start a new analysis session with no results from the prior session."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

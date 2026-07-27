"""Shared detector settings, value objects, and diagnostic state."""

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



PROCESSING_WIDTH = 1024
MAX_SPIN_REVOLUTIONS_PER_SECOND = 150

PathLike = Union[str, Path]
Point = Tuple[float, float]
TrackPoint = Tuple[int, float, float, float]
Candidate = Tuple[float, float, float]
Track = List[TrackPoint]
Bounce = Tuple[TrackPoint, Track, Track]
Calibration = Dict[str, Any]

@dataclass(frozen=True)
class DetectorSettings:
    """Tunable classical-CV thresholds, optionally overridden per camera."""

    max_gap: int = 5
    min_track_points: int = 9
    min_launch_track_points: int = 18
    min_track_observations: int = 3
    track_match_distance: float = 100
    min_track_speed: float = 0.5
    max_track_speed: float = 200
    max_track_acceleration: float = 200
    max_prediction_error: float = 100
    prediction_error_per_gap: float = 10
    max_direction_change_degrees: float = 170
    launch_min_horizontal_distance: float = 120
    launch_min_directional_ratio: float = 0.8
    return_min_horizontal_distance: float = 120
    return_min_horizontal_speed: float = 4
    return_reconnect_max_gap: int = 24
    return_reconnect_max_forward_distance: float = 320
    return_reconnect_backtrack_tolerance: float = 20
    profile_overlap_max_frames: int = 3
    profile_overlap_match_distance: float = 12
    min_shadow_contact_score: float = 28
    net_shadow_exclusion_distance: float = 70
    motion_threshold: int = 18
    bright_ball_lower: Tuple[int, int, int] = (0, 0, 210)
    bright_ball_upper: Tuple[int, int, int] = (180, 145, 255)
    min_candidate_area: int = 2
    near_min_candidate_area: int = 4
    max_candidate_area: int = 500
    far_max_candidate_area_ratio: float = 0.20
    max_candidate_aspect_ratio: float = 2.2
    min_candidate_compactness: float = 0.45
    min_candidate_brightness: float = 210
    max_candidate_saturation: float = 145
    ball_hue_center: Optional[float] = None
    ball_hue_tolerance: float = 9
    ball_min_saturation: float = 70
    ball_min_value: float = 70
    table_hue_center: float = 65
    table_hue_tolerance: float = 23
    table_min_saturation: float = 80
    return_direction_x: float = 1.0
    return_direction_y: float = 0.0
    min_vertical_turn: float = 1
    min_pre_bounce_speed: float = 12
    max_post_bounce_speed_ratio: float = 0.35
    flattening_strength_weight: float = 0.6
    table_contact_margin: float = 10
    terminal_shadow_frames: int = 2

    @classmethod
    def from_calibration(cls, calibration: Calibration) -> "DetectorSettings":
        configured = calibration.get("detector_settings", {})
        valid = {item.name for item in fields(cls)}
        values = {name: value for name, value in configured.items() if name in valid}
        if table_color := calibration.get("table_color"):
            values.update({
                "table_hue_center": table_color["hue_center"],
                "table_hue_tolerance": table_color["hue_tolerance"],
                "table_min_saturation": table_color["min_saturation"],
            })
        if ball_color := calibration.get("ball_color"):
            values.update({
                "ball_hue_center": ball_color["hue_center"],
                "ball_hue_tolerance": ball_color["hue_tolerance"],
                "ball_min_saturation": ball_color["min_saturation"],
                "ball_min_value": ball_color["min_value"],
                "min_candidate_brightness": ball_color["min_value"],
                "max_candidate_saturation": 255,
            })
        if calibration.get("camera_geometry") == "elevated_end_view":
            controls = {
                item["name"]: item["image"]
                for item in calibration.get("control_points", [])
            }
            player = controls.get("x0_player_edge")
            opponent = controls.get("x0_opponent_edge")
            if player is not None and opponent is not None:
                dx, dy = opponent[0] - player[0], opponent[1] - player[1]
                length = math.hypot(dx, dy)
                if length:
                    values.update({
                        "return_direction_x": dx / length,
                        "return_direction_y": dy / length,
                        "net_shadow_exclusion_distance": 5,
                        "min_track_points": 4,
                        "return_min_horizontal_distance": 60,
                        "return_min_horizontal_speed": 2,
                    })
        return cls(**values)


@dataclass
class BounceEvent:
    video_time_seconds: float
    video_timestamp: str
    hit_table: bool
    is_in: bool
    outcome: str
    posx: Optional[float]
    posy: Optional[float]
    posz: Optional[float]
    confidence: float
    frame_number: int
    pixel: Point = field(repr=False)
    draw_frame: int = field(repr=False)
    attempt_frame_number: Optional[int] = None
    return_crossed_net: Optional[bool] = None
    hit: Optional[Dict[str, Any]] = None
    machine: Optional[Dict[str, Any]] = None

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record.pop("pixel")
        record.pop("draw_frame")
        if record["attempt_frame_number"] is None:
            record.pop("attempt_frame_number")
        if record["return_crossed_net"] is None:
            record.pop("return_crossed_net")
        if record["hit"] is None:
            record.pop("hit")
        if record["machine"] is None:
            record.pop("machine")
        return record


@dataclass(frozen=True)
class TelemetryReading:
    frame_number: int
    speed_mps: float
    spin_revolutions_per_second: int
    spin_direction: Dict[str, Any]

    def to_record(self, fps: float) -> Dict[str, Any]:
        return {
            "speed_mps": self.speed_mps,
            "spin_revolutions_per_second": self.spin_revolutions_per_second,
            "spin_direction": self.spin_direction,
            "video_time_seconds": round(self.frame_number / fps, 3),
        }

    def to_player_record(self, fps: float) -> Dict[str, Any]:
        record = self.to_record(fps)
        if self.spin_revolutions_per_second < 20:
            # The net post can hide the leading digit of a three-digit player
            # return. This player does not produce 200+ rev/s, so a visible
            # 00--19 suffix is conservatively the tail of 100--119 rev/s.
            record["spin_revolutions_per_second"] += 100
            record["spin_leading_digit_inferred"] = True
        return record


@dataclass(frozen=True)
class ContactCandidate:
    """Bounded evidence for one possible physical ball contact."""

    frame_number: int
    pixel: Point
    log_position: Optional[Tuple[float, float, float]]
    table_side: str
    signal_type: str
    strength: float
    confidence: float
    source_track_key: Tuple[int, int, int]
    approach: Tuple[TrackPoint, ...] = field(repr=False)
    departure: Tuple[TrackPoint, ...] = field(repr=False)
    accepted: bool = True
    rejection_reason: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "pixel": list(self.pixel),
            "log_position": (
                list(self.log_position) if self.log_position is not None else None
            ),
            "table_side": self.table_side,
            "signal_type": self.signal_type,
            "strength": self.strength,
            "confidence": self.confidence,
            "source_track_key": list(self.source_track_key),
            "approach": [list(point) for point in self.approach],
            "departure": [list(point) for point in self.departure],
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class Attempt:
    """Tracks one ball-machine launch and any possible return paths."""

    frame: int
    pixel: Point
    state: str = "launched"
    launch_track_key: Optional[Tuple[int, int, int]] = None
    owned_track_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
    last_evidence_frame: Optional[int] = None
    report_no_bounce: bool = True
    returns: List[Track] = field(default_factory=list)
    bounces: List[BounceEvent] = field(default_factory=list)
    bounce_track_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
    contact_candidates: List[ContactCandidate] = field(default_factory=list)
    rejected_contact_candidates: List[ContactCandidate] = field(default_factory=list)
    contact_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
    classified_contact_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
    machine_telemetry: Optional[TelemetryReading] = None
    telemetry_after_launch: List[TelemetryReading] = field(default_factory=list)


@dataclass
class ActiveTrack:
    """One currently visible candidate path and its missed-frame count."""

    points: Track
    gap: int = 0
    confirmed: bool = False


@dataclass(frozen=True)
class CandidateDiagnostic:
    """One current-frame blob shown by the diagnostic renderer."""

    center: Point
    kind: str
    reason: str = ""


@dataclass(frozen=True)
class TrackDiagnostic:
    """A completed track and the classifier decision made for it."""

    points: Track
    kind: str
    reason: str = ""


class DetectorDiagnostics:
    """Bounded rendering state that never participates in detection decisions."""

    def __init__(self, track_lifetime_frames: int = 30) -> None:
        self.track_lifetime_frames = track_lifetime_frames
        self.candidates: List[CandidateDiagnostic] = []
        self.unconfirmed_tracks: List[Track] = []
        self.recent_tracks: List[Tuple[int, TrackDiagnostic]] = []

    def begin_frame(self) -> None:
        self.candidates = []
        self.unconfirmed_tracks = []

    def candidate(self, center: Point, kind: str, reason: str = "") -> None:
        self.candidates.append(CandidateDiagnostic(center, kind, reason))

    def set_unconfirmed_tracks(self, tracks: Sequence[ActiveTrack]) -> None:
        self.unconfirmed_tracks = [
            track.points[-12:] for track in tracks if not track.confirmed
        ]

    def completed_track(self, diagnostic: TrackDiagnostic, frame_number: int) -> None:
        expires = frame_number + self.track_lifetime_frames
        self.recent_tracks.append((expires, diagnostic))
        self.recent_tracks = [
            item for item in self.recent_tracks if item[0] >= frame_number
        ]

    def visible_completed_tracks(self, frame_number: int) -> List[TrackDiagnostic]:
        self.recent_tracks = [
            item for item in self.recent_tracks if item[0] >= frame_number
        ]
        return [diagnostic for _, diagnostic in self.recent_tracks]

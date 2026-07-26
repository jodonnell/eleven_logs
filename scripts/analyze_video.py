#!/usr/bin/env python3
"""Streaming bounce analysis for Eleven Table Tennis fixed spectator footage.

Uses only the current/previous frame and bounded trajectory history.  It is
deliberately conservative: an incomplete/occluded trajectory is unknown,
rather than a fabricated table coordinate.
"""
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

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install dependencies first: python3 -m pip install --user opencv-python-headless numpy") from exc

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


DIGIT_TEMPLATES = {
    digit: np.uint8([[pixel == "1" for pixel in row] for row in bitmap.split("/")])
    for digit, bitmap in {
        "0": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000111111100000/0011111111111000/0111110001111100/0111100000111100/0111100000111110/0111100000111110/0111100000111110/0111100000111110/0111110000111100/0011111001111100/0001111111111000/0000001111000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "1": "0000000000000000/0000000011111100/0000011111111100/0000011111111100/0111111111111100/0111111011111100/0111110011111100/0000000011111100/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000011111110/0000000000000000",
        "2": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0011111111110000/0111111111111000/0011000001111100/0000000000111100/0000000001111100/0000000011111000/0000001111110000/0000111110000000/0001111100000000/0111111111111110/0111111111111110/0011000000000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "3": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0001111111100000/0111111111111000/0011000001111100/0000000000111100/0000000001111100/0000111111111000/0000111111111100/0000000000111110/0000000000111110/0110000001111100/0111111111111000/0011111111100000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "4": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000000011111000/0000001111111000/0000011111111000/0000111111111000/0001111011111000/0001110011111000/0111100011111000/0111111111111110/0111111111111110/0000000011111000/0000000001110000/0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "5": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0011111111111000/0011111111111000/0011110000000000/0011110000000000/0011111110000000/0011111111111100/0011001111111100/0000000000111110/0000000000111110/0111100011111100/0111111111111000/0000111111000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "6": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000011111111000/0001111111111000/0011111000000000/0011110011000000/0011111111111000/0111111001111100/0111100000011110/0011100000011110/0011111000111110/0001111111111100/0000011111110000/0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "7": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0111111111111110/0111111111111110/0111111111111110/0000000001111100/0000000001111000/0000000011110000/0000000111110000/0000001111100000/0000001111000000/0000011111000000/0000111110000000/0000111100000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "8": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000001111000000/0001111111111000/0011111001111100/0011110000111110/0011110000111100/0001111111111000/0001111111111100/0011110000111110/0111100000011110/0111110000011110/0011111111111110/0000111111111000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
        "9": "0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000111111110000/0011111111111100/0011110000111100/0111100000011110/0011100000011110/0011111111111110/0000111111111110/0000000000111100/0000000011111100/0001111111110000/0001111110000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000/0000000000000000",
    }.items()
}

# At the 1024px sample's TV scale a digit is only 3--4 pixels high. These
# native-resolution cores retain distinctions that disappear when the larger
# templates are downsampled (notably 0/9 and 5/6/8).
LOW_RES_DIGIT_TEMPLATES = {
    digit: [
        np.uint8([[pixel == "1" for pixel in row] for row in variant.split("/")])
        for variant in bitmap.split("|")
    ]
    for digit, bitmap in {
        "0": "0100/1011/1001/1011|1001/1001/1111|1011/1001/1111|1101/1001/1101|011110/110011/110011/011110",
        "1": "111/001/001|111/011/011|111/111/011/011|111/001/011/001|111/001/001/001",
        "2": "0011/0010/1100|0100/0011/0110/1100|0100/0011/0110/1110|01111/00001/00110/11111",
        "3": "011/110/011|100/011/110/011|11110/00110/00011/00011",
        "4": "0010/0110/1010/0011|0010/0110/1010/1011|0110/1010/1111|00011/01111/11011/00011|000110/011010/111111/000010",
        "5": "1000/1111/0011|1100/0111/0001|1110/1000/1111/0011|1110/1000/1111/1011|11110/11110/00111/10111|10000/11110/00011/11110|010000/011110/000011/111110|11110/11110/00011/00011",
        "6": "1000/1111/1011|1011/1110/1001|1100/1111/1001|01100/11000/11110/11110/01100|011110/011110/110011/010011",
        "7": "011/010/100|011/010/110|111/001/011/010|00011/00110/01100/11000",
        "8": "1011/1110/1001|1011/1110/1011|1101/0111/1101|011011/011110/110011/111111|010011/011110/110011/111111|011110/011110/011111/110011",
        "9": "0100/1011/1111/0010|1101/1111/0001|1011/1111/0011|010011/110011/001011/011110|011011/110011/001011/011110|011110/110011/011111/000110",
    }.items()
}


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


def calibration_geometry(
    data: Calibration, video_width: int, video_height: int, scale: float,
    source: str = "automatic calibration",
) -> Tuple[Calibration, np.ndarray, np.ndarray]:
    required = ("image_size", "table_surface_y", "table_polygon", "tracking_polygon", "net_line")
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"{source} is missing: {', '.join(missing)}")
    expected_size = [video_width, video_height]
    if data["image_size"] != expected_size:
        raise SystemExit(
            f"{source} is for {data['image_size']}, but this video is {expected_size}. "
            "Create a calibration for this camera/video; do not reuse it."
        )
    if "control_points" in data:
        if len(data["control_points"]) != 4:
            raise SystemExit("Calibration needs exactly four image/log control points")
        image = np.float32([point["image"] for point in data["control_points"]]) * scale
        log = np.float32([point["log"] for point in data["control_points"]])
    else:
        names = ("far_left", "far_right", "near_right", "near_left")
        image = np.float32([data["image_corners"][name] for name in names]) * scale
        log = np.float32([data["log_corners"][name] for name in names])
    table_polygon = np.float32(data["table_polygon"]) * scale
    # Contacts may use a deliberately smaller reviewed surface than the
    # rendered table outline. Keep it camera calibration data rather than a
    # detector-wide pixel constant.
    data.setdefault("table_contact_polygon", data["table_polygon"])
    return data, cv2.getPerspectiveTransform(image, log), table_polygon


def load_calibration(
    path: PathLike, video_width: int, video_height: int, scale: float
) -> Tuple[Calibration, np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text())
    return calibration_geometry(data, video_width, video_height, scale, f"Calibration {path}")


def fmt_timestamp(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def point_near_polygon(point: Point, polygon: np.ndarray, margin: float) -> bool:
    """Include contacts whose ball center is just outside a calibrated rail.

    Calibration follows the visible table edge, while the rendered ball has a
    non-zero radius and can be centred a few processing pixels beyond that
    edge on a legitimate edge bounce.
    """
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, True) >= -margin


def point_in_rectangle(point: Point, rectangle: Sequence[float], scale: float) -> bool:
    x, y = point[0] / scale, point[1] / scale
    left, top, right, bottom = rectangle
    return left <= x <= right and top <= y <= bottom


def signed_distance_to_line(point: Point, line: np.ndarray) -> float:
    """Signed perpendicular pixel distance from point to a calibrated line."""
    start, end = line
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    return ((dx * (point[1] - start[1])) - (dy * (point[0] - start[0]))) / length


def _qualified_bounces(
    points: Track,
    table_polygon: np.ndarray,
    net_line: Optional[np.ndarray] = None,
    settings: DetectorSettings = DetectorSettings(),
    allow_terminal_shadow: bool = True,
) -> List[Tuple[str, float, Bounce]]:
    """Return every qualified contact and the evidence used to rank it."""
    if len(points) < settings.min_track_points:
        return []
    qualified: List[Tuple[str, float, Bounce]] = []

    def return_progress(beginning: TrackPoint, end: TrackPoint) -> float:
        return (
            (end[1] - beginning[1]) * settings.return_direction_x
            + (end[2] - beginning[2]) * settings.return_direction_y
        )
    # A rendered ball casts a compact, moving shadow on the green table. At
    # contact the ball/shadow separation collapses, even when perspective
    # makes the bright ball's screen-space path continue in one direction.
    # This catches the clear 17s sample bounce that has no vertical reversal.
    for index in range(2, len(points)):
        score = points[index][3] if len(points[index]) > 3 else 0
        if score < settings.min_shadow_contact_score:
            continue
        pixel = (points[index][1], points[index][2])
        if not point_near_polygon(pixel, table_polygon, settings.table_contact_margin):
            continue
        if net_line is not None and abs(signed_distance_to_line(pixel, net_line)) < settings.net_shadow_exclusion_distance:
            continue  # net mesh creates a false dark "shadow"
        previous_score = points[index - 1][3]
        next_score = points[index + 1][3] if index + 1 < len(points) else None
        terminal = index >= len(points) - settings.terminal_shadow_frames
        # A real contact is an isolated convergence peak. A several-frame
        # dark plateau is usually the tracker attaching to a static table
        # marking after an off-table ball has disappeared.
        local_peak = score >= previous_score and (next_score is None or score > next_score)
        forward_contact = return_progress(points[index - 1], points[index]) >= 0
        terminal_confirmed = (
            index < len(points) - settings.terminal_shadow_frames
            or allow_terminal_shadow
        )
        if (
            local_peak
            and forward_contact
            and terminal_confirmed
            and (not terminal or return_progress(points[index - 2], points[index]) > 0)
        ):
            qualified.append((
                "shadow", float(score),
                (points[index], points[index - 2:index], points[index + 1:index + 3]),
            ))
    # Two post-contact frames are enough for a terminal turn when the ball
    # disappears behind the launcher immediately afterwards.
    for index in range(3, len(points) - 2):
        before = [p[2] for p in points[index - 3:index]]
        after = [p[2] for p in points[index + 1:index + 3]]
        y = points[index][2]
        before_mean, after_mean = sum(before) / len(before), sum(after) / len(after)
        maximum = (
            y - before_mean >= settings.min_vertical_turn
            and y - after_mean >= settings.min_vertical_turn
        )
        minimum = (
            before_mean - y >= settings.min_vertical_turn
            and after_mean - y >= settings.min_vertical_turn
        )
        if not point_near_polygon(
            (points[index][1], points[index][2]), table_polygon,
            settings.table_contact_margin,
        ):
            continue
        # A sudden backward jump along the calibrated player-to-opponent axis
        # is a tracker hand-off to a marking/shadow, not a physical bounce.
        if return_progress(points[index - 1], points[index]) < 0:
            continue
        # The short approach must belong to that same forward-moving ball.
        # A path that walks backward and then jumps forward at the apparent
        # turn is a tracker hand-off, even if its post-contact direction looks
        # plausible in isolation.
        if any(
            return_progress(beginning, end) <= 0
            for beginning, end in zip(
                points[index - 3:index - 1], points[index - 2:index]
            )
        ):
            continue
        # The same direction must hold across the two frames that confirm the
        # departure. A tracker that reverses overall immediately after the
        # apparent contact has handed off to a different blob; a returned ball
        # continues toward the opponent even while its vertical direction turns.
        if return_progress(points[index], points[index + 2]) <= 0:
            continue
        if maximum or minimum:
            strength = min(abs(y - before_mean), abs(y - after_mean))
            qualified.append((
                "trajectory", float(strength),
                (points[index], points[index - 3:index], points[index + 1:index + 3]),
            ))
        # A far-side bounce can be partly hidden by the launcher: perspective
        # may preserve the y direction but sharply flatten its velocity. This
        # is accepted only at a visible in-table point with a large slowdown.
        if index >= 3 and index + 2 < len(points):
            before_speeds = [abs(points[j][2] - points[j - 1][2]) for j in range(index - 2, index + 1)]
            after_speeds = [abs(points[j + 1][2] - points[j][2]) for j in range(index, index + 2)]
            before_speed = sum(before_speeds) / len(before_speeds)
            after_speed = sum(after_speeds) / len(after_speeds)
            if (
                before_speed >= settings.min_pre_bounce_speed
                and after_speed <= before_speed * settings.max_post_bounce_speed_ratio
            ):
                flattening = (before_speed - after_speed) * settings.flattening_strength_weight
                qualified.append((
                    "trajectory", float(flattening),
                    (points[index], points[index - 3:index], points[index + 1:index + 3]),
                ))
    return qualified


def find_bounces(
    points: Track,
    table_polygon: np.ndarray,
    net_line: Optional[np.ndarray] = None,
    settings: DetectorSettings = DetectorSettings(),
    allow_terminal_shadow: bool = True,
) -> List[Bounce]:
    """Find all qualified physical contacts in chronological order.

    Multiple evidence mechanisms can describe the same contact frame. Prefer
    the shadow signal (the legacy detector's strongest evidence family), then
    the stronger bounded trajectory signal, and expose one contact per frame.
    """
    by_frame: Dict[int, Tuple[str, float, Bounce]] = {}
    for candidate in _qualified_bounces(
        points, table_polygon, net_line, settings, allow_terminal_shadow,
    ):
        family, strength, bounce = candidate
        frame = bounce[0][0]
        previous = by_frame.get(frame)
        rank = (family == "shadow", strength)
        if previous is None or rank > (previous[0] == "shadow", previous[1]):
            by_frame[frame] = candidate
    return [by_frame[frame][2] for frame in sorted(by_frame)]


def find_bounce(
    points: Track,
    table_polygon: np.ndarray,
    net_line: Optional[np.ndarray] = None,
    settings: DetectorSettings = DetectorSettings(),
    allow_terminal_shadow: bool = True,
) -> Optional[Bounce]:
    """Preserve the legacy single-contact choice for existing callers."""
    qualified = _qualified_bounces(
        points, table_polygon, net_line, settings, allow_terminal_shadow,
    )
    shadows = [candidate for candidate in qualified if candidate[0] == "shadow"]
    if shadows:
        return shadows[0][2]
    trajectories = [candidate for candidate in qualified if candidate[0] == "trajectory"]
    return max(trajectories, key=lambda candidate: candidate[1])[2] if trajectories else None


def bounce_signal(
    hit: TrackPoint, approach: Track, departure: Track,
) -> str:
    """Describe which bounded evidence shape produced a bounce decision."""
    if len(approach) == 2:
        return "terminal" if not departure else "shadow"
    if len(approach) != 3 or len(departure) != 2:
        return "other"
    y = hit[2]
    before_mean = sum(point[2] for point in approach) / len(approach)
    after_mean = sum(point[2] for point in departure) / len(departure)
    if y >= before_mean and y >= after_mean:
        return "vertical_maximum"
    if y <= before_mean and y <= after_mean:
        return "vertical_minimum"
    return "velocity_flattening"


def bounce_strength(
    hit: TrackPoint, approach: Track, departure: Track,
) -> float:
    """Return a comparable, observation-local magnitude for diagnostics."""
    signal = bounce_signal(hit, approach, departure)
    if signal in ("shadow", "terminal"):
        return round(float(hit[3]), 3)
    if not approach or not departure:
        return 0.0
    y = hit[2]
    before_mean = sum(point[2] for point in approach) / len(approach)
    after_mean = sum(point[2] for point in departure) / len(departure)
    if signal in ("vertical_maximum", "vertical_minimum"):
        return round(min(abs(y - before_mean), abs(y - after_mean)), 3)
    before_speed = sum(
        abs(end[2] - beginning[2])
        for beginning, end in zip(approach, approach[1:] + [hit])
    ) / len(approach)
    after_speed = sum(
        abs(end[2] - beginning[2])
        for beginning, end in zip([hit] + departure[:-1], departure)
    ) / len(departure)
    return round(max(0.0, before_speed - after_speed), 3)


class MultiBallTracker:
    """Bounded multi-hypothesis tracker for several bright moving blobs.

    Keeping competing paths prevents a reflection or net highlight from
    replacing the ball during a player return. Completed tracks are emitted
    immediately, so memory is bounded by active tracks and short histories.
    """
    def __init__(self, settings: DetectorSettings = DetectorSettings()) -> None:
        self.settings = settings
        self.tracks: List[ActiveTrack] = []

    @staticmethod
    def velocity(start: TrackPoint, end: TrackPoint) -> Point:
        elapsed = end[0] - start[0]
        if elapsed <= 0:
            return (0.0, 0.0)
        return ((end[1] - start[1]) / elapsed, (end[2] - start[2]) / elapsed)

    def match_error(
        self, points: Track, frame_number: int, candidate: Candidate,
    ) -> Optional[float]:
        """Return prediction error when a candidate is a plausible next point."""
        last = points[-1]
        elapsed = frame_number - last[0]
        if elapsed <= 0:
            return None
        displacement = (candidate[0] - last[1], candidate[1] - last[2])
        velocity = (displacement[0] / elapsed, displacement[1] / elapsed)
        speed = math.hypot(*velocity)
        if not self.settings.min_track_speed <= speed <= self.settings.max_track_speed:
            return None

        if len(points) < 2:
            return math.hypot(*displacement)

        previous_velocity = self.velocity(points[-2], last)
        # Predict one observation ahead. During an occlusion, uncertainty
        # grows instead of blindly extrapolating through every absent frame;
        # a bounce or partial shadow handoff can occur inside that gap.
        previous_elapsed = last[0] - points[-2][0]
        predicted = (
            last[1] + previous_velocity[0] * previous_elapsed,
            last[2] + previous_velocity[1] * previous_elapsed,
        )
        error = math.dist(predicted, candidate[:2])
        allowed_error = (
            self.settings.max_prediction_error
            + max(0, elapsed - 1) * self.settings.prediction_error_per_gap
        )
        if error > min(self.settings.track_match_distance, allowed_error):
            return None

        previous_speed = math.hypot(*previous_velocity)
        acceleration = math.dist(previous_velocity, velocity) / elapsed
        if acceleration > self.settings.max_track_acceleration:
            return None
        if previous_speed > 0 and speed > 0:
            cosine = sum(a * b for a, b in zip(previous_velocity, velocity)) / (
                previous_speed * speed
            )
            turn = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
            if turn > self.settings.max_direction_change_degrees:
                return None
        return error

    def update(self, frame_number: int, candidates: Sequence[Candidate]) -> List[Track]:
        pairs: List[Tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            points = track.points
            for candidate_index, candidate in enumerate(candidates):
                error = self.match_error(points, frame_number, candidate)
                if error is not None:
                    pairs.append((error, track_index, candidate_index))
        pairs.sort()
        used_tracks: Set[int] = set()
        used_candidates: Set[int] = set()
        for _, track_index, candidate_index in pairs:
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_index]
            candidate = candidates[candidate_index]
            track.points.append((frame_number, candidate[0], candidate[1], candidate[2]))
            track.gap = 0
            if len(track.points) >= self.settings.min_track_observations:
                track.confirmed = True
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)
        for track_index, track in enumerate(self.tracks):
            if track_index not in used_tracks:
                track.gap += 1
        completed: List[Track] = []
        active: List[ActiveTrack] = []
        for track in self.tracks:
            if track.gap > self.settings.max_gap:
                if track.confirmed:
                    completed.append(track.points)
            else:
                active.append(track)
        self.tracks = active
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index not in used_candidates:
                self.tracks.append(ActiveTrack([
                    (frame_number, candidate[0], candidate[1], candidate[2]),
                ], confirmed=self.settings.min_track_observations <= 1))
        return completed

    @property
    def visible_points(self) -> List[TrackPoint]:
        return [
            point
            for track in self.tracks if track.confirmed
            for point in track.points[-12:]
        ]

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Return live paths that have enough observations to classify."""
        return [track.points for track in self.tracks if track.confirmed]


def telemetry_title_bounds(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Locate the wide blue Multiplayer title, our HUD scale/position anchor."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 90, 100), (145, 255, 255))
    row_counts = np.count_nonzero(blue, axis=1)
    minimum_row_pixels = max(8, frame.shape[1] * .008)
    rows = np.flatnonzero(row_counts >= minimum_row_pixels)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    groups = [group for group in groups if len(group) >= 2]
    if not groups:
        return None

    candidates = []
    for group in groups:
        local_maximum = int(row_counts[group].max())
        core_rows = group[row_counts[group] >= local_maximum * .3]
        core_groups = np.split(
            core_rows, np.flatnonzero(np.diff(core_rows) > 1) + 1,
        )
        for core in core_groups:
            if len(core) < 2:
                continue
            y0, y1 = int(core[0]), int(core[-1] + 1)
            _, xs = np.nonzero(blue[y0:y1] > 0)
            if not len(xs):
                continue
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            width_ratio = (x1 - x0) / frame.shape[1]
            # The title is wide but bounded to the TV. Reject long blue table
            # edges, which can be much stronger than the title itself.
            if .05 <= width_ratio <= .3:
                candidates.append((
                    int(row_counts[core].sum()),
                    (x0, y0, x1, y1),
                ))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def normalize_digit(mask: np.ndarray) -> np.ndarray:
    """Place one tightly cropped HUD digit in the template coordinate space."""
    height, width = mask.shape
    scale = min(14 / width, 18 / height)
    resized = cv2.resize(
        mask, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    resized = resized >= 80
    canvas = np.zeros((20, 16), dtype=np.uint8)
    y = (canvas.shape[0] - resized.shape[0]) // 2
    x = (canvas.shape[1] - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def classify_digit(mask: np.ndarray) -> Tuple[str, float]:
    normalized = normalize_digit(mask)
    if mask.shape[0] <= 5:
        scores = {}
        for digit, raw_templates in LOW_RES_DIGIT_TEMPLATES.items():
            candidate_scores = []
            for raw_template in raw_templates:
                template = normalize_digit(raw_template * 255)
                intersection = np.count_nonzero(normalized & template)
                total = np.count_nonzero(normalized) + np.count_nonzero(template)
                score = 2 * intersection / total if total else 0.0
                if raw_template.shape == mask.shape:
                    native = mask > 0
                    native_intersection = np.count_nonzero(
                        native & raw_template,
                    )
                    native_total = (
                        np.count_nonzero(native)
                        + np.count_nonzero(raw_template)
                    )
                    # Preserve native pixel geometry as a tiebreaker. Scaling a
                    # four-pixel glyph to the common template canvas can make
                    # distinct 6/8 shapes nearly identical.
                    if native_total:
                        score += .04 * (
                            2 * native_intersection / native_total
                        )
                candidate_scores.append(score)
            scores[digit] = max(candidate_scores)
        ranked = sorted(scores, key=scores.get, reverse=True)
        if len(ranked) > 1 and scores[ranked[0]] - scores[ranked[1]] < .015:
            return "?", 0.0
        digit = ranked[0]
        return digit, round(min(1.0, scores[digit]), 3)
    contours, hierarchy = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    holes = [] if hierarchy is None else [
        index for index, item in enumerate(hierarchy[0])
        if item[3] >= 0 and cv2.contourArea(contours[index]) >= mask.size * .015
    ]
    scores = {}
    for digit, template in DIGIT_TEMPLATES.items():
        template_holes = 2 if digit == "8" else (1 if digit in "0469" else 0)
        if template_holes != len(holes):
            continue
        intersection = np.count_nonzero(normalized & template)
        total = np.count_nonzero(normalized) + np.count_nonzero(template)
        scores[digit] = 2 * intersection / total if total else 0.0
    if len(holes) == 1:
        moments = cv2.moments(contours[holes[0]])
        if moments["m00"]:
            hole_y = moments["m01"] / moments["m00"] / mask.shape[0]
            if hole_y > .58:
                scores = {"6": scores.get("6", 0.0)}
            elif hole_y < .42:
                scores = {"9": scores.get("9", 0.0)}
    if not scores:
        return "?", 0.0
    digit = max(scores, key=scores.get)
    return digit, round(scores[digit], 3)


def split_wide_component(mask: np.ndarray, box: Tuple[int, int, int, int, int]) -> List[np.ndarray]:
    """Split digits joined by a one-pixel compression bridge."""
    x, y, width, height, _ = box
    glyph = mask[y:y + height, x:x + width]
    pieces = max(1, round(width / max(height * 1.05, 1)))
    if pieces == 1:
        return [glyph]
    projection = np.count_nonzero(glyph, axis=0)
    cuts = []
    for piece in range(1, pieces):
        expected = round(width * piece / pieces)
        radius = max(1, round(width / pieces * .25))
        # ``end`` is exclusive. Allow the search to include the final column;
        # capping it at ``width - 1`` makes a narrow component's slice empty
        # (for example, width=2, expected=1).
        start, end = max(1, expected - radius), min(width, expected + radius + 1)
        cuts.append(start + int(np.argmin(projection[start:end])))
    return [part for part in np.split(glyph, cuts, axis=1) if part.shape[1] > 0]


def read_hud_number(
    frame: np.ndarray, bounds: Tuple[int, int, int, int], kind: str,
) -> Optional[Union[float, int]]:
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    if kind == "speed":
        top, bottom, right, needs_decimal = 3.0, 4.2, .62, True
    else:
        top, bottom, right, needs_decimal = 4.5, 5.7, .58, False
    left = .44
    roi = frame[
        round(y0 + top * height):round(y0 + bottom * height),
        round(x0 + left * width):round(x0 + right * width),
    ]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # The 1024px capture renders these glyphs only five pixels tall, so retain
    # gray antialiasing/compression pixels as well as the white core.
    white = cv2.inRange(hsv, (0, 0, 100), (180, 180, 255))
    core = cv2.inRange(hsv, (0, 0, 145), (180, 120, 255))
    if height < 15 and kind == "spin":
        ys, xs = np.nonzero(white)
        if not len(xs):
            return None
        start, end = int(xs.min()), int(xs.max() + 1)
        pitch = max(1.0, width / 28)
        digit_count = max(1, min(3, round((end - start) / pitch)))
        digits = []
        confidence = 1.0
        for index in range(digit_count):
            left = round(start + (end - start) * index / digit_count)
            right_edge = round(start + (end - start) * (index + 1) / digit_count)
            cell = core[:, left:right_edge]
            cell_ys, cell_xs = np.nonzero(cell)
            if not len(cell_xs):
                return None
            glyph = cell[
                cell_ys.min():cell_ys.max() + 1,
                cell_xs.min():cell_xs.max() + 1,
            ]
            digit, score = classify_digit(glyph)
            digits.append(digit)
            confidence = min(confidence, score)
        return int("".join(digits)) if confidence >= .38 else None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core)
    minimum_area = max(1, round(height * height * .015))
    boxes = sorted(
        [tuple(map(int, box)) for box in stats[1:] if box[4] >= minimum_area],
        key=lambda box: box[0],
    )
    if not boxes:
        return None
    full_height = max(box[3] for box in boxes)
    decimal_centers: List[float] = []
    glyphs: List[Tuple[int, np.ndarray]] = []
    for box in boxes:
        x, y, box_width, box_height, _ = box
        if box_height < full_height * .45:
            if needs_decimal:
                decimal_centers.append(x + box_width / 2)
            continue
        glyph_mask = core if height < 15 else white
        parts = split_wide_component(glyph_mask, box)
        for index, part in enumerate(parts):
            glyphs.append((round(x + box_width * (index + .5) / len(parts)), part))
    if not glyphs:
        return None
    glyph_centers = [center for center, _ in glyphs]
    decimal_x = next((
        center for center in decimal_centers
        if any(left < center < right for left, right in zip(glyph_centers, glyph_centers[1:]))
    ), None)
    recognized = []
    confidence = 1.0
    for center_x, glyph in glyphs:
        digit, score = classify_digit(glyph)
        recognized.append((center_x, digit))
        confidence = min(confidence, score)
    if confidence < .38:
        return None
    if 6 <= height < 15 and needs_decimal and len(recognized) >= 2:
        # Speed is always rendered with exactly one decimal place. At the
        # distant-TV scale its one-pixel dot can disappear or merge with the
        # net post, so its detected position is less reliable than the fixed
        # format. A value such as 13.1 must never become 1.31.
        decimal_x = (
            recognized[-2][0] + recognized[-1][0]
        ) / 2
    if height < 15 and needs_decimal and decimal_x is not None:
        decimal_index = next((
            index for index, (center_x, _) in enumerate(recognized)
            if center_x > decimal_x
        ), len(recognized))
        # Compression can fragment one final speed digit into two core
        # components (the sample's 10.6 otherwise becomes 10.71). Rejoin the
        # wider antialiased glyph to preserve the HUD's one-decimal format.
        if len(recognized) - decimal_index > 1:
            right_start = math.ceil(decimal_x + 1)
            right_side = white[:, right_start:]
            ys, xs = np.nonzero(right_side)
            if len(xs):
                glyph = right_side[
                    ys.min():ys.max() + 1,
                    xs.min():xs.max() + 1,
                ]
                digit, score = classify_digit(glyph)
                if score >= .38:
                    recognized = recognized[:decimal_index] + [
                        (right_start + round(float(xs.mean())), digit)
                    ]
                    confidence = min(confidence, score)
    text = ""
    for center_x, digit in recognized:
        if needs_decimal and decimal_x is not None and decimal_x < center_x and "." not in text:
            text += "."
        text += digit
    if needs_decimal and "." not in text:
        return None
    try:
        return float(text) if needs_decimal else int(text)
    except ValueError:
        return None


def read_spin_direction(
    frame: np.ndarray, bounds: Tuple[int, int, int, int],
) -> Optional[Dict[str, Any]]:
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    roi = frame[
        round(y0 + 2.3 * height):round(y0 + 7.0 * height),
        round(x0 + .74 * width):round(x0 + 1.08 * width),
    ]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 90, 100), (145, 255, 255))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == component)
    if len(xs) < 8:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axis = axes[0]
    projections = centered @ axis
    span = float(projections.max() - projections.min())
    if span < 3:
        return None
    lower = points[projections <= projections.min() + span * .3]
    upper = points[projections >= projections.max() - span * .3]
    # The triangular arrowhead contains more blue pixels than the shaft end.
    if len(lower) > len(upper):
        axis = -axis
    dx, image_dy = map(float, axis)
    angle = (math.degrees(math.atan2(-image_dy, dx)) + 360) % 360
    labels_by_octant = ("right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right")
    label = labels_by_octant[round(angle / 45) % 8]
    return {
        "x": round(dx, 3),
        "y": round(-image_dy, 3),
        "angle_degrees": round(angle, 1),
        "label": label,
    }


def read_telemetry(
    frame: np.ndarray,
    frame_number: int,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[TelemetryReading]:
    bounds = bounds or telemetry_title_bounds(frame)
    if bounds is None:
        return None
    speed, spin, direction = read_telemetry_components(frame, bounds)
    if speed is None or spin is None or direction is None:
        return None
    return TelemetryReading(frame_number, speed, spin, direction)


def read_telemetry_components(
    frame: np.ndarray,
    bounds: Tuple[int, int, int, int],
) -> Tuple[
    Optional[float],
    Optional[int],
    Optional[Dict[str, Any]],
]:
    """Read and validate each HUD field without requiring one perfect frame."""
    raw_speed = read_hud_number(frame, bounds, "speed")
    raw_spin = read_hud_number(frame, bounds, "spin")
    direction = read_spin_direction(frame, bounds)
    # At low capture resolutions, unit text or compression artifacts can be
    # mistaken for another digit (for example, "51 rev/s" becoming 517).
    # Reject readings outside Eleven's displayed range instead of attaching a
    # confidently repeated but physically bogus value to every attempt. Do
    # this per field so a bad spin read does not throw away a good speed read.
    speed = (
        float(raw_speed)
        if raw_speed is not None and 0 < raw_speed < 100
        else None
    )
    spin = (
        int(raw_spin)
        if raw_spin is not None
        and 0 <= raw_spin <= MAX_SPIN_REVOLUTIONS_PER_SECOND
        else None
    )
    return speed, spin, direction


class TelemetryReader:
    """Debounce repeated HUD OCR into timestamped screen state changes."""

    def __init__(
        self,
        stable_samples: int = 3,
        evidence_window: int = 6,
    ) -> None:
        self.stable_samples = stable_samples
        self.evidence_window = evidence_window
        self.candidate: Optional[TelemetryReading] = None
        self.candidate_count = 0
        self.latest: Optional[TelemetryReading] = None
        self.bounds: Optional[Tuple[int, int, int, int]] = None
        self.component_samples: deque[
            Tuple[
                Optional[float],
                Optional[int],
                Optional[Dict[str, Any]],
            ]
        ] = deque(maxlen=evidence_window)

    @staticmethod
    def same_values(left: TelemetryReading, right: TelemetryReading) -> bool:
        # Five-pixel-tall HUD digits occasionally fluctuate by one final
        # speed/spin unit during a screen transition. Treat that as the same
        # displayed state so it cannot displace the actual machine reading.
        return (
            abs(left.speed_mps - right.speed_mps) <= .11
            and abs(
                left.spin_revolutions_per_second
                - right.spin_revolutions_per_second
            ) <= 1
            and left.spin_direction["label"] == right.spin_direction["label"]
        )

    def update(self, frame: np.ndarray, frame_number: int) -> Optional[TelemetryReading]:
        if self.bounds is None:
            self.bounds = telemetry_title_bounds(frame)
        if self.bounds is None:
            return None
        self.component_samples.append(
            read_telemetry_components(frame, self.bounds)
        )
        reading = self.consensus_reading(frame_number)
        if reading is None:
            return None
        if self.candidate is not None and self.same_values(reading, self.candidate):
            self.candidate_count += 1
        else:
            self.candidate = reading
            self.candidate_count = 1
        if self.candidate_count < self.stable_samples:
            return None
        if self.latest is not None and self.same_values(reading, self.latest):
            return None
        self.latest = reading
        return reading

    def consensus_reading(
        self, frame_number: int,
    ) -> Optional[TelemetryReading]:
        """Combine repeated field reads from adjacent frames of one HUD state."""
        # Two matching observations make a field usable; the separate
        # candidate debounce below keeps the assembled state from publishing
        # during a screen transition.
        minimum_evidence = 2
        numeric_pairs = Counter(
            (speed, spin) for speed, spin, _ in self.component_samples
            if speed is not None and spin is not None
        )
        directions = Counter(
            direction["label"] for _, _, direction in self.component_samples
            if direction is not None
        )
        if not numeric_pairs or not directions:
            return None
        (speed, spin), numeric_count = numeric_pairs.most_common(1)[0]
        direction_label, direction_count = directions.most_common(1)[0]
        if min(numeric_count, direction_count) < minimum_evidence:
            return None
        direction = next(
            direction for _, _, direction in reversed(self.component_samples)
            if direction is not None
            and direction["label"] == direction_label
        )
        return TelemetryReading(
            frame_number,
            speed,
            spin,
            direction,
        )

def shadow_contact_score(
    hsv: np.ndarray, center: Point, settings: DetectorSettings = DetectorSettings(),
) -> float:
    """Local calibrated-table darkening directly below a ball candidate."""
    x, y = map(round, center)
    height, width = hsv.shape[:2]
    local = hsv[max(0, y + 5):min(height, y + 28), max(0, x - 18):min(width, x + 19)]
    surrounding = hsv[max(0, y - 35):min(height, y + 36), max(0, x - 35):min(width, x + 36)]
    def table_values(region: np.ndarray) -> np.ndarray:
        if region.size == 0:
            return np.array([])
        mask = (
            (hue_distance(region[:, :, 0], settings.table_hue_center)
             <= settings.table_hue_tolerance)
            & (region[:, :, 1] >= settings.table_min_saturation)
        )
        return region[:, :, 2][mask]
    dark, baseline = table_values(local), table_values(surrounding)
    if len(dark) < 8 or len(baseline) < 20:
        return 0.0
    return max(0.0, float(np.median(baseline) - np.percentile(dark, 5)))


def candidates_for_frame(
    frame: np.ndarray,
    previous_gray: Optional[np.ndarray],
    tracking_polygon: np.ndarray,
    settings: DetectorSettings = DetectorSettings(),
    diagnostics: Optional[DetectorDiagnostics] = None,
) -> Tuple[np.ndarray, List[Candidate]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Difference rejects static markings/text/net edges without retaining a
    # background frame. Legacy calibrations use a bright white range; automatic
    # color calibration supplies a circular saturated-ball hue instead.
    if settings.ball_hue_center is None:
        bright = cv2.inRange(
            hsv, settings.bright_ball_lower, settings.bright_ball_upper,
        )
    else:
        bright = np.uint8(
            (hue_distance(hsv[:, :, 0], settings.ball_hue_center)
             <= settings.ball_hue_tolerance)
            & (hsv[:, :, 1] >= settings.ball_min_saturation)
            & (hsv[:, :, 2] >= settings.ball_min_value)
        ) * 255
    if previous_gray is None:
        return gray, []
    moving = cv2.threshold(cv2.absdiff(gray, previous_gray), settings.motion_threshold, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_and(bright, moving)
    # Preserve compact two-pixel distant balls, but discard single-pixel
    # codec shimmer before it can seed a track.
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    choices = []
    polygon_y = tracking_polygon[:, 1]
    corridor_top, corridor_bottom = float(polygon_y.min()), float(polygon_y.max())
    corridor_height = max(1.0, corridor_bottom - corridor_top)
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        center = tuple(map(float, centers[i]))
        if not point_in_polygon(center, tracking_polygon):
            if diagnostics is not None:
                diagnostics.candidate(center, "rejected", "outside tracking region")
            continue

        # Perspective makes the same ball occupy more pixels near the bottom
        # of a fixed spectator view. Interpolate diameter (rather than area)
        # through the calibrated flight corridor, then square it for a smooth
        # area limit. This keeps small distant balls while rejecting large
        # moving room highlights high in the image.
        depth = min(1.0, max(0.0, (center[1] - corridor_top) / corridor_height))
        far_diameter_ratio = math.sqrt(settings.far_max_candidate_area_ratio)
        diameter_ratio = far_diameter_ratio + (1.0 - far_diameter_ratio) * depth
        maximum_area = max(
            settings.min_candidate_area,
            round(settings.max_candidate_area * diameter_ratio ** 2),
        )
        minimum_area = round(
            settings.min_candidate_area
            + (settings.near_min_candidate_area - settings.min_candidate_area) * depth
        )
        if not minimum_area <= area <= maximum_area:
            if diagnostics is not None:
                diagnostics.candidate(
                    center, "rejected",
                    f"area {area} outside {minimum_area}-{maximum_area}",
                )
            continue

        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect_ratio = max(width, height) / max(1, min(width, height))
        if aspect_ratio > settings.max_candidate_aspect_ratio:
            if diagnostics is not None:
                diagnostics.candidate(center, "rejected", f"aspect ratio {aspect_ratio:.2f}")
            continue
        compactness = area / max(1, width * height)
        if compactness < settings.min_candidate_compactness:
            if diagnostics is not None:
                diagnostics.candidate(center, "rejected", f"compactness {compactness:.2f}")
            continue

        component_pixels = hsv[labels == i]
        saturation = float(np.median(component_pixels[:, 1]))
        brightness = float(np.median(component_pixels[:, 2]))
        if brightness < settings.min_candidate_brightness:
            if diagnostics is not None:
                diagnostics.candidate(center, "rejected", f"brightness {brightness:.0f}")
            continue
        if saturation > settings.max_candidate_saturation:
            if diagnostics is not None:
                diagnostics.candidate(center, "rejected", f"saturation {saturation:.0f}")
            continue
        if diagnostics is not None:
            diagnostics.candidate(center, "raw")
        choices.append((area, center, shadow_contact_score(hsv, center, settings)))
    # At track start, prefer the compact moving ball over single-pixel codec
    # shimmer; once a track exists, motion prediction chooses continuity.
    choices.sort(key=lambda item: item[0], reverse=True)
    return gray, [(center[0], center[1], score) for _, center, score in choices]


def map_log_coordinate(
    homography: np.ndarray, image_point: Point, surface_y: float
) -> Tuple[float, float, float]:
    mapped = cv2.perspectiveTransform(np.float32([[image_point]]), homography)[0][0]
    return round(float(mapped[0]), 4), round(float(surface_y), 4), round(float(mapped[1]), 4)


def draw_overlay(
    frame: np.ndarray,
    table: np.ndarray,
    net_line: np.ndarray,
    track: Sequence[TrackPoint],
    events: Sequence[BounceEvent],
    homography: np.ndarray,
    surface_y: float,
    diagnostics: Optional[DetectorDiagnostics] = None,
    frame_number: Optional[int] = None,
) -> np.ndarray:
    view = frame.copy()
    poly = np.int32(table).reshape((-1, 1, 2))
    cv2.polylines(view, [poly], True, (0, 255, 255), 3)
    cv2.line(view, tuple(map(int, net_line[0])), tuple(map(int, net_line[1])), (255, 0, 255), 3)
    # Calibration grid: x is across the table width; z is player(-) to
    # opponent(+). This makes a bad corner/axis calibration obvious before
    # any bounce coordinates are trusted.
    inverse_homography = np.linalg.inv(homography)
    for z in (-1.37, -0.685, 0.0, 0.685, 1.37):
        line = np.float32([[[-0.7625, z]], [[0.7625, z]]])
        projected = cv2.perspectiveTransform(line, inverse_homography).reshape(-1, 2)
        cv2.line(view, tuple(map(int, projected[0])), tuple(map(int, projected[1])), (80, 160, 255), 1)
    for x in (-0.7625, -0.38125, 0.0, 0.38125, 0.7625):
        line = np.float32([[[x, -1.37]], [[x, 1.37]]])
        projected = cv2.perspectiveTransform(line, inverse_homography).reshape(-1, 2)
        cv2.line(view, tuple(map(int, projected[0])), tuple(map(int, projected[1])), (80, 160, 255), 1)
    center = cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), inverse_homography)[0][0]
    cv2.drawMarker(view, tuple(map(int, center)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(view, "log-space grid; red = (0,0)", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    for point in track:
        x, y = point[1], point[2]
        cv2.circle(view, (round(x), round(y)), 3, (0, 255, 255), -1)
    if diagnostics is not None:
        colors = {
            "rejected": (0, 128, 255),
            "launcher": (255, 80, 40),
            "association": (255, 180, 40),
            "return": (40, 220, 40),
            "contact_candidate": (255, 120, 255),
            "rejected_contact_candidate": (0, 128, 255),
            "confirmed_bounce": (0, 0, 255),
        }
        for candidate in diagnostics.candidates:
            center = tuple(map(round, candidate.center))
            if candidate.kind == "raw":
                cv2.circle(view, center, 2, (190, 190, 190), 1)
                continue
            cv2.drawMarker(view, center, colors["rejected"], cv2.MARKER_TILTED_CROSS, 8, 1)
        for path in diagnostics.unconfirmed_tracks:
            points = np.int32([(point[1], point[2]) for point in path])
            if len(points) >= 2:
                cv2.polylines(view, [points], False, (255, 255, 0), 1)
        current_frame = frame_number if frame_number is not None else 0
        for completed in diagnostics.visible_completed_tracks(current_frame)[-8:]:
            color = colors[completed.kind]
            points = np.int32([(point[1], point[2]) for point in completed.points])
            if len(points) >= 2:
                cv2.polylines(view, [points], False, color, 2)
            if len(points):
                label = completed.kind
                if completed.kind in (
                    "contact_candidate", "rejected_contact_candidate",
                ) and completed.reason:
                    record = json.loads(completed.reason)
                    label = (
                        f"{'rejected ' if not record['accepted'] else ''}contact "
                        f"{record['table_side']} "
                        f"{record['signal_type']} f={record['frame_number']}"
                    )
                elif completed.reason:
                    label += f": {completed.reason}"
                endpoint = completed.points[-1]
                cv2.putText(
                    view, label, (round(endpoint[1]) + 6, round(endpoint[2]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1,
                )
    for event in events:
        if frame_number is None:
            visible = event.frame_number == event.draw_frame
        else:
            visible = event.draw_frame <= frame_number <= event.draw_frame + 30
        if not visible:
            continue
        p = event.pixel
        cv2.drawMarker(view, (round(p[0]), round(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
        label = f"{event.outcome} {event.confidence:.2f}"
        if event.posx is not None:
            label += f" x={event.posx:.2f} z={event.posz:.2f}"
        cv2.putText(view, label, (round(p[0]) + 12, round(p[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    if diagnostics is not None:
        legend = (
            ("raw candidate", (190, 190, 190)),
            ("rejected", (0, 128, 255)),
            ("unconfirmed", (255, 255, 0)),
            ("launcher", (255, 80, 40)),
            ("owned return", (255, 180, 40)),
            ("return", (40, 220, 40)),
            ("contact candidate", (255, 120, 255)),
            ("rejected contact", (0, 128, 255)),
            ("confirmed bounce", (0, 0, 255)),
        )
        cv2.rectangle(view, (8, 39), (180, 169), (20, 20, 20), -1)
        for index, (label, color) in enumerate(legend):
            cv2.putText(
                view, label, (16, 54 + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1,
            )
    return view


class AttemptClassifier:
    """Turn completed ball tracks into one result for each launcher cycle."""

    def __init__(
        self,
        fps: float,
        calibration: Calibration,
        table: np.ndarray,
        net_line: np.ndarray,
        occlusion: np.ndarray,
        homography: np.ndarray,
        video_width: int,
        video_height: int,
        scale: float,
        settings: DetectorSettings,
        on_event: Optional[Callable[[BounceEvent], None]] = None,
        on_attempt_finished: Optional[Callable[[Optional[int]], None]] = None,
        on_confirmed_hit: Optional[Callable[[BounceEvent], None]] = None,
        on_confirmed_non_hit: Optional[Callable[[BounceEvent], None]] = None,
        on_track_diagnostic: Optional[Callable[[TrackDiagnostic, int], None]] = None,
        on_attempt_started: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.fps = fps
        self.calibration = calibration
        self.table = table
        self.net_line = net_line
        self.occlusion = occlusion
        self.homography = homography
        self.scale = scale
        self.settings = settings
        self.on_event = on_event
        self.on_attempt_finished = on_attempt_finished
        self.on_confirmed_hit = on_confirmed_hit
        self.on_confirmed_non_hit = on_confirmed_non_hit
        self.on_track_diagnostic = on_track_diagnostic
        self.on_attempt_started = on_attempt_started
        self.events: List[BounceEvent] = []
        self.emitted: Set[Tuple[int, int]] = set()
        self.confirmed_hit_notifications: Set[Tuple[int, int, int]] = set()
        self.started_launcher_tracks: Set[Tuple[int, int, int]] = set()
        self.reported_non_hit_tracks: Set[Tuple[int, int, int]] = set()
        self.active_attempts: List[Attempt] = []
        self.track_owners: Dict[Tuple[int, int, int], int] = {}
        self.launch_frames: List[int] = []
        self.latest_telemetry: Optional[TelemetryReading] = None
        self.telemetry_history: List[TelemetryReading] = []
        self.launcher_tracks_seen = 0
        configured_launcher_region = calibration.get("launcher_region")
        self.launcher_region = configured_launcher_region or [
            video_width * .58, 0, video_width, video_height,
        ]
        # In a wide view, only launches beginning above the opponent-side rail
        # and away from the outermost frame edge are strong enough to justify
        # a standalone no-bounce result. Other left-moving tracks may still
        # hold a subsequently verified bounce, but cannot emit an unknown.
        table_top = min(float(point[1]) for point in table) / scale
        table_bottom = max(float(point[1]) for point in table) / scale
        wide_view = table_bottom - table_top < video_height * .4
        self.reportable_launcher_region = self.launcher_region
        if wide_view:
            launcher_bottom = min(video_height, table_top + video_height * .05)
            self.reportable_launcher_region = configured_launcher_region or [
                video_width * .58, 0, video_width * .95, launcher_bottom,
            ]
        self.return_region = calibration.get(
            "return_region", [0, 0, video_width * .28, video_height]
        )
        launcher_center_x = (self.launcher_region[0] + self.launcher_region[2]) / 2
        return_center_x = (self.return_region[0] + self.return_region[2]) / 2
        legacy_launch_direction = 1 if return_center_x > launcher_center_x else -1
        self.return_direction = (
            settings.return_direction_x * -legacy_launch_direction,
            settings.return_direction_y * -legacy_launch_direction,
        ) if calibration.get("camera_geometry") != "elevated_end_view" else (
            settings.return_direction_x, settings.return_direction_y,
        )
        self.launch_vector = (-self.return_direction[0], -self.return_direction[1])
        self.warmup_launcher_tracks = calibration.get("warmup_launcher_tracks", 0)

    @staticmethod
    def projected_travel(
        beginning: TrackPoint, end: TrackPoint, direction: Point,
    ) -> float:
        return (
            (end[1] - beginning[1]) * direction[0]
            + (end[2] - beginning[2]) * direction[1]
        )

    @property
    def active_attempt(self) -> Optional[Attempt]:
        """Newest open attempt, retained for callers that inspect live state."""
        return self.active_attempts[-1] if self.active_attempts else None

    def attempt_by_frame(self, frame: int) -> Optional[Attempt]:
        return next((item for item in self.active_attempts if item.frame == frame), None)

    def observed_launch_period(self) -> float:
        gaps = [
            later - earlier
            for earlier, later in zip(self.launch_frames, self.launch_frames[1:])
            if later > earlier
        ]
        return float(sorted(gaps)[len(gaps) // 2]) if gaps else self.fps * 2.2

    def attempt_lifetime(self) -> int:
        """Bound overlap from observed machine cadence, with room for occlusion."""
        return max(round(self.fps * 2.2), round(self.observed_launch_period() * 2.5))

    def diagnose_track(
        self, path: Track, kind: str, draw_frame: int, reason: str = "",
    ) -> None:
        if self.on_track_diagnostic is not None:
            self.on_track_diagnostic(TrackDiagnostic(path, kind, reason), draw_frame)

    @staticmethod
    def physical_contact_key(hit: TrackPoint) -> Tuple[int, int, int]:
        return hit[0], round(hit[1]), round(hit[2])

    def profile_table_side(self, pixel: Point) -> str:
        """Classify a profile-view turn by which side of the net contains it.

        A true side view collapses table width, so a homography cannot recover
        a landing coordinate.  Depth along the table remains directly visible,
        however, and the reviewed opponent control tells us which signed side
        of the net is the successful-return side.
        """
        controls = {
            item["name"]: item["image"]
            for item in self.calibration.get("control_points", [])
        }
        opponent = controls.get("x0_opponent_edge")
        if opponent is None:
            return "occluded"
        opponent_pixel = (
            float(opponent[0]) * self.scale,
            float(opponent[1]) * self.scale,
        )
        opponent_sign = signed_distance_to_line(opponent_pixel, self.net_line)
        pixel_sign = signed_distance_to_line(pixel, self.net_line)
        if abs(pixel_sign) <= 1e-6 or abs(opponent_sign) <= 1e-6:
            return "occluded"
        return "opponent" if pixel_sign * opponent_sign > 0 else "player"

    def contact_candidate(
        self,
        path: Track,
        hit: TrackPoint,
        approach: Track,
        departure: Track,
    ) -> ContactCandidate:
        pixel = (hit[1], hit[2])
        if self.calibration.get("camera_geometry") == "profile_side_view":
            table_side = self.profile_table_side(pixel)
            continuity = min(1.0, len(approach + departure) / 5)
            return ContactCandidate(
                frame_number=hit[0],
                pixel=pixel,
                log_position=None,
                table_side=table_side,
                signal_type=bounce_signal(hit, approach, departure),
                strength=bounce_strength(hit, approach, departure),
                confidence=round(
                    (0.92 if table_side == "opponent" else 0.72) * continuity,
                    2,
                ),
                source_track_key=self.track_key(path),
                approach=tuple(approach),
                departure=tuple(departure),
            )
        in_occlusion = (
            len(self.occlusion) > 2 and point_in_polygon(pixel, self.occlusion)
        )
        posx, posy, posz = map_log_coordinate(
            self.homography, pixel, self.calibration["table_surface_y"],
        )
        far = posz > 0.03
        continuity = min(1.0, len(approach + departure) / 6)
        confidence = round(
            (0.82 if far else 0.72)
            * continuity
            * (0.45 if in_occlusion else 1.0),
            2,
        )
        return ContactCandidate(
            frame_number=hit[0],
            pixel=pixel,
            log_position=None if in_occlusion else (posx, posy, posz),
            table_side=(
                "occluded" if in_occlusion else ("opponent" if far else "player")
            ),
            signal_type=bounce_signal(hit, approach, departure),
            strength=bounce_strength(hit, approach, departure),
            confidence=confidence,
            source_track_key=self.track_key(path),
            approach=tuple(approach),
            departure=tuple(departure),
        )

    def record_contact_candidate(
        self, candidate: ContactCandidate, path: Track, draw_frame: int,
        attempt: Optional[Attempt] = None,
    ) -> ContactCandidate:
        """Attach one physical contact once, independent of tracker ownership."""
        attempt = attempt or self.active_attempt
        if attempt is None:
            return candidate
        key = (
            candidate.frame_number,
            round(candidate.pixel[0]),
            round(candidate.pixel[1]),
        )
        for existing in attempt.contact_candidates:
            existing_key = (
                existing.frame_number,
                round(existing.pixel[0]),
                round(existing.pixel[1]),
            )
            if existing_key == key:
                return existing
        attempt.contact_candidates.append(candidate)
        attempt.contact_candidates.sort(
            key=lambda item: (item.frame_number, item.pixel)
        )
        attempt.contact_keys.add(key)
        attempt.last_evidence_frame = candidate.frame_number
        self.diagnose_track(
            path, "contact_candidate", draw_frame,
            json.dumps({
                **candidate.to_record(),
                "attempt_frame_number": attempt.frame,
            }, separators=(",", ":")),
        )
        return candidate

    def notify_confirmed_hit(self, event: BounceEvent) -> None:
        """Publish a direct or delayed hit to the live ledger exactly once."""
        if (
            not event.hit_table
            or event.outcome != "far_table"
            or self.on_confirmed_hit is None
        ):
            return
        key = (
            event.frame_number,
            round(event.pixel[0]),
            round(event.pixel[1]),
        )
        if key in self.confirmed_hit_notifications:
            return
        self.confirmed_hit_notifications.add(key)
        self.on_confirmed_hit(event)

    def emit(self, event: BounceEvent) -> None:
        # A contact can be discovered while a newer attempt is open. It was
        # deliberately withheld from the live ledger until launch-order
        # settlement; publish it as a hit before cadence fills that slot with
        # an immutable miss.
        self.notify_confirmed_hit(event)
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def observe_telemetry(self, reading: TelemetryReading) -> None:
        self.latest_telemetry = reading
        self.telemetry_history.append(reading)
        if self.active_attempt is None:
            return
        machine = self.active_attempt.machine_telemetry
        if machine is None or not TelemetryReader.same_values(reading, machine):
            self.active_attempt.telemetry_after_launch.append(reading)

    def telemetry_near(self, frame: int) -> Optional[TelemetryReading]:
        if not self.telemetry_history:
            return None
        reading = min(
            self.telemetry_history,
            key=lambda item: abs(item.frame_number - frame),
        )
        return reading if abs(reading.frame_number - frame) <= self.fps * .4 else None

    def telemetry_pair_for_attempt(
        self, attempt: Attempt, frame: int,
    ) -> Tuple[Optional[TelemetryReading], Optional[TelemetryReading]]:
        """Return the post-hit screen update and this attempt's delivery."""
        machine = attempt.machine_telemetry
        if machine is None:
            return None, None
        post_launch = [
            item for item in attempt.telemetry_after_launch
            if item.frame_number <= frame
            and frame - item.frame_number <= self.fps * .6
            and not TelemetryReader.same_values(item, machine)
        ]
        return (post_launch[-1] if post_launch else None), machine

    def launcher_rejection_reason(self, path: Track) -> Optional[str]:
        """Explain why a completed path cannot establish a machine launch."""
        start = (path[0][1], path[0][2])
        if not point_in_rectangle(start, self.launcher_region, self.scale):
            return "did not begin near launcher"
        if len(path) < self.settings.min_launch_track_points:
            return f"launch too short ({len(path)}/{self.settings.min_launch_track_points})"

        directed_steps = [
            self.projected_travel(beginning, end, self.launch_vector)
            for beginning, end in zip(path, path[1:])
        ]
        directed_distance = self.projected_travel(
            path[0], path[-1], self.launch_vector,
        )
        if directed_distance < self.settings.launch_min_horizontal_distance:
            return "insufficient travel toward player"

        horizontal_travel = sum(abs(step) for step in directed_steps)
        directional_ratio = directed_distance / max(horizontal_travel, 1e-6)
        if directional_ratio < self.settings.launch_min_directional_ratio:
            return "inconsistent travel toward player"
        return None

    def is_launcher_track(self, path: Track) -> bool:
        return self.launcher_rejection_reason(path) is None

    def is_return_track(
        self, path: Track, attempt: Optional[Attempt] = None,
    ) -> bool:
        return self.return_rejection_reason(path, attempt) is None

    def return_candidate_segment(self, path: Track) -> Optional[Track]:
        """Discard a false prefix before a clean player-to-table return.

        Bright static objects can own a tracker hypothesis until the moving
        ball crosses them. The resulting path still contains an unambiguous
        return, but its first point is on the wrong side of the frame. Locate
        the first in-region point that has enough subsequent camera-relative
        travel toward the opponent instead of requiring the track to have
        been clean from birth.
        """
        for index, point in enumerate(path):
            if (
                point_in_rectangle((point[1], point[2]), self.return_region, self.scale)
                and self.projected_travel(point, path[-1], self.return_direction)
                >= self.settings.return_min_horizontal_distance
            ):
                return path[index:]
        return None

    def perspective_round_trip_return(self, path: Track) -> Optional[Track]:
        """Split an end-view machine-delivery track at the player turnaround.

        In an elevated view the colored ball can remain continuously visible
        through the machine delivery, racket contact, and outbound return.  A
        prefix of that path has already established the launch; ignoring the
        same tracker identity afterwards discards the clearest contact
        evidence.  Locate the deepest playerward point on the calibrated
        player-to-opponent axis and retain only a fully qualified return
        suffix.
        """
        if self.calibration.get("camera_geometry") != "elevated_end_view":
            return None
        if len(path) < self.settings.min_launch_track_points + 3:
            return None
        start = (path[0][1], path[0][2])
        if not point_in_rectangle(start, self.launcher_region, self.scale):
            return None
        projected = [
            point[1] * self.return_direction[0]
            + point[2] * self.return_direction[1]
            for point in path
        ]
        turn_index = min(range(len(path)), key=projected.__getitem__)
        if turn_index < self.settings.min_launch_track_points - 1:
            return None
        prefix = path[:turn_index + 1]
        returned = path[turn_index:]
        if self.launcher_rejection_reason(prefix) is not None:
            return None
        if len(returned) < self.settings.min_track_observations:
            return None
        progress = self.projected_travel(
            returned[0], returned[-1], self.return_direction,
        )
        elapsed = returned[-1][0] - returned[0][0]
        if (
            progress < self.settings.return_min_horizontal_distance
            or elapsed <= 0
            or progress / elapsed < self.settings.return_min_horizontal_speed
        ):
            return None
        return returned

    def perspective_round_trip_bounce(
        self, returned: Track, allow_terminal_shadow: bool,
    ) -> Optional[Bounce]:
        """Choose a contact beyond the net-adjacent perspective ambiguity.

        Immediately past the net, the ball overlaps its ordinary table shadow
        even while still in flight.  Require a round-trip-only contact to be
        at least one fifth of the opponent half's regulation depth, and prefer
        the deepest bounded signal when several shadow peaks describe the
        same outbound path.
        """
        minimum_depth = 1.37 * .2
        candidates = []
        for bounce in find_bounces(
            returned, self.table, self.net_line, self.settings,
            allow_terminal_shadow=allow_terminal_shadow,
        ):
            hit = bounce[0]
            _, _, depth = map_log_coordinate(
                self.homography, (hit[1], hit[2]),
                self.calibration["table_surface_y"],
            )
            if depth >= minimum_depth:
                candidates.append((depth, bounce))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def return_rejection_reason(
        self, path: Track, attempt: Optional[Attempt] = None,
    ) -> Optional[str]:
        """Explain why a path cannot be the active launch's player return."""
        in_return_region = any(
            point_in_rectangle((point[1], point[2]), self.return_region, self.scale)
            for point in path
        )
        if not in_return_region:
            return "did not begin near player"

        returned = self.return_candidate_segment(path)
        if returned is None:
            return "insufficient travel toward opponent"
        elapsed = returned[-1][0] - returned[0][0]
        directed_distance = self.projected_travel(
            returned[0], returned[-1], self.return_direction,
        )
        if (
            elapsed <= 0
            or directed_distance / elapsed
            < self.settings.return_min_horizontal_speed
        ):
            return "return moved too slowly"
        if attempt is not None:
            post_launch_observations = sum(
                point[0] > attempt.frame for point in returned
            )
            if post_launch_observations < self.settings.min_track_observations:
                return (
                    "too few return observations after launch "
                    f"({post_launch_observations}/"
                    f"{self.settings.min_track_observations})"
                )
        return None

    def reconnected_return(
        self, path: Track, attempt: Attempt,
    ) -> Optional[Track]:
        """Join a return after a short player/avatar or net occlusion.

        Completed fragments remain owned by the current attempt. A later
        fragment may extend one only when time and camera-relative motion are
        consistent with the same outbound ball. Slow rolling balls and the
        next machine launch therefore cannot take over the return identity.
        """
        if not path or path[0][0] <= attempt.frame:
            return None
        overlapping = self.profile_overlapping_return(path, attempt)
        if overlapping is not None:
            return overlapping
        candidates: List[Tuple[int, float, Track]] = []
        for previous in attempt.returns:
            gap = path[0][0] - previous[-1][0]
            if not 0 < gap <= self.settings.return_reconnect_max_gap:
                continue
            forward_gap = self.projected_travel(
                previous[-1], path[0], self.return_direction,
            )
            if not (
                -self.settings.return_reconnect_backtrack_tolerance
                <= forward_gap
                <= self.settings.return_reconnect_max_forward_distance
            ):
                continue
            fragment_elapsed = path[-1][0] - path[0][0]
            fragment_progress = self.projected_travel(
                path[0], path[-1], self.return_direction,
            )
            if (
                fragment_elapsed <= 0
                or fragment_progress / fragment_elapsed
                < self.settings.return_min_horizontal_speed
            ):
                continue
            combined = previous + path
            if self.return_rejection_reason(combined, attempt) is None:
                candidates.append((gap, -forward_gap, combined))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:2])[2]

    def profile_overlapping_return(
        self, path: Track, attempt: Attempt,
    ) -> Optional[Track]:
        """Join two identities that overlap at a profile-view table contact.

        The tracker can retain the descending identity for a few frames while
        simultaneously starting a new identity on the rising ball.  Require
        the identities to observe nearly the same point in an exact shared
        frame, to keep moving toward the opponent, and for the new identity to
        reach that side's table surface immediately afterwards. Replacing the
        old overlap with the new identity leaves one chronological path for
        the ordinary vertical-turn detector.
        """
        if (
            self.calibration.get("camera_geometry") != "profile_side_view"
            or not path
            or path[0][0] <= attempt.frame
        ):
            return None
        fragment_elapsed = path[-1][0] - path[0][0]
        fragment_progress = self.projected_travel(
            path[0], path[-1], self.return_direction,
        )
        if (
            fragment_elapsed <= 0
            or fragment_progress / fragment_elapsed
            < self.settings.return_min_horizontal_speed
        ):
            return None

        candidates: List[Tuple[float, int, Track]] = []
        path_by_frame = {point[0]: point for point in path}
        for previous in attempt.returns:
            overlap = previous[-1][0] - path[0][0]
            if not 0 <= overlap <= self.settings.profile_overlap_max_frames:
                continue
            previous_tail = previous[-min(4, len(previous)):]
            previous_elapsed = previous_tail[-1][0] - previous_tail[0][0]
            previous_progress = self.projected_travel(
                previous_tail[0], previous_tail[-1], self.return_direction,
            )
            if (
                previous_elapsed <= 0
                or previous_progress / previous_elapsed
                < self.settings.return_min_horizontal_speed
            ):
                continue
            shared = [
                (old, path_by_frame[old[0]])
                for old in previous
                if old[0] in path_by_frame
            ]
            if not shared:
                continue
            match_distance = min(
                math.dist(old[1:3], new[1:3])
                for old, new in shared
            )
            if match_distance > self.settings.profile_overlap_match_distance:
                continue
            reaches_opponent_table = any(
                self.profile_table_side((point[1], point[2])) == "opponent"
                and point_near_polygon(
                    (point[1], point[2]), self.table,
                    self.settings.table_contact_margin,
                )
                for point in path[:8]
            )
            if not reaches_opponent_table:
                continue
            combined = [
                point for point in previous if point[0] < path[0][0]
            ] + path
            if self.return_rejection_reason(combined, attempt) is None:
                candidates.append((match_distance, -len(combined), combined))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:2])[2]

    def return_segment(
        self, path: Track, attempt: Optional[Attempt] = None,
    ) -> Optional[Track]:
        if self.return_rejection_reason(path, attempt) is not None:
            return None
        return self.return_candidate_segment(path)

    def is_reportable_launcher_track(self, path: Track) -> bool:
        start = (path[0][1], path[0][2])
        return self.is_launcher_track(path) and point_in_rectangle(
            start, self.reportable_launcher_region, self.scale,
        )

    @staticmethod
    def track_key(path: Track) -> Tuple[int, int, int]:
        return path[0][0], round(path[0][1]), round(path[0][2])

    def association_score(self, path: Track, attempt: Attempt) -> Optional[Tuple[float, ...]]:
        """Score one exclusive return owner; tuple ordering is deterministic."""
        if not path or path[-1][0] <= attempt.frame or attempt.state in {"settled", "expired"}:
            return None
        key = self.track_key(path)
        if self.track_owners.get(key) == attempt.frame:
            return (1000.0, 0.0, 0.0, float(-attempt.frame))
        returned = self.return_segment(path, attempt)
        reconnected = self.reconnected_return(path, attempt)
        if returned is None and reconnected is None:
            return None
        continuity = 0.0
        gap = path[0][0] - attempt.frame
        if reconnected is not None:
            previous = max(
                attempt.returns,
                key=lambda item: item[-1][0] if item[-1][0] < path[0][0] else -1,
            )
            continuity = 500.0 - max(0, path[0][0] - previous[-1][0])
        # Prefer an explicit reconnection, then the launch immediately before
        # a new fragment. The final key makes equal scores choose the older
        # launch, which preserves durable prior ownership.
        return (
            500.0 if reconnected is not None else 100.0,
            continuity,
            float(-gap),
            float(-attempt.frame),
        )

    def associate_return(self, path: Track, draw_frame: int) -> Tuple[Optional[Attempt], Optional[Track]]:
        key = self.track_key(path)
        owner_frame = self.track_owners.get(key)
        if owner_frame is not None:
            owner = self.attempt_by_frame(owner_frame)
            if owner is not None:
                returned = self.return_segment(path, owner) or self.reconnected_return(path, owner)
                if returned is not None:
                    return owner, returned
        scored = [
            (score, attempt)
            for attempt in self.active_attempts
            if (score := self.association_score(path, attempt)) is not None
        ]
        if not scored:
            return None, None
        score, owner = max(scored, key=lambda item: item[0])
        returned = self.reconnected_return(path, owner) or self.return_segment(path, owner)
        if returned is None:
            return None, None
        self.track_owners[key] = owner.frame
        owner.owned_track_keys.add(key)
        owner.last_evidence_frame = path[-1][0]
        self.diagnose_track(
            path, "association", draw_frame,
            json.dumps({
                "attempt_frame_number": owner.frame,
                "track_key": list(key),
                "score": list(score),
                "candidate_scores": [
                    {
                        "attempt_frame_number": candidate.frame,
                        "score": list(candidate_score),
                    }
                    for candidate_score, candidate in sorted(
                        scored, key=lambda item: item[1].frame,
                    )
                ],
            }, separators=(",", ":")),
        )
        return owner, returned

    def mark_attempt_state(self, attempt: Attempt) -> None:
        if attempt.bounces:
            attempt.state = "contact_pending"
        elif attempt.returns or attempt.contact_candidates:
            attempt.state = "return_seen"
        else:
            attempt.state = "launched"

    def has_credible_unresolved_evidence(self, attempt: Attempt) -> bool:
        if attempt.bounces:
            return False
        if any(
            candidate.signal_type == "net"
            for candidate in attempt.contact_candidates
        ):
            return True
        return any(
            point_in_polygon((path[-1][1], path[-1][2]), self.table)
            for path in attempt.returns
        )

    def drain_settled_attempts(
        self,
        draw_frame: int,
        settlement_frame: Optional[int] = None,
        notify: bool = True,
    ) -> None:
        """Publish only the settled launch-order prefix."""
        while self.active_attempts and self.active_attempts[0].state in {"settled", "expired"}:
            attempt = self.active_attempts.pop(0)
            self.finalize_attempt(attempt, draw_frame)
            for key in attempt.owned_track_keys:
                if self.track_owners.get(key) == attempt.frame:
                    self.track_owners.pop(key)
            if notify and self.on_attempt_finished is not None:
                self.on_attempt_finished(settlement_frame)

    def expire_attempts(self, draw_frame: int) -> None:
        lifetime = self.attempt_lifetime()
        for attempt in self.active_attempts:
            if draw_frame - attempt.frame > lifetime:
                attempt.state = "expired"
        while len(self.active_attempts) > 3:
            self.active_attempts[0].state = "expired"
            self.drain_settled_attempts(draw_frame)

    def start_attempt(self, path: Track, draw_frame: int) -> None:
        self.launcher_tracks_seen += 1
        if self.launcher_tracks_seen <= self.warmup_launcher_tracks:
            return
        launch_frame = path[0][0]
        self.launch_frames.append(launch_frame)
        for attempt in self.active_attempts:
            self.mark_attempt_state(attempt)
            # Net/return evidence may still acquire a delayed continuation.
            # Everything else is complete at the next credible launch.
            if not self.has_credible_unresolved_evidence(attempt):
                attempt.state = "settled"
        self.drain_settled_attempts(draw_frame, launch_frame)
        launch_key = self.track_key(path)
        attempt = Attempt(
            path[0][0], (path[0][1], path[0][2]),
            launch_track_key=launch_key,
            owned_track_keys={launch_key},
            last_evidence_frame=path[-1][0],
            report_no_bounce=self.is_reportable_launcher_track(path),
            machine_telemetry=self.telemetry_near(path[0][0]),
        )
        self.active_attempts.append(attempt)
        self.track_owners[launch_key] = attempt.frame
        self.expire_attempts(draw_frame)
        if self.on_attempt_started is not None:
            self.on_attempt_started(attempt.frame)

    def return_evidence(self, path: Track) -> Tuple[bool, bool]:
        start_pixel = (path[0][1], path[0][2])
        terminal_pixel = (path[-1][1], path[-1][2])
        crossed_net = (
            signed_distance_to_line(start_pixel, self.net_line)
            * signed_distance_to_line(terminal_pixel, self.net_line)
            <= 0
        )
        return crossed_net, not point_in_polygon(terminal_pixel, self.table)

    def projected_profile_contact(self, path: Track) -> Optional[TrackPoint]:
        """Project a long return that disappears behind the center scoreboard.

        The reviewed profile view places an opaque in-game scoreboard above
        the net. A clean return can vanish immediately before the net and stay
        hidden through its opponent-table contact. Accept only a well-sampled,
        low-error ballistic arc whose next descending table intersection is
        on the calibrated opponent half.
        """
        if (
            self.calibration.get("camera_geometry") != "profile_side_view"
            or len(path) < 12
            or self.return_evidence(path)[0]
        ):
            return None
        table_width = float(np.ptp(self.table[:, 0]))
        table_height = float(np.ptp(self.table[:, 1]))
        if table_width <= 0 or table_height <= 0:
            return None
        progress = self.projected_travel(
            path[0], path[-1], self.return_direction,
        )
        minimum_progress = (
            self.calibration.get("profile_projection_min_travel_fraction", .35)
            * table_width
        )
        if progress < minimum_progress:
            return None
        terminal_pixel = (path[-1][1], path[-1][2])
        net_distance = abs(signed_distance_to_line(terminal_pixel, self.net_line))
        maximum_net_distance = (
            self.calibration.get("profile_projection_net_distance_fraction", .08)
            * table_width
        )
        if net_distance > maximum_net_distance:
            return None

        xs = np.float64([point[1] for point in path])
        ys = np.float64([point[2] for point in path])
        if float(np.ptp(xs)) < minimum_progress:
            return None
        coefficients = np.polyfit(xs, ys, 2)
        predicted = np.polyval(coefficients, xs)
        fit_error = float(np.sqrt(np.mean(np.square(predicted - ys))))
        if (
            coefficients[0] <= 0
            or fit_error
            > self.calibration.get("profile_projection_max_fit_error", 5.0)
        ):
            return None

        # The visible cyan rail is the top edge of the calibrated table band.
        table_y = float(min(point[1] for point in self.table))
        roots = np.roots((
            coefficients[0], coefficients[1], coefficients[2] - table_y,
        ))
        direction_x = self.return_direction[0]
        if abs(direction_x) < .5:
            return None
        frame_steps = [
            (later[1] - earlier[1]) * direction_x
            for earlier, later in zip(path[-6:-1], path[-5:])
            if later[0] > earlier[0]
        ]
        forward_speed = float(np.median(frame_steps)) if frame_steps else 0.0
        if forward_speed < self.settings.return_min_horizontal_speed:
            return None
        candidates: List[Tuple[float, float]] = []
        for value in roots:
            if abs(float(value.imag)) > 1e-6:
                continue
            x = float(value.real)
            forward_distance = (x - path[-1][1]) * direction_x
            if forward_distance <= 0:
                continue
            frames_ahead = forward_distance / forward_speed
            if frames_ahead > self.calibration.get(
                "profile_projection_max_frames", 24,
            ):
                continue
            pixel = (x, table_y)
            if (
                self.profile_table_side(pixel) == "opponent"
                and point_near_polygon(
                    pixel, self.table, self.settings.table_contact_margin,
                )
            ):
                candidates.append((frames_ahead, x))
        if not candidates:
            return None
        frames_ahead, x = min(candidates)
        # Anchor the result to the last observed frame. The projected contact
        # lies in an opaque region, so claiming a future evidence timestamp
        # would produce negative publication latency.
        return path[-1][0], x, table_y, round(fit_error, 3)

    def add_projected_profile_bounce(
        self,
        path: Track,
        hit: TrackPoint,
        draw_frame: int,
        attempt: Attempt,
    ) -> None:
        """Publish a conservative opponent contact inferred through occlusion."""
        contact_key = self.physical_contact_key(hit)
        if contact_key in attempt.classified_contact_keys:
            return
        key = (path[0][0], hit[0])
        if key in self.emitted:
            return
        self.emitted.add(key)
        self.diagnose_track(
            path, "confirmed_bounce", draw_frame,
            (
                "profile_projection "
                f"contact=({hit[1]:.1f},{hit[2]:.1f}) "
                f"fit_error={hit[3]:.3f}"
            ),
        )
        candidate = ContactCandidate(
            frame_number=hit[0],
            pixel=(hit[1], hit[2]),
            log_position=None,
            table_side="opponent",
            signal_type="profile_projection",
            strength=max(0.0, 5.0 - hit[3]),
            confidence=self.calibration.get(
                "profile_projection_confidence", .48,
            ),
            source_track_key=self.track_key(path),
            approach=tuple(path[-6:]),
            departure=(),
        )
        self.record_contact_candidate(candidate, path, draw_frame, attempt)
        event = BounceEvent(
            video_time_seconds=round(hit[0] / self.fps, 3),
            video_timestamp=fmt_timestamp(hit[0] / self.fps),
            hit_table=True,
            is_in=True,
            outcome="far_table",
            posx=None,
            posy=None,
            posz=None,
            confidence=candidate.confidence,
            frame_number=hit[0],
            pixel=candidate.pixel,
            draw_frame=draw_frame,
            attempt_frame_number=attempt.frame,
        )
        attempt.classified_contact_keys.add(contact_key)
        attempt.bounces.append(event)
        attempt.bounce_track_keys.add(self.track_key(path))
        attempt.last_evidence_frame = hit[0]
        attempt.state = "contact_pending"
        self.notify_confirmed_hit(event)

    def select_return(self, attempt: Attempt) -> Track:
        return max(
            attempt.returns,
            key=lambda path: (
                all(self.return_evidence(path)),
                math.dist(path[0][1:3], path[-1][1:3]),
            ),
        )

    def no_bounce_event(self, attempt: Attempt, draw_frame: int) -> BounceEvent:
        """Describe a launcher cycle without a confirmed returned bounce."""
        crossed_net = False
        if attempt.returns:
            returned = self.select_return(attempt)
            terminal = returned[-1]
            terminal_pixel = (terminal[1], terminal[2])
            start_pixel = (returned[0][1], returned[0][2])
            start_side = signed_distance_to_line(start_pixel, self.net_line)
            end_side = signed_distance_to_line(terminal_pixel, self.net_line)
            crossed_net = start_side * end_side <= 0
            net_distance = abs(end_side)
            # Crossing the net establishes a credible return, but does not
            # make its landing unknowable. A track that continues beyond the
            # calibrated table boundary is the direct visual evidence for an
            # off-table return, whether or not it crossed the net first.
            if not point_in_polygon(terminal_pixel, self.table):
                outcome, confidence = "off_table", 0.58
            elif crossed_net:
                outcome, confidence = "unknown", 0.5
            elif net_distance <= self.calibration.get("net_proximity_fraction", 0.2) * math.dist(self.net_line[0], self.net_line[1]):
                outcome, confidence = "net", 0.55
            else:
                outcome, confidence = "unknown", 0.35
        else:
            outcome, confidence = "unknown", 0.2
        return BounceEvent(
            video_time_seconds=round(attempt.frame / self.fps, 3),
            video_timestamp=fmt_timestamp(attempt.frame / self.fps),
            hit_table=False,
            is_in=False,
            outcome=outcome,
            posx=None,
            posy=None,
            posz=None,
            confidence=confidence,
            frame_number=attempt.frame,
            pixel=attempt.pixel,
            draw_frame=draw_frame,
            attempt_frame_number=attempt.frame,
            return_crossed_net=bool(crossed_net) if attempt.returns else None,
            machine=(
                attempt.machine_telemetry.to_record(self.fps)
                if attempt.machine_telemetry else None
            ),
        )

    @staticmethod
    def player_contact_rejection_reason(
        near: ContactCandidate, far: ContactCandidate,
    ) -> Optional[str]:
        vertical_signals = {"vertical_maximum", "vertical_minimum"}
        if near.signal_type not in vertical_signals:
            return "player-side evidence is an in-flight velocity turn"
        evidence_horizon = len(near.approach) + len(near.departure)
        if far.frame_number - near.frame_number > evidence_horizon:
            return "opponent contact is outside bounded approach/departure evidence"
        if near.log_position is None or far.log_position is None:
            return "table position is occluded"
        if abs(near.log_position[2]) <= abs(far.log_position[2]):
            return "player-side turn is shallower than opponent contact"
        return None

    @staticmethod
    def confirmed_player_then_opponent(
        attempt: Attempt,
    ) -> Optional[Tuple[ContactCandidate, ContactCandidate]]:
        """Find a physically supported own-side bounce before a far contact.

        A small in-flight turn commonly appears just before a clean landing.
        It is not enough that a candidate happens first: the player-side
        candidate must contain a bounded vertical reversal and lie deeper on
        the player half than the later contact lies on the opponent half.
        Velocity flattening alone remains ambiguous and cannot establish the
        first physical contact.
        """
        contacts = [
            candidate for candidate in attempt.contact_candidates
            if candidate.accepted and candidate.log_position is not None
        ]
        for near in contacts:
            if near.table_side != "player":
                continue
            for far in contacts:
                if (
                    far.frame_number <= near.frame_number
                    or far.table_side != "opponent"
                    or far.source_track_key != near.source_track_key
                ):
                    continue
                if AttemptClassifier.player_contact_rejection_reason(
                    near, far,
                ) is None:
                    return near, far
        return None

    @staticmethod
    def contact_belongs_to_ordered_miss(
        candidate: ContactCandidate,
        ordered_contact: Optional[
            Tuple[ContactCandidate, ContactCandidate]
        ],
    ) -> bool:
        """Limit an own-side miss history to its physical source track."""
        if ordered_contact is None:
            return False
        near, far = ordered_contact
        return (
            candidate.source_track_key == far.source_track_key
            and candidate.frame_number > near.frame_number
        )

    def record_contact_history_rejections(
        self, attempt: Attempt, draw_frame: int,
    ) -> None:
        """Keep rejected ordered-history evidence available to diagnostics."""
        for near in attempt.contact_candidates:
            if near.table_side != "player":
                continue
            later = [
                far for far in attempt.contact_candidates
                if (
                    far.frame_number > near.frame_number
                    and far.table_side == "opponent"
                    and far.source_track_key == near.source_track_key
                )
            ]
            if not later or any(
                self.player_contact_rejection_reason(near, far) is None
                for far in later
            ):
                continue
            reason = self.player_contact_rejection_reason(near, later[0])
            rejected = replace(
                near, accepted=False, rejection_reason=reason,
            )
            if rejected in attempt.rejected_contact_candidates:
                continue
            attempt.rejected_contact_candidates.append(rejected)
            path = next(
                (
                    item for item in attempt.returns
                    if self.track_key(item) == near.source_track_key
                ),
                list(near.approach) + [(
                    near.frame_number, near.pixel[0], near.pixel[1], 0.0,
                )] + list(near.departure),
            )
            self.diagnose_track(
                path, "rejected_contact_candidate", draw_frame,
                json.dumps({
                    **rejected.to_record(),
                    "attempt_frame_number": attempt.frame,
                }, separators=(",", ":")),
            )

    @staticmethod
    def confirmed_net_then_opponent(
        attempt: Attempt,
    ) -> Optional[Tuple[ContactCandidate, ContactCandidate]]:
        """Find a net interaction followed by a visible opponent landing."""
        contacts = [
            candidate for candidate in attempt.contact_candidates
            if candidate.accepted
        ]
        for net in contacts:
            if net.signal_type != "net":
                continue
            for far in contacts:
                if (
                    far.frame_number > net.frame_number
                    and far.table_side == "opponent"
                    and far.log_position is not None
                ):
                    return net, far
        return None

    def record_net_contact(
        self, path: Track, draw_frame: int, attempt: Optional[Attempt] = None,
    ) -> Optional[ContactCandidate]:
        """Record a return fragment that visibly terminates at the net."""
        attempt = attempt or self.active_attempt
        if attempt is None or not path:
            return None
        crossed_net, _ = self.return_evidence(path)
        terminal = path[-1]
        pixel = (terminal[1], terminal[2])
        net_distance = abs(signed_distance_to_line(pixel, self.net_line))
        proximity = self.calibration.get("net_proximity_fraction", 0.2) * math.dist(
            self.net_line[0], self.net_line[1]
        )
        if (
            crossed_net
            or not point_in_polygon(pixel, self.table)
            or net_distance > proximity
        ):
            return None
        log_position = map_log_coordinate(
            self.homography, pixel, self.calibration["table_surface_y"],
        )
        candidate = ContactCandidate(
            frame_number=terminal[0],
            pixel=pixel,
            log_position=log_position,
            table_side="net",
            signal_type="net",
            strength=round(float(max(
                0.0, 1.0 - net_distance / max(proximity, 1e-6)
            )), 3),
            confidence=0.55,
            source_track_key=self.track_key(path),
            approach=tuple(path[-3:-1]),
            departure=(),
        )
        return self.record_contact_candidate(candidate, path, draw_frame, attempt)

    def own_side_miss_event(
        self,
        attempt: Attempt,
        selected: BounceEvent,
        near: ContactCandidate,
        far: ContactCandidate,
    ) -> BounceEvent:
        """Represent an ordered own-side then far-side history as one miss."""
        posx, posy, posz = near.log_position or (None, None, None)
        return replace(
            selected,
            video_time_seconds=round(near.frame_number / self.fps, 3),
            video_timestamp=fmt_timestamp(near.frame_number / self.fps),
            hit_table=True,
            is_in=False,
            outcome="near_table",
            posx=posx,
            posy=posy,
            posz=posz,
            confidence=round(min(near.confidence, far.confidence), 2),
            frame_number=near.frame_number,
            pixel=near.pixel,
        )

    def finalize_attempt(self, attempt: Attempt, draw_frame: int) -> None:
        """Emit one settled attempt without changing ownership order."""
        if attempt.bounces:
            ordered_contact = self.confirmed_player_then_opponent(
                attempt
            )
            self.record_contact_history_rejections(
                attempt, draw_frame,
            )
            if ordered_contact is not None:
                near, far = ordered_contact
                selected = min(
                    attempt.bounces,
                    key=lambda item: abs(item.frame_number - far.frame_number),
                )
                related_far_frames = {
                    candidate.frame_number
                    for candidate in attempt.contact_candidates
                    if (
                        candidate.table_side == "opponent"
                        and candidate.source_track_key == far.source_track_key
                        and candidate.frame_number > near.frame_number
                    )
                }
                for event in sorted(
                    attempt.bounces,
                    key=lambda item: item.frame_number,
                ):
                    if event is selected:
                        self.emit(self.own_side_miss_event(
                            attempt, selected, near, far,
                        ))
                    elif event.frame_number not in related_far_frames:
                        self.emit(event)
            else:
                for event in sorted(
                    attempt.bounces,
                    key=lambda item: item.frame_number,
                ):
                    self.emit(event)
            last_bounce_frame = max(event.frame_number for event in attempt.bounces)
            later_misses = [
                path for path in attempt.returns
                if self.track_key(path) not in attempt.bounce_track_keys
                and path[0][0] > last_bounce_frame
                and all(self.return_evidence(path))
            ]
            if later_misses:
                returned = max(
                    later_misses,
                    key=lambda path: math.dist(path[0][1:3], path[-1][1:3]),
                )
                missed_attempt = Attempt(
                    returned[0][0], (returned[0][1], returned[0][2]),
                    returns=[returned],
                )
                self.emit(self.no_bounce_event(missed_attempt, draw_frame))
        elif attempt.report_no_bounce or any(
            all(self.return_evidence(path)) for path in attempt.returns
        ):
            self.emit(self.no_bounce_event(attempt, draw_frame))

    def finish_attempt(self, draw_frame: int) -> None:
        """Settle every open attempt at EOF, preserving launch order."""
        for attempt in self.active_attempts:
            attempt.state = "settled"
        self.drain_settled_attempts(draw_frame, notify=False)

    def add_bounce(
        self,
        path: Track,
        hit: TrackPoint,
        approach: Track,
        departure: Track,
        draw_frame: int,
        attempt: Optional[Attempt] = None,
    ) -> None:
        attempt = attempt or self.active_attempt
        if attempt is None:
            return
        signal = bounce_signal(hit, approach, departure)
        # In a strict side view, the direct hit signal is the ball descending
        # and then rising on the opponent side of the net.  Other trajectory
        # turns and shadow-only peaks are useful in perspective views but add
        # ambiguity here without adding information.
        if (
            self.calibration.get("camera_geometry") == "profile_side_view"
            and signal != "vertical_maximum"
        ):
            return
        contact_key = self.physical_contact_key(hit)
        if contact_key in attempt.classified_contact_keys:
            return
        key = (path[0][0], hit[0])
        if key in self.emitted:
            return
        self.emitted.add(key)
        self.diagnose_track(
            path, "confirmed_bounce", draw_frame,
            f"{signal} hit_frame={hit[0]}",
        )
        candidate = self.record_contact_candidate(
            self.contact_candidate(path, hit, approach, departure),
            path, draw_frame, attempt,
        )
        pixel = candidate.pixel
        in_occlusion = candidate.table_side == "occluded"
        posx, posy, posz = candidate.log_position or (None, None, None)
        far = candidate.table_side == "opponent"
        confidence = candidate.confidence
        outcome = "unknown" if in_occlusion else ("far_table" if far else "near_table")
        hit_telemetry, machine_telemetry = self.telemetry_pair_for_attempt(
            attempt, hit[0],
        )
        event = BounceEvent(
            video_time_seconds=round(hit[0] / self.fps, 3),
            video_timestamp=fmt_timestamp(hit[0] / self.fps),
            hit_table=not in_occlusion,
            is_in=bool(far and not in_occlusion),
            outcome=outcome,
            posx=posx if not in_occlusion else None,
            posy=posy if not in_occlusion else None,
            posz=posz if not in_occlusion else None,
            confidence=confidence,
            frame_number=hit[0],
            pixel=pixel,
            draw_frame=draw_frame,
            attempt_frame_number=attempt.frame,
            hit=(
                hit_telemetry.to_player_record(self.fps)
                if hit_telemetry else None
            ),
            machine=(
                machine_telemetry.to_record(self.fps) if machine_telemetry else None
            ),
        )
        attempt.classified_contact_keys.add(contact_key)
        attempt.bounces.append(event)
        attempt.bounce_track_keys.add(self.track_key(path))
        attempt.last_evidence_frame = hit[0]
        attempt.state = "contact_pending"
        ordered_contact = self.confirmed_player_then_opponent(attempt)
        if (
            event.hit_table
            and event.outcome == "far_table"
            and not self.contact_belongs_to_ordered_miss(
                candidate, ordered_contact,
            )
        ):
            self.notify_confirmed_hit(event)

    def record_track_contacts(
        self,
        path: Track,
        draw_frame: int,
        allow_terminal_shadow: bool = True,
        attempt: Optional[Attempt] = None,
    ) -> None:
        """Observe every qualified contact without changing event selection."""
        for hit, approach, departure in find_bounces(
            path, self.table, self.net_line, self.settings,
            allow_terminal_shadow=allow_terminal_shadow,
        ):
            self.record_contact_candidate(
                self.contact_candidate(path, hit, approach, departure),
                path, draw_frame, attempt,
            )

    def process_tracks(self, tracks: Sequence[Track], draw_frame: int) -> None:
        for path in tracks:
            if len(path) < self.settings.min_track_points:
                attempt, reconnected = self.associate_return(path, draw_frame)
                if attempt is not None and reconnected is not None:
                    path = reconnected
                else:
                    self.diagnose_track(
                        path, "rejected", draw_frame,
                        f"too short ({len(path)}/{self.settings.min_track_points})",
                    )
                    continue
            if self.is_launcher_track(path):
                key = self.track_key(path)
                if key in self.started_launcher_tracks:
                    continue
                self.diagnose_track(path, "launcher", draw_frame)
                self.started_launcher_tracks.add(key)
                self.start_attempt(path, draw_frame)
                continue
            attempt, returned = self.associate_return(path, draw_frame)
            if attempt is None:
                reason = self.launcher_rejection_reason(path) or "no active launch"
                self.diagnose_track(path, "rejected", draw_frame, reason)
                continue
            if returned is None:
                self.diagnose_track(
                    path, "rejected", draw_frame,
                    self.return_rejection_reason(path, attempt)
                    or "not a plausible return",
                )
                continue
            path = returned
            self.diagnose_track(path, "return", draw_frame)
            if path not in attempt.returns:
                attempt.returns.append(path)
            attempt.last_evidence_frame = path[-1][0]
            attempt.state = "return_seen"
            self.record_track_contacts(path, draw_frame, attempt=attempt)
            bounce = find_bounce(path, self.table, self.net_line, self.settings)
            if bounce:
                self.add_bounce(path, *bounce, draw_frame, attempt=attempt)
                if attempt is not self.active_attempt:
                    attempt.state = "settled"
                    self.drain_settled_attempts(draw_frame)
                continue
            projected_contact = self.projected_profile_contact(path)
            if projected_contact is not None:
                self.add_projected_profile_bounce(
                    path, projected_contact, draw_frame, attempt,
                )
                if attempt is not self.active_attempt:
                    attempt.state = "settled"
                    self.drain_settled_attempts(draw_frame)
                continue
            net_contact = self.record_net_contact(path, draw_frame, attempt)
            key = self.track_key(path)
            if self.on_confirmed_non_hit is not None and key not in self.reported_non_hit_tracks:
                event = self.no_bounce_event(attempt, draw_frame)
                # A net interaction can still fall onto the opponent table.
                # Keep it pending until the attempt settles; an off-table
                # terminal path is unambiguous and remains immediate.
                if event.outcome == "off_table":
                    self.reported_non_hit_tracks.add(key)
                    self.on_confirmed_non_hit(event)
            crossed_net, _ = self.return_evidence(path)
            terminal = path[-1]
            terminal_pixel = (terminal[1], terminal[2])
            if crossed_net and point_in_polygon(terminal_pixel, self.table):
                _, _, posz = map_log_coordinate(
                    self.homography, terminal_pixel,
                    self.calibration["table_surface_y"],
                )
                # A long return that vanishes over the opponent's table is a
                # bounded-contact observation: the ball is occluded at the
                # surface before a departure segment can be tracked. This is
                # deliberately weaker evidence than a visible turn/shadow.
                if posz > 0.03 and math.dist(path[0][1:3], path[-1][1:3]) >= 300:
                    self.add_bounce(
                        path, terminal, path[-3:-1], [], draw_frame,
                        attempt=attempt,
                    )
            if attempt is not self.active_attempt and net_contact is None:
                attempt.state = "settled"
                self.drain_settled_attempts(draw_frame)

    def process_active_tracks(
        self, tracks: Sequence[Track], draw_frame: int,
    ) -> None:
        """Report a visible bounce without waiting for its track to disappear.

        Completed-track processing remains authoritative for attempts, misses,
        and diagnostics. Here we only act on a return associated with the
        current launch and require post-contact evidence for a shadow peak;
        an apparent contact on the newest frame may still become a plateau.
        """
        # Close the prior attempt as soon as the next launch has the same
        # evidence required of a completed launcher path. Waiting for its
        # tracker gap adds avoidable live latency.
        for path in tracks:
            key = self.track_key(path)
            if key in self.started_launcher_tracks or not self.is_launcher_track(path):
                continue
            self.diagnose_track(path, "launcher", draw_frame)
            self.started_launcher_tracks.add(key)
            self.start_attempt(path, draw_frame)
        if not self.active_attempts:
            return
        for path in tracks:
            key = self.track_key(path)
            if key in self.started_launcher_tracks:
                owner_frame = self.track_owners.get(key)
                attempt = (
                    self.attempt_by_frame(owner_frame)
                    if owner_frame is not None else None
                )
                returned = self.perspective_round_trip_return(path)
                if attempt is not None and returned is not None:
                    if returned not in attempt.returns:
                        attempt.returns.append(returned)
                    attempt.last_evidence_frame = returned[-1][0]
                    attempt.state = "return_seen"
                    self.record_track_contacts(
                        returned, draw_frame, allow_terminal_shadow=False,
                        attempt=attempt,
                    )
                    bounce = self.perspective_round_trip_bounce(
                        returned, allow_terminal_shadow=False,
                    )
                    if bounce:
                        self.add_bounce(
                            returned, *bounce, draw_frame, attempt=attempt,
                        )
                continue
            attempt, returned = self.associate_return(path, draw_frame)
            if attempt is None or returned is None:
                continue
            self.record_track_contacts(
                returned, draw_frame, allow_terminal_shadow=False,
                attempt=attempt,
            )
            bounce = find_bounce(
                returned, self.table, self.net_line, self.settings,
                allow_terminal_shadow=False,
            )
            if bounce:
                self.add_bounce(returned, *bounce, draw_frame, attempt=attempt)
        self.expire_attempts(draw_frame)


def infer_attempt_period(hit_frames: Sequence[int], fps: float) -> Optional[float]:
    """Infer the repeating ball-machine cycle from confirmed table contacts."""
    if len(hit_frames) < 3:
        return None
    phase = hit_frames[0]
    best: Optional[Tuple[float, float]] = None
    lower, upper, step = fps, fps * 2.2, 0.1
    period = lower
    while period <= upper:
        residuals = sorted(
            abs((frame - phase + period / 2) % period - period / 2)
            for frame in hit_frames
        )
        kept = residuals[:max(3, round(len(residuals) * .9))]
        score = sum(kept) / len(kept)
        if best is None or score < best[0]:
            best = (score, period)
        period += step
    return best[1] if best else None


def attempt_event_slots(
    events: Sequence[BounceEvent],
    total_frames: int,
    fps: float,
    fixed_period: Optional[float] = None,
    fixed_phase: Optional[float] = None,
) -> Tuple[Optional[float], List[Tuple[int, BounceEvent]]]:
    """Build the canonical cadence slots used by live and final output."""
    hits = [event for event in events if event.hit_table and event.outcome == "far_table"]
    period = fixed_period or infer_attempt_period(
        [event.frame_number for event in hits], fps,
    )
    if period is None:
        return None, []

    if fixed_phase is None:
        phase = hits[0].frame_number
        signed = [
            (event.frame_number - phase + period / 2) % period - period / 2
            for event in hits
        ]
        phase += sorted(signed)[len(signed) // 2]
        earliest_evidence = min(event.frame_number for event in events)
        while phase - period >= earliest_evidence - period * .3:
            phase -= period
    else:
        # A live ledger cannot rename attempt IDs when another hit slightly
        # refines the cadence estimate. Keep the phase that established it.
        phase = fixed_phase
    # A live source may sit idle for minutes before and after a drill. Cadence
    # can fill gaps *between* observed attempts, but must not manufacture a
    # cycle after the machine disappears. Contact evidence can occur well
    # after a cadence anchor, so bound the tail from the launch/attempt marker
    # when it is available. Three quarters of a period includes that marker's
    # own nearest anchor while staying short of the following unobserved one.
    # draw_frame can be the moment Ctrl-C/EOF finally closes an attempt, long
    # after the shot itself, and therefore never bounds active cadence.
    latest_attempt = max(
        (
            event.attempt_frame_number
            if event.attempt_frame_number is not None
            else event.frame_number
        )
        for event in events
    )
    total_frames = min(total_frames, round(latest_attempt + period * .75))
    anchors: List[int] = []
    anchor = phase
    while anchor < total_frames:
        anchors.append(round(anchor))
        anchor += period
    if not anchors:
        return None, []
    slots: List[Optional[BounceEvent]] = [None] * len(anchors)
    hit_slots: Dict[int, int] = {}
    for event in hits:
        event_frame = event.frame_number
        slot = min(range(len(anchors)), key=lambda index: abs(anchors[index] - event_frame))
        if abs(anchors[slot] - event_frame) > period * .3:
            continue
        current = slots[slot]
        if (
            current is None
            or abs(anchors[slot] - event.frame_number)
            < abs(anchors[slot] - current.frame_number)
            or (
                abs(anchors[slot] - event.frame_number)
                == abs(anchors[slot] - current.frame_number)
                and event.confidence > current.confidence
            )
        ):
            slots[slot] = replace(event, outcome="hit")
        hit_slots[id(event)] = slot

    normalized: List[Tuple[int, BounceEvent]] = []
    for anchor, event in zip(anchors, slots):
        if event is not None:
            normalized.append((anchor, event))
            continue
        frame = min(anchor, total_frames - 1)
        normalized.append((anchor, BounceEvent(
            video_time_seconds=round(frame / fps, 3),
            video_timestamp=fmt_timestamp(frame / fps),
            hit_table=False,
            is_in=False,
            outcome="miss",
            posx=None,
            posy=None,
            posz=None,
            confidence=0.3,
            frame_number=frame,
            pixel=(0, 0),
            draw_frame=frame,
        )))
    return period, normalized


def normalize_attempt_events(
    events: Sequence[BounceEvent], total_frames: int, fps: float,
) -> List[BounceEvent]:
    """Return exactly one user-facing result for every inferred launch cycle.

    Confirmed opponent-table contacts establish the machine's cadence. Gaps
    in that cadence become misses when the next cycle arrives, which is the
    only reliable way to report a ball that was completely occluded.
    """
    period, slots = attempt_event_slots(events, total_frames, fps)
    if period is None:
        return [
            replace(
                event,
                outcome=(
                    "hit"
                    if event.hit_table and event.outcome == "far_table"
                    else "miss"
                ),
            )
            for event in events
        ]
    normalized = [
        replace(event, attempt_frame_number=anchor) for anchor, event in slots
    ]
    return [
        replace(event, outcome="hit" if event.outcome == "hit" else "miss")
        for event in normalized
    ]


class LiveAttemptNormalizer:
    """Publish one monotonic ledger entry for every inferred machine launch.

    Cadence is needed because an entirely unseen ball has no visual track to
    anchor it. Slot indexes become stable attempt IDs as soon as three hits
    establish cadence. The newest slot remains pending until later credible
    evidence closes it; finalized entries are never revised.
    """

    def __init__(
        self,
        fps: float,
        on_attempt: Callable[[Dict[str, Any]], None],
        minimum_cadence_hits: int = 3,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.fps = fps
        self.on_attempt = on_attempt
        self.minimum_cadence_hits = minimum_cadence_hits
        self.on_status = on_status
        self.launches_seen = 0
        self.events: List[BounceEvent] = []
        self.period: Optional[float] = None
        self.phase: Optional[float] = None
        self.ledger: List[Dict[str, Any]] = []
        self.pending_attempt_events: List[BounceEvent] = []
        self.latest_trusted_frame: Optional[int] = None
        self.trusted_allows_following_slot = True

    def remember_event(self, event: BounceEvent) -> None:
        # The classifier can confirm the same physical table contact through
        # two overlapping tracks. Canonical normalization collapses those
        # into one cadence slot; do that before live cadence inference too.
        if (
            event.hit_table
            and event.outcome == "far_table"
            and any(
                item.hit_table
                and item.outcome == "far_table"
                and abs(item.frame_number - event.frame_number)
                <= self.fps * .25
                for item in self.events
            )
        ):
            return
        if not any(
            item.frame_number == event.frame_number
            and item.outcome == event.outcome
            and item.pixel == event.pixel
            for item in self.events
        ):
            self.events.append(event)

    def observe(self, event: BounceEvent) -> None:
        self.remember_event(event)
        self.pending_attempt_events.append(event)
        # Every emitted far-table event is already a finalized detector
        # decision. Some contact paths deliberately skip the earlier
        # low-latency callback, so publish the hit here as a reliable fallback
        # instead of waiting for finish_session() to revise an old miss.
        if event.hit_table and event.outcome == "far_table":
            self.observe_confirmed_hit(event)

    def observe_attempt_started(self, _anchor: int) -> None:
        self.launches_seen += 1
        if self.period is None and self.on_status is not None:
            self.on_status({
                "type": "counter_status",
                "status": "warming_up",
                "message": (
                    f"Calibrating ball cadence "
                    f"({self.launches_seen} launches observed)"
                ),
            })

    def finished_attempt_event(self) -> Optional[BounceEvent]:
        """Build the non-hit closed by a new launch, if one is needed."""
        pending = self.pending_attempt_events
        self.pending_attempt_events = []
        if not pending or any(
            event.hit_table and event.outcome == "far_table"
            for event in pending
        ):
            return None
        event = max(
            pending,
            key=lambda item: (
                item.outcome == "off_table",
                item.outcome == "net",
                item.confidence,
            ),
        )
        return replace(event, outcome="miss")

    def candidate_slots(
        self, extra: Optional[BounceEvent] = None, total_frames: Optional[int] = None,
    ) -> List[Tuple[int, BounceEvent]]:
        evidence = list(self.events)
        if extra is not None and not any(
            (
                item.frame_number == extra.frame_number
                and item.outcome == extra.outcome
            )
            or (
                item.hit_table
                and item.outcome == "far_table"
                and extra.hit_table
                and extra.outcome == "far_table"
                and abs(item.frame_number - extra.frame_number)
                <= self.fps * .25
            )
            for item in evidence
        ):
            evidence.append(extra)
        hits = [
            item for item in evidence
            if item.hit_table and item.outcome == "far_table"
        ]
        if self.period is None and len(hits) < self.minimum_cadence_hits:
            return []
        horizon = total_frames
        if horizon is None:
            if not evidence:
                return []
            estimated_period = self.period or infer_attempt_period(
                [item.frame_number for item in hits], self.fps,
            )
            if estimated_period is None:
                return []
            horizon = round(
                max(item.frame_number for item in evidence)
                + estimated_period * 1.05
            )
        period, slots = attempt_event_slots(
            evidence,
            horizon,
            self.fps,
            fixed_period=self.period,
            fixed_phase=self.phase,
        )
        if period is None:
            return []
        if self.period is None:
            self.period = period
            self.phase = float(slots[0][0])
        if self.latest_trusted_frame is not None:
            tail = period * (
                1.05 if self.trusted_allows_following_slot else .3
            )
            slots = [
                item for item in slots
                if item[0] <= self.latest_trusted_frame + tail
            ]
        return slots

    def attempt_record(
        self, index: int, anchor: int, state: str,
        event: Optional[BounceEvent] = None,
        decision_frame_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "attempt_id": f"attempt-{index + 1:04d}",
            "sequence": index + 1,
            "anchor_frame_number": anchor,
            "state": state,
        }
        if event is not None:
            record.update(event.to_record())
            record["outcome"] = "hit" if event.outcome == "hit" else "miss"
            record["attempt_frame_number"] = anchor
            record["decision_frame_number"] = (
                event.draw_frame
                if decision_frame_number is None
                else decision_frame_number
            )
        return record

    def sync_slots(
        self, slots: Sequence[Tuple[int, BounceEvent]], finalize_through: int,
        decision_frame_number: Optional[int] = None,
    ) -> None:
        for index, (anchor, _event) in enumerate(slots):
            if index < len(self.ledger):
                continue
            pending = self.attempt_record(index, anchor, "pending")
            self.ledger.append(pending)
            self.on_attempt(pending)
        last = min(finalize_through, len(slots) - 1)
        for index in range(last + 1):
            if self.ledger[index]["state"] == "finalized":
                continue
            anchor = self.ledger[index]["anchor_frame_number"]
            finalized = self.attempt_record(
                index, anchor, "finalized", slots[index][1],
                decision_frame_number=decision_frame_number,
            )
            self.ledger[index] = finalized
            self.on_attempt(finalized)

    def finalize_direct(
        self, event: BounceEvent, outcome: str,
        target_frame: Optional[float] = None,
        infer_prior_misses: bool = False,
    ) -> None:
        direct = replace(event, outcome=outcome)
        slots = self.candidate_slots(event)
        if not slots or self.period is None:
            return
        self.sync_slots(slots, -1)
        logical_frame = event.frame_number if target_frame is None else target_frame
        target = min(
            range(len(self.ledger)),
            key=lambda index: abs(
                self.ledger[index]["anchor_frame_number"] - logical_frame
            ),
        )
        if (
            abs(self.ledger[target]["anchor_frame_number"] - logical_frame)
            > self.period * .5
        ):
            return
        existing = self.ledger[target]
        if existing["state"] == "finalized":
            if outcome != "hit" or existing.get("outcome") == "hit":
                return
            anchor = existing["anchor_frame_number"]
            corrected = self.attempt_record(
                target, anchor, "finalized", direct,
            )
            corrected["revision"] = existing.get("revision", 0) + 1
            self.ledger[target] = corrected
            self.on_attempt(corrected)
            return
        pending = [
            index for index, item in enumerate(self.ledger)
            if item["state"] == "pending"
        ]
        if not pending:
            return
        if infer_prior_misses:
            for index in pending:
                if index >= target:
                    break
                anchor = self.ledger[index]["anchor_frame_number"]
                missed = BounceEvent(
                    video_time_seconds=round(anchor / self.fps, 3),
                    video_timestamp=fmt_timestamp(anchor / self.fps),
                    hit_table=False,
                    is_in=False,
                    outcome="miss",
                    posx=None,
                    posy=None,
                    posz=None,
                    confidence=0.3,
                    frame_number=anchor,
                    pixel=(0, 0),
                    draw_frame=anchor,
                    attempt_frame_number=anchor,
                )
                finalized_miss = self.attempt_record(
                    index, anchor, "finalized", missed,
                    decision_frame_number=direct.draw_frame,
                )
                self.ledger[index] = finalized_miss
                self.on_attempt(finalized_miss)
        else:
            self.sync_slots(
                slots,
                target - 1,
                decision_frame_number=direct.draw_frame,
            )
        if self.ledger[target]["state"] == "finalized":
            return
        anchor = self.ledger[target]["anchor_frame_number"]
        finalized = self.attempt_record(target, anchor, "finalized", direct)
        self.ledger[target] = finalized
        self.on_attempt(finalized)

    def observe_confirmed_hit(self, event: BounceEvent) -> None:
        """Publish direct visual evidence without waiting for cadence."""
        self.remember_event(event)
        self.refine_cadence()
        self.latest_trusted_frame = max(
            self.latest_trusted_frame or event.frame_number,
            event.frame_number,
        )
        self.trusted_allows_following_slot = True
        self.finalize_direct(event, "hit")
        self.retry_confirmed_hits()
        if (
            self.period is not None
            and self.ledger
            and not any(item["state"] == "pending" for item in self.ledger)
        ):
            anchor = round(self.ledger[-1]["anchor_frame_number"] + self.period)
            pending = self.attempt_record(len(self.ledger), anchor, "pending")
            self.ledger.append(pending)
            self.on_attempt(pending)

    def refine_cadence(self) -> None:
        """Refine future slot spacing without moving published attempt IDs."""
        if self.period is None or self.phase is None:
            return
        hits = [
            item for item in self.events
            if item.hit_table and item.outcome == "far_table"
        ]
        if len(hits) < self.minimum_cadence_hits + 2:
            return
        refined = infer_attempt_period(
            [item.frame_number for item in hits], self.fps,
        )
        if (
            refined is None
            or abs(refined - self.period) > self.period * .03
        ):
            return
        if self.ledger:
            last_index = len(self.ledger) - 1
            last_anchor = self.ledger[last_index]["anchor_frame_number"]
            self.phase = last_anchor - last_index * refined
        self.period = refined

    def retry_confirmed_hits(self) -> None:
        """Publish remembered hits once cadence exposes their ledger slots."""
        if self.period is None:
            return
        for event in self.events:
            if event.hit_table and event.outcome == "far_table":
                self.finalize_direct(event, "hit")

    def observe_confirmed_non_hit(self, event: BounceEvent) -> None:
        """Hold non-hit evidence until the current attempt is closed.

        A completed return track can look off-table before another track from
        the same attempt confirms the bounce. Keep it with the current attempt:
        ``settle_attempt`` will prefer any confirmed hit, or finalize this as a
        genuine non-hit when the next launch closes the attempt. A still-later
        confirmed hit is allowed to correct that inferred boundary.
        """
        self.pending_attempt_events.append(event)

    def advance(self, frame_number: int) -> None:
        """Finalize an overdue unseen slot after a conservative cadence wait."""
        if self.period is None:
            return
        pending = next((
            index for index, item in enumerate(self.ledger)
            if item["state"] == "pending"
        ), None)
        if pending is None:
            return
        anchor = self.ledger[pending]["anchor_frame_number"]
        if frame_number < anchor + self.period * 2.2:
            return
        missed = BounceEvent(
            video_time_seconds=round(anchor / self.fps, 3),
            video_timestamp=fmt_timestamp(anchor / self.fps),
            hit_table=False,
            is_in=False,
            outcome="miss",
            posx=None,
            posy=None,
            posz=None,
            confidence=0.3,
            frame_number=anchor,
            pixel=(0, 0),
            draw_frame=frame_number,
            attempt_frame_number=anchor,
        )
        finalized = self.attempt_record(pending, anchor, "finalized", missed)
        self.ledger[pending] = finalized
        self.on_attempt(finalized)

    def settle_attempt(self, next_launch_frame: Optional[int] = None) -> None:
        """Advance once after a detected launch closes the prior attempt."""
        self.finished_attempt_event()
        total_frames = None
        launch_marker = None
        if next_launch_frame is not None and self.period is not None:
            total_frames = round(next_launch_frame + self.period * 1.05)
            launch_marker = BounceEvent(
                video_time_seconds=round(next_launch_frame / self.fps, 3),
                video_timestamp=fmt_timestamp(next_launch_frame / self.fps),
                hit_table=False,
                is_in=False,
                outcome="unknown",
                posx=None,
                posy=None,
                posz=None,
                confidence=0.2,
                frame_number=next_launch_frame,
                pixel=(0, 0),
                draw_frame=next_launch_frame,
                attempt_frame_number=next_launch_frame,
            )
        slots = self.candidate_slots(
            extra=launch_marker, total_frames=total_frames,
        )
        if slots:
            # A single later launcher-like track can be a fragment of the same
            # attempt. Two later cadence slots make an unseen miss stable while
            # leaving time for a long visible out track to finish.
            self.sync_slots(
                slots,
                len(slots) - 3,
                decision_frame_number=next_launch_frame,
            )
            self.retry_confirmed_hits()

    def finalize(self, total_frames: int) -> List[BounceEvent]:
        return normalize_attempt_events(self.events, total_frames, self.fps)

    def finish_session(self, total_frames: Optional[int] = None) -> None:
        """Flush the final detected attempt without inventing trailing cycles."""
        final_event = self.finished_attempt_event()
        if total_frames is None:
            return
        # Direct-hit callbacks normally finalize these during processing. Also
        # replay them here so callers that only feed completed events get the
        # same ledger without converting an inferred tail into attempts.
        hits = [
            event for event in self.events
            if event.hit_table and event.outcome == "far_table"
        ]
        if hits:
            self.latest_trusted_frame = max(event.frame_number for event in hits)
            self.trusted_allows_following_slot = True
            for event in hits:
                self.finalize_direct(event, "hit")
        if final_event is not None:
            self.latest_trusted_frame = max(
                self.latest_trusted_frame or final_event.frame_number,
                final_event.frame_number,
            )
            self.trusted_allows_following_slot = False
        slots = self.candidate_slots(total_frames=total_frames)
        self.sync_slots(
            slots,
            len(slots) - 1,
            decision_frame_number=total_frames,
        )
        if final_event is not None:
            self.finalize_direct(
                final_event, "miss", target_frame=final_event.frame_number,
            )
        if self.period is not None:
            latest_evidence_anchor = max(
                (
                    event.attempt_frame_number
                    if event.attempt_frame_number is not None
                    else event.frame_number
                )
                for event in self.events
            ) if self.events else None
            for index, record in enumerate(self.ledger):
                if record["state"] != "pending":
                    continue
                anchor = record["anchor_frame_number"]
                nearby_non_hits = [
                    event for event in self.events
                    if not (
                        event.hit_table and event.outcome == "far_table"
                    )
                    and abs(
                        (
                            event.attempt_frame_number
                            if event.attempt_frame_number is not None
                            else event.frame_number
                        )
                        - anchor
                    )
                    <= self.period * .55
                ]
                if (
                    not nearby_non_hits
                    and (
                        latest_evidence_anchor is None
                        or anchor
                        > latest_evidence_anchor + self.period * .55
                    )
                ):
                    continue
                event = (
                    min(
                        nearby_non_hits,
                        key=lambda item: abs(
                            (
                                item.attempt_frame_number
                                if item.attempt_frame_number is not None
                                else item.frame_number
                            )
                            - anchor
                        ),
                    )
                    if nearby_non_hits
                    else BounceEvent(
                        video_time_seconds=round(anchor / self.fps, 3),
                        video_timestamp=fmt_timestamp(anchor / self.fps),
                        hit_table=False,
                        is_in=False,
                        outcome="miss",
                        posx=None,
                        posy=None,
                        posz=None,
                        confidence=0.3,
                        frame_number=anchor,
                        pixel=(0, 0),
                        draw_frame=total_frames,
                        attempt_frame_number=anchor,
                    )
                )
                finalized = self.attempt_record(
                    index, anchor, "finalized",
                    replace(event, outcome="miss"),
                    decision_frame_number=total_frames,
                )
                self.ledger[index] = finalized
                self.on_attempt(finalized)


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
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
    if not writer.isOpened():
        writer.release()
        raise SystemExit(f"Could not create annotated video at {path}")
    return writer


def reset_output_file(path: PathLike) -> None:
    """Start a new analysis session with no results from the prior session."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")


def process_video(
    source: VideoSource,
    scale: float,
    calibration: Calibration,
    homography: np.ndarray,
    table: np.ndarray,
    end_seconds: Optional[float] = None,
    writer: Optional[cv2.VideoWriter] = None,
    first_frame: Optional[VideoFrame] = None,
    on_event: Optional[Callable[[BounceEvent], None]] = None,
    on_attempt_finished: Optional[Callable[[Optional[int]], None]] = None,
    on_confirmed_hit: Optional[Callable[[BounceEvent], None]] = None,
    on_confirmed_non_hit: Optional[Callable[[BounceEvent], None]] = None,
    on_processing_frame: Optional[Callable[[int], None]] = None,
    clean_writer: Optional[cv2.VideoWriter] = None,
    clean_frame_limit: Optional[int] = None,
    clean_start_on_launch: bool = False,
    on_attempt_started: Optional[Callable[[int], None]] = None,
    on_track_diagnostic: Optional[
        Callable[[TrackDiagnostic, int], None]
    ] = None,
    on_frame_processed: Optional[
        Callable[[int, Dict[str, Any]], None]
    ] = None,
    on_preview_frame: Optional[Callable[[int, np.ndarray], None]] = None,
) -> List[BounceEvent]:
    """Process an already-open source and return its detected bounce events."""
    fps = source.fps
    video_width, video_height = source.width, source.height
    width, height = round(video_width * scale), round(video_height * scale)
    settings = DetectorSettings.from_calibration(calibration)
    net_line = np.float32(calibration["net_line"]) * scale
    occlusion = np.float32(calibration.get("occlusion_polygon", [])) * scale
    tracking_polygon = np.float32(calibration["tracking_polygon"]) * scale
    contact_polygon = np.float32(
        calibration.get("table_contact_polygon", calibration["table_polygon"])
    ) * scale
    tracker = MultiBallTracker(settings)
    diagnostics = (
        DetectorDiagnostics(track_lifetime_frames=max(1, round(fps * .5)))
        if writer is not None or on_preview_frame is not None else None
    )
    telemetry = TelemetryReader()
    clean_recording_started = not clean_start_on_launch

    def mark_clean_recording_started(frame_number: int) -> None:
        nonlocal clean_recording_started
        clean_recording_started = True
        if on_attempt_started is not None:
            on_attempt_started(frame_number)

    def report_track(diagnostic: TrackDiagnostic, frame_number: int) -> None:
        if diagnostics is not None:
            diagnostics.completed_track(diagnostic, frame_number)
        if on_track_diagnostic is not None:
            on_track_diagnostic(diagnostic, frame_number)

    classifier = AttemptClassifier(
        fps, calibration, contact_polygon, net_line, occlusion, homography,
        video_width, video_height, scale, settings, on_event,
        on_attempt_finished, on_confirmed_hit, on_confirmed_non_hit,
        report_track if diagnostics is not None or on_track_diagnostic is not None else None,
        mark_clean_recording_started,
    )
    previous_gray = None
    next_frame = first_frame
    frame_number = first_frame.number if first_frame is not None else 0
    clean_frames_written = 0
    clean_seed_written = False
    try:
        while True:
            video_frame = next_frame if next_frame is not None else source.read()
            next_frame = None
            if video_frame is None or (
                end_seconds is not None
                and video_frame.time_seconds >= end_seconds
            ):
                break
            frame_number = video_frame.number
            original = video_frame.image
            if on_processing_frame is not None:
                on_processing_frame(frame_number)
            if (
                clean_writer is not None
                and clean_start_on_launch
                and not clean_seed_written
            ):
                # Keep one stable setup frame for automatic calibration. The
                # continuous bounded recording still waits for detector
                # activity, so a long headset setup does not consume it.
                clean_seed_written = True
            if frame_number % 3 == 0:
                reading = telemetry.update(original, frame_number)
                if reading is not None:
                    classifier.observe_telemetry(reading)
            frame = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
            wrote_clean_seed = (
                clean_writer is not None
                and clean_start_on_launch
                and clean_frames_written == 0
                and clean_seed_written
            )
            if wrote_clean_seed and clean_writer is not None:
                clean_writer.write(frame)
                clean_frames_written += 1
            elif (
                clean_writer is not None
                and clean_recording_started
                and (
                    clean_frame_limit is None
                    or clean_frames_written < clean_frame_limit
                )
            ):
                # Store the exact resized detector input losslessly before any
                # diagnostics are drawn. This makes offline replay pixel-equivalent.
                clean_writer.write(frame)
                clean_frames_written += 1
            if diagnostics is not None:
                diagnostics.begin_frame()
            gray, candidates = candidates_for_frame(
                frame, previous_gray, tracking_polygon, settings, diagnostics,
            )
            previous_gray = gray
            completed_tracks = tracker.update(frame_number, candidates)
            classifier.process_tracks(completed_tracks, frame_number)
            classifier.process_active_tracks(tracker.confirmed_tracks, frame_number)
            if diagnostics is not None:
                diagnostics.set_unconfirmed_tracks(tracker.tracks)
            if writer is not None or on_preview_frame is not None:
                annotated_frame = draw_overlay(
                    frame, table, net_line, tracker.visible_points, classifier.events,
                    homography, calibration["table_surface_y"],
                    diagnostics, frame_number,
                )
                if writer is not None:
                    writer.write(annotated_frame)
                if on_preview_frame is not None:
                    on_preview_frame(frame_number, annotated_frame)
            if on_frame_processed is not None:
                on_frame_processed(frame_number, {
                    "candidate_count": len(candidates),
                    "active_track_count": len(tracker.tracks),
                    "confirmed_track_count": len(tracker.confirmed_tracks),
                    "completed_track_count": len(completed_tracks),
                    "detected_event_count": len(classifier.events),
                })
            frame_number += 1
    except KeyboardInterrupt:
        # A live source normally ends when the user stops it. Preserve and
        # flush the completed session instead of discarding every event.
        pass
    if on_processing_frame is not None:
        on_processing_frame(frame_number)
    classifier.finish_attempt(frame_number)
    normalized = normalize_attempt_events(classifier.events, frame_number, fps)
    return attach_missing_machine_telemetry(
        normalized, classifier.telemetry_history, fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="video file or srt:// URL")
    parser.add_argument("--calibration", help="Optional manually reviewed JSON calibration")
    parser.add_argument("--extract-calibration-frame", metavar="PNG", help="write a frame for per-camera corner calibration, then exit")
    parser.add_argument("--output", default="video_bounces.jsonl")
    parser.add_argument(
        "--live-stdout",
        action="store_true",
        help="print each detected event immediately instead of watching the JSONL file",
    )
    parser.add_argument(
        "--annotated",
        nargs="?",
        const="video_bounces_annotated.mp4",
        metavar="MP4",
        help="write annotated video, optionally to a custom path",
    )
    parser.add_argument(
        "--clean-recording",
        metavar="MP4",
        help="record clean decoded detector input before overlays are drawn",
    )
    parser.add_argument(
        "--clean-recording-seconds",
        type=float,
        default=120,
        help="maximum clean recording length (default: 120 seconds)",
    )
    parser.add_argument(
        "--clean-recording-start",
        choices=("launch", "immediate"),
        default="launch",
        help="start clean capture at the first detected launch (default) or immediately",
    )
    parser.add_argument(
        "--clean-recording-codec",
        choices=("ffv1", "mjpeg"),
        default="ffv1",
        help="lossless FFV1 or lower-overhead MJPEG capture (default: ffv1)",
    )
    parser.add_argument(
        "--live-events",
        metavar="JSONL",
        help="append-only live publication log with shot and publication frames",
    )
    parser.add_argument(
        "--bounce-diagnostics",
        metavar="JSONL",
        help="write observation-only confirmed-bounce paths and signal kinds",
    )
    parser.add_argument(
        "--contact-diagnostics",
        metavar="JSONL",
        help="write ordered contact candidates without changing classification",
    )
    parser.add_argument(
        "--track-diagnostics",
        metavar="JSONL",
        help="write every completed track decision and rejection reason",
    )
    parser.add_argument("--no-annotated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-seconds", type=float, default=0, help="seek point; useful when reviewing a short interval")
    parser.add_argument("--end-seconds", type=float, help="stop after this video timestamp")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace prerecorded input against wall-clock time",
    )
    parser.add_argument(
        "--preview-stdout",
        action="store_true",
        help="emit throttled JPEG detector previews on stdout for the counter server",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=12,
        help="maximum browser preview rate (default: 12 FPS)",
    )
    args = parser.parse_args()
    if args.clean_recording_seconds <= 0:
        parser.error("--clean-recording-seconds must be greater than zero")
    if args.preview_fps <= 0:
        parser.error("--preview-fps must be greater than zero")
    annotated_path = None if args.no_annotated else args.annotated
    if not args.extract_calibration_frame:
        # Live SRT sources can block while waiting for the sender. Clear stale
        # events before opening the source so observers immediately see that a
        # new server session has begun.
        reset_output_file(args.output)
        if args.live_events is not None:
            reset_output_file(args.live_events)
    try:
        source = open_video_source(args.video, realtime=args.realtime)
        try:
            source.seek_seconds(args.start_seconds)
        except VideoSourceError:
            source.close()
            raise
    except VideoSourceError as exc:
        raise SystemExit(str(exc)) from exc
    fps = source.fps
    video_width, video_height = source.width, source.height
    scale = min(1.0, PROCESSING_WIDTH / video_width)
    width, height = round(video_width * scale), round(video_height * scale)
    if args.extract_calibration_frame:
        video_frame = source.read()
        source.close()
        if video_frame is None:
            raise SystemExit("Could not read a calibration frame")
        if not cv2.imwrite(args.extract_calibration_frame, video_frame.image):
            raise SystemExit(f"Could not write calibration frame to {args.extract_calibration_frame}")
        print(json.dumps({"calibration_frame": args.extract_calibration_frame, "image_size": [video_width, video_height]}, indent=2))
        return
    writer = None
    clean_writer = None
    live_events_output = None
    bounce_diagnostics_output = None
    contact_diagnostics_output = None
    track_diagnostics_output = None
    first_frame = None
    try:
        if args.calibration:
            calibration, homography, table = load_calibration(
                args.calibration, video_width, video_height, scale,
            )
        else:
            first_frame = source.read()
            if first_frame is None:
                raise SystemExit("Could not read the first frame for automatic calibration")
            try:
                detected, _ = calibration_from_frame(
                    first_frame.image, first_frame.number,
                )
            except ValueError as exc:
                raise SystemExit(f"Automatic calibration failed: {exc}.") from exc
            calibration_start_frame = first_frame.number
            calibration_frame_image = first_frame.image

            def color_calibration_frames() -> Iterable[np.ndarray]:
                yield calibration_frame_image
                for _ in range(max(0, round(fps * 8) - 1)):
                    sampled = source.read()
                    if sampled is None:
                        break
                    yield sampled.image

            # Ball hue needs motion evidence and therefore cannot come from the
            # single setup frame used for table geometry. Saved files rewind
            # after this prefix; a live stream intentionally treats it as a
            # short color-calibration warm-up.
            ball_color = infer_ball_color(color_calibration_frames(), detected)
            if ball_color is not None:
                detected["ball_color"] = ball_color
            if source.seekable:
                source.seek_frame(calibration_start_frame)
                first_frame = source.read()
                if first_frame is None:
                    raise SystemExit("Could not rewind after automatic color calibration")
            else:
                first_frame = None
            calibration, homography, table = calibration_geometry(
                detected, video_width, video_height, scale,
            )
        if annotated_path is not None:
            writer = create_video_writer(annotated_path, fps, (width, height))
        if args.clean_recording is not None:
            if Path(args.clean_recording).suffix.lower() != ".mkv":
                raise SystemExit("--clean-recording must use an .mkv path")
            clean_codec = "FFV1" if args.clean_recording_codec == "ffv1" else "MJPG"
            clean_writer = create_video_writer(
                args.clean_recording, fps, (width, height), codec=clean_codec,
            )
        if args.live_events is not None:
            live_events_output = open(args.live_events, "a", encoding="utf-8")
        if args.bounce_diagnostics is not None:
            reset_output_file(args.bounce_diagnostics)
            bounce_diagnostics_output = open(
                args.bounce_diagnostics, "a", encoding="utf-8",
            )
        if args.contact_diagnostics is not None:
            reset_output_file(args.contact_diagnostics)
            contact_diagnostics_output = open(
                args.contact_diagnostics, "a", encoding="utf-8",
            )
        if args.track_diagnostics is not None:
            reset_output_file(args.track_diagnostics)
            track_diagnostics_output = open(
                args.track_diagnostics, "a", encoding="utf-8",
            )
        with open(args.output, "w", encoding="utf-8") as output:
            processing_frame = first_frame.number if first_frame is not None else 0
            live_normalizer: Optional[Any] = None
            health_monitor: Optional[LiveCounterHealthMonitor] = None
            live_stdout_open = args.live_stdout
            attempt_states: Dict[str, str] = {}
            attempt_upsert_count = 0

            def observe_processing_frame(frame_number: int) -> None:
                nonlocal processing_frame
                processing_frame = frame_number
                if live_normalizer is not None:
                    live_normalizer.advance(frame_number)

            def write_live_record(record: Dict[str, Any]) -> None:
                nonlocal live_stdout_open
                serialized = json.dumps(record)
                if live_events_output is not None:
                    live_events_output.write(serialized + "\n")
                    live_events_output.flush()
                if live_stdout_open:
                    try:
                        print(serialized, flush=True)
                    except BrokenPipeError:
                        live_stdout_open = False
                        sys.stdout = open("/dev/null", "w", encoding="utf-8")

            if source.live:
                source.set_event_callback(write_live_record)
                health_monitor = LiveCounterHealthMonitor(
                    write_live_record,
                    publisher_event_threshold=12,
                )

            def write_attempt(attempt: Dict[str, Any]) -> None:
                nonlocal attempt_upsert_count
                record = {"type": "attempt_upsert", **attempt}
                record["publication_frame_number"] = processing_frame
                record["publication_video_time_seconds"] = round(
                    processing_frame / fps, 3,
                )
                if attempt["state"] == "finalized":
                    evidence_frame = attempt.get(
                        "frame_number", attempt["anchor_frame_number"],
                    )
                    decision_frame = attempt.get(
                        "decision_frame_number", evidence_frame,
                    )
                    record["publication_delay_frames"] = (
                        processing_frame - evidence_frame
                    )
                    record["publication_delay_seconds"] = round(
                        (processing_frame - evidence_frame) / fps, 3,
                    )
                    record["attempt_publication_delay_seconds"] = round(
                        (processing_frame - attempt["anchor_frame_number"]) / fps,
                        3,
                    )
                    record["decision_publication_delay_seconds"] = round(
                        (processing_frame - decision_frame) / fps,
                        3,
                    )
                    record["feedback_delay_seconds"] = (
                        record["publication_delay_seconds"]
                        if attempt.get("outcome") == "hit"
                        else record["decision_publication_delay_seconds"]
                    )
                attempt_upsert_count += 1
                attempt_states[record["attempt_id"]] = record["state"]
                write_live_record(record)
                if health_monitor is not None:
                    health_monitor.observe_attempt(record)

            # Detector-native launch fragments create extra misses and omit
            # fully occluded launches. Establish cadence from several distinct
            # contacts before publishing a stable live ledger; status messages
            # keep startup visible while evidence accumulates.
            live_normalizer = LiveAttemptNormalizer(
                fps,
                write_attempt,
                minimum_cadence_hits=6,
                on_status=write_live_record if source.live else None,
            )
            pipeline_logger = None
            if source.live and live_events_output is not None:
                def write_pipeline_record(record: Dict[str, Any]) -> None:
                    live_events_output.write(json.dumps(record) + "\n")
                    live_events_output.flush()

                pipeline_logger = LivePipelineLogger(
                    fps, write_pipeline_record,
                )

            def write_pipeline_heartbeat(
                frame_number: int,
                metrics: Dict[str, Any],
            ) -> None:
                live_metrics = {
                    **metrics,
                    "attempt_upsert_count": attempt_upsert_count,
                    "attempt_count": len(attempt_states),
                    "pending_attempt_count": sum(
                        state == "pending" for state in attempt_states.values()
                    ),
                    "finalized_attempt_count": sum(
                        state == "finalized" for state in attempt_states.values()
                    ),
                }
                if pipeline_logger is not None:
                    pipeline_logger.observe(frame_number, live_metrics)
                if health_monitor is not None:
                    health_monitor.observe_pipeline(live_metrics)

            preview_interval_frames = max(1, round(fps / args.preview_fps))

            def write_preview_frame(
                frame_number: int,
                frame: np.ndarray,
            ) -> None:
                nonlocal live_stdout_open
                if (
                    not args.preview_stdout
                    or not live_stdout_open
                    or frame_number % preview_interval_frames
                ):
                    return
                encoded, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72],
                )
                if not encoded:
                    return
                record = {
                    "type": "_preview_frame",
                    "frame_number": frame_number,
                    "video_time_seconds": round(frame_number / fps, 3),
                    "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
                }
                try:
                    print(json.dumps(record, separators=(",", ":")), flush=True)
                except BrokenPipeError:
                    live_stdout_open = False
                    sys.stdout = open("/dev/null", "w", encoding="utf-8")

            def write_bounce_diagnostic(
                diagnostic: TrackDiagnostic, draw_frame: int,
            ) -> None:
                if track_diagnostics_output is not None:
                    track_diagnostics_output.write(json.dumps({
                        "draw_frame": draw_frame,
                        "kind": diagnostic.kind,
                        "reason": diagnostic.reason,
                        "points": diagnostic.points,
                    }) + "\n")
                    track_diagnostics_output.flush()
                if diagnostic.kind in (
                    "contact_candidate", "rejected_contact_candidate",
                ):
                    if contact_diagnostics_output is None:
                        return
                    record = json.loads(diagnostic.reason)
                    record["draw_frame"] = draw_frame
                    contact_diagnostics_output.write(
                        json.dumps(record) + "\n"
                    )
                    contact_diagnostics_output.flush()
                    return
                if bounce_diagnostics_output is None or diagnostic.kind not in (
                    "confirmed_bounce", "association",
                ):
                    return
                bounce_diagnostics_output.write(json.dumps({
                    "draw_frame": draw_frame,
                    "kind": diagnostic.kind,
                    "reason": diagnostic.reason,
                    "points": diagnostic.points,
                }) + "\n")
                bounce_diagnostics_output.flush()

            events = process_video(
                source, scale, calibration, homography, table,
                args.end_seconds, writer, first_frame,
                live_normalizer.observe, live_normalizer.settle_attempt,
                live_normalizer.observe_confirmed_hit,
                live_normalizer.observe_confirmed_non_hit,
                observe_processing_frame,
                clean_writer,
                round(args.clean_recording_seconds * fps),
                args.clean_recording_start == "launch",
                on_attempt_started=getattr(
                    live_normalizer, "observe_attempt_started", None,
                ),
                on_track_diagnostic=write_bounce_diagnostic,
                on_frame_processed=write_pipeline_heartbeat,
                on_preview_frame=(
                    write_preview_frame if args.preview_stdout else None
                ),
            )
            live_normalizer.finish_session(processing_frame)
            if pipeline_logger is not None:
                pipeline_logger.finish(processing_frame, "processing_ended")

            # Cadence-based normalization can rename events and infer missed
            # launches only after enough of the session is known. Preserve the
            # canonical analysis file independently of the live ledger log.
            output.seek(0)
            output.truncate()
            for event in events:
                output.write(json.dumps(event.to_record()) + "\n")
            output.flush()
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if clean_writer is not None:
            clean_writer.release()
        if live_events_output is not None:
            live_events_output.close()
        if bounce_diagnostics_output is not None:
            bounce_diagnostics_output.close()
        if contact_diagnostics_output is not None:
            contact_diagnostics_output.close()
        if track_diagnostics_output is not None:
            track_diagnostics_output.close()
    print(json.dumps({
        "events": len(events),
        "output": args.output,
        "annotated": annotated_path,
        "clean_recording": args.clean_recording,
        "live_events": args.live_events,
        "bounce_diagnostics": args.bounce_diagnostics,
        "contact_diagnostics": args.contact_diagnostics,
    }, indent=2), file=sys.stderr if args.live_stdout else sys.stdout)


if __name__ == "__main__":
    main()

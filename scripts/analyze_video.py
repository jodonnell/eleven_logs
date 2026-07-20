#!/usr/bin/env python3
"""Streaming bounce analysis for Eleven Table Tennis fixed spectator footage.

Uses only the current/previous frame and bounded trajectory history.  It is
deliberately conservative: an incomplete/occluded trajectory is unknown,
rather than a fabricated table coordinate.
"""
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install dependencies first: python3 -m pip install --user opencv-python-headless numpy") from exc

from auto_calibrate import calibration_from_frame
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
        "0": "0100/1011/1001/1011|1001/1001/1111|1011/1001/1111|1101/1001/1101",
        "1": "111/001/001|111/011/011",
        "2": "0011/0010/1100|0100/0011/0110/1100|0100/0011/0110/1110",
        "3": "011/110/011|100/011/110/011",
        "4": "0010/0110/1010/0011|0010/0110/1010/1011|0110/1010/1111",
        "5": "1000/1111/0011|1100/0111/0001|1110/1000/1111/0011|1110/1000/1111/1011",
        "6": "1000/1111/1011|1011/1110/1001|1100/1111/1001|01100/11000/11110/11110/01100",
        "7": "011/010/100|011/010/110|111/001/011/010",
        "8": "1011/1110/1001|1011/1110/1011|1101/0111/1101",
        "9": "0100/1011/1111/0010|1101/1111/0001|1011/1111/0011",
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
        return cls(**{name: value for name, value in configured.items() if name in valid})


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
    return_crossed_net: Optional[bool] = None
    hit: Optional[Dict[str, Any]] = None
    machine: Optional[Dict[str, Any]] = None

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record.pop("pixel")
        record.pop("draw_frame")
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


@dataclass
class Attempt:
    """Tracks one ball-machine launch and any possible return paths."""

    frame: int
    pixel: Point
    report_no_bounce: bool = True
    returns: List[Track] = field(default_factory=list)
    bounces: List[BounceEvent] = field(default_factory=list)
    bounce_track_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
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


def find_bounce(
    points: Track,
    table_polygon: np.ndarray,
    net_line: Optional[np.ndarray] = None,
    settings: DetectorSettings = DetectorSettings(),
    allow_terminal_shadow: bool = True,
) -> Optional[Bounce]:
    """Find a visible table-plane turn in one completed candidate track."""
    if len(points) < settings.min_track_points:
        return None
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
        rightward_contact = points[index][1] >= points[index - 1][1]
        terminal_confirmed = (
            index < len(points) - settings.terminal_shadow_frames
            or allow_terminal_shadow
        )
        if (
            local_peak
            and rightward_contact
            and terminal_confirmed
            and (not terminal or points[index][1] > points[index - 2][1])
        ):
            return points[index], points[index - 2:index], points[index + 1:index + 3]
    # Two post-contact frames are enough for a terminal turn when the ball
    # disappears behind the launcher immediately afterwards.
    best = None
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
        # Player returns travel toward increasing screen x in the supported
        # spectator views. A sudden backward x jump at the apparent turn is
        # a tracker hand-off to a marking/shadow, not a physical bounce.
        if points[index][1] < points[index - 1][1]:
            continue
        if maximum or minimum:
            strength = min(abs(y - before_mean), abs(y - after_mean))
            candidate = (strength, points[index], points[index - 3:index], points[index + 1:index + 3])
            if best is None or strength > best[0]:
                best = candidate
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
                candidate = (flattening, points[index], points[index - 3:index], points[index + 1:index + 3])
                if best is None or flattening > best[0]:
                    best = candidate
    return best[1:] if best else None


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
    maximum = int(row_counts.max())
    if maximum < max(8, frame.shape[1] * .008):
        return None
    rows = np.flatnonzero(row_counts >= maximum * .3)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    groups = [group for group in groups if len(group) >= 2]
    if not groups:
        return None
    title_rows = max(groups, key=lambda group: int(row_counts[group].sum()))
    y0, y1 = int(title_rows[0]), int(title_rows[-1] + 1)
    selected = blue[y0:y1] > 0
    _, xs = np.nonzero(selected)
    if not len(xs):
        return None
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    if x1 - x0 < frame.shape[1] * .05:
        return None
    return x0, y0, x1, y1


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
                candidate_scores.append(2 * intersection / total if total else 0.0)
            scores[digit] = max(candidate_scores)
        ranked = sorted(scores, key=scores.get, reverse=True)
        if len(ranked) > 1 and scores[ranked[0]] - scores[ranked[1]] < .015:
            return "?", 0.0
        digit = ranked[0]
        return digit, round(scores[digit], 3)
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
        start, end = max(1, expected - radius), min(width - 1, expected + radius + 1)
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
        top, bottom, right, needs_decimal = 4.5, 5.7, .59, False
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
    speed = read_hud_number(frame, bounds, "speed")
    spin = read_hud_number(frame, bounds, "spin")
    direction = read_spin_direction(frame, bounds)
    if speed is None or spin is None or direction is None:
        return None
    # At low capture resolutions, unit text or compression artifacts can be
    # mistaken for another digit (for example, "51 rev/s" becoming 517).
    # Reject readings outside Eleven's displayed range instead of attaching a
    # confidently repeated but physically bogus value to every attempt.
    if (
        not 0 < speed < 100
        or not 0 <= spin <= MAX_SPIN_REVOLUTIONS_PER_SECOND
    ):
        return None
    return TelemetryReading(frame_number, float(speed), int(spin), direction)


class TelemetryReader:
    """Debounce repeated HUD OCR into timestamped screen state changes."""

    def __init__(self, stable_samples: int = 2) -> None:
        self.stable_samples = stable_samples
        self.candidate: Optional[TelemetryReading] = None
        self.candidate_count = 0
        self.latest: Optional[TelemetryReading] = None
        self.bounds: Optional[Tuple[int, int, int, int]] = None

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
        reading = read_telemetry(frame, frame_number, self.bounds)
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

def shadow_contact_score(hsv: np.ndarray, center: Point) -> float:
    """Local green-table darkening directly below a bright-ball candidate."""
    x, y = map(round, center)
    height, width = hsv.shape[:2]
    local = hsv[max(0, y + 5):min(height, y + 28), max(0, x - 18):min(width, x + 19)]
    surrounding = hsv[max(0, y - 35):min(height, y + 36), max(0, x - 35):min(width, x + 36)]
    def green_values(region: np.ndarray) -> np.ndarray:
        if region.size == 0:
            return np.array([])
        mask = (region[:, :, 0] >= 42) & (region[:, :, 0] <= 88) & (region[:, :, 1] >= 80)
        return region[:, :, 2][mask]
    dark, baseline = green_values(local), green_values(surrounding)
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
    # White ball: very bright and low saturation. Difference rejects static
    # white markings/text/net edges without retaining a background frame.
    bright = cv2.inRange(hsv, settings.bright_ball_lower, settings.bright_ball_upper)
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
        choices.append((area, center, shadow_contact_score(hsv, center)))
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
            "return": (40, 220, 40),
            "confirmed_bounce": (0, 0, 255),
        }
        for candidate in diagnostics.candidates:
            center = tuple(map(round, candidate.center))
            if candidate.kind == "raw":
                cv2.circle(view, center, 2, (190, 190, 190), 1)
                continue
            cv2.drawMarker(view, center, colors["rejected"], cv2.MARKER_TILTED_CROSS, 8, 1)
            if candidate.reason:
                cv2.putText(
                    view, candidate.reason, (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .32, colors["rejected"], 1,
                )
        for path in diagnostics.unconfirmed_tracks:
            points = np.int32([(point[1], point[2]) for point in path])
            if len(points) >= 2:
                cv2.polylines(view, [points], False, (255, 255, 0), 1)
        current_frame = frame_number if frame_number is not None else 0
        for completed in diagnostics.visible_completed_tracks(current_frame):
            color = colors[completed.kind]
            points = np.int32([(point[1], point[2]) for point in completed.points])
            if len(points) >= 2:
                cv2.polylines(view, [points], False, color, 2)
            if len(points):
                label = completed.kind
                if completed.reason:
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
            ("return", (40, 220, 40)),
            ("confirmed bounce", (0, 0, 255)),
        )
        cv2.rectangle(view, (8, 39), (180, 151), (20, 20, 20), -1)
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
        on_attempt_finished: Optional[Callable[[], None]] = None,
        on_confirmed_hit: Optional[Callable[[BounceEvent], None]] = None,
        on_track_diagnostic: Optional[Callable[[TrackDiagnostic, int], None]] = None,
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
        self.on_track_diagnostic = on_track_diagnostic
        self.events: List[BounceEvent] = []
        self.emitted: Set[Tuple[int, int]] = set()
        self.active_attempt: Optional[Attempt] = None
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
        self.launch_direction = 1 if return_center_x > launcher_center_x else -1
        self.warmup_launcher_tracks = calibration.get("warmup_launcher_tracks", 0)

    def diagnose_track(
        self, path: Track, kind: str, draw_frame: int, reason: str = "",
    ) -> None:
        if self.on_track_diagnostic is not None:
            self.on_track_diagnostic(TrackDiagnostic(path, kind, reason), draw_frame)

    def emit(self, event: BounceEvent) -> None:
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

    def telemetry_pair_before(
        self, frame: int,
    ) -> Tuple[Optional[TelemetryReading], Optional[TelemetryReading]]:
        """Return (player hit, preceding machine delivery) at a landing."""
        readings = [item for item in self.telemetry_history if item.frame_number <= frame]
        if not readings:
            return None, None
        hit = readings[-1]
        machine = readings[-2] if len(readings) >= 2 else None
        return hit, machine

    def launcher_rejection_reason(self, path: Track) -> Optional[str]:
        """Explain why a completed path cannot establish a machine launch."""
        start = (path[0][1], path[0][2])
        if not point_in_rectangle(start, self.launcher_region, self.scale):
            return "did not begin near launcher"
        if len(path) < self.settings.min_launch_track_points:
            return f"launch too short ({len(path)}/{self.settings.min_launch_track_points})"

        directed_steps = [
            (end[1] - beginning[1]) * self.launch_direction
            for beginning, end in zip(path, path[1:])
        ]
        directed_distance = (path[-1][1] - path[0][1]) * self.launch_direction
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
        terminal_x = path[-1][1]
        return_direction = -self.launch_direction
        for index, point in enumerate(path):
            if (
                point_in_rectangle((point[1], point[2]), self.return_region, self.scale)
                and (terminal_x - point[1]) * return_direction
                >= self.settings.return_min_horizontal_distance
            ):
                return path[index:]
        return None

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

    def start_attempt(self, path: Track, draw_frame: int) -> None:
        self.launcher_tracks_seen += 1
        if self.launcher_tracks_seen <= self.warmup_launcher_tracks:
            return
        finished_previous = self.active_attempt is not None
        self.finish_attempt(draw_frame)
        if finished_previous and self.on_attempt_finished is not None:
            self.on_attempt_finished()
        self.active_attempt = Attempt(
            path[0][0], (path[0][1], path[0][2]),
            report_no_bounce=self.is_reportable_launcher_track(path),
            machine_telemetry=self.telemetry_near(path[0][0]),
        )

    def return_evidence(self, path: Track) -> Tuple[bool, bool]:
        start_pixel = (path[0][1], path[0][2])
        terminal_pixel = (path[-1][1], path[-1][2])
        crossed_net = (
            signed_distance_to_line(start_pixel, self.net_line)
            * signed_distance_to_line(terminal_pixel, self.net_line)
            <= 0
        )
        return crossed_net, not point_in_polygon(terminal_pixel, self.table)

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
            return_crossed_net=bool(crossed_net) if attempt.returns else None,
            machine=(
                attempt.machine_telemetry.to_record(self.fps)
                if attempt.machine_telemetry else None
            ),
        )

    def finish_attempt(self, draw_frame: int) -> None:
        if self.active_attempt is None:
            return
        if self.active_attempt.bounces:
            for event in sorted(self.active_attempt.bounces, key=lambda item: item.frame_number):
                self.emit(event)
            last_bounce_frame = max(event.frame_number for event in self.active_attempt.bounces)
            later_misses = [
                path for path in self.active_attempt.returns
                if self.track_key(path) not in self.active_attempt.bounce_track_keys
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
        elif self.active_attempt.report_no_bounce or any(
            all(self.return_evidence(path)) for path in self.active_attempt.returns
        ):
            self.emit(self.no_bounce_event(self.active_attempt, draw_frame))
        self.active_attempt = None

    def add_bounce(
        self,
        path: Track,
        hit: TrackPoint,
        approach: Track,
        departure: Track,
        draw_frame: int,
    ) -> None:
        if self.active_attempt is None:
            return
        key = (path[0][0], hit[0])
        if key in self.emitted:
            return
        self.emitted.add(key)
        self.diagnose_track(path, "confirmed_bounce", draw_frame)
        pixel = (hit[1], hit[2])
        in_occlusion = len(self.occlusion) > 2 and point_in_polygon(pixel, self.occlusion)
        posx, posy, posz = map_log_coordinate(self.homography, pixel, self.calibration["table_surface_y"])
        far = posz > 0.03
        continuity = min(1.0, len(approach + departure) / 6)
        confidence = round((0.82 if far else 0.72) * continuity * (0.45 if in_occlusion else 1.0), 2)
        outcome = "unknown" if in_occlusion else ("far_table" if far else "near_table")
        hit_telemetry, machine_telemetry = self.telemetry_pair_before(hit[0])
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
            hit=(
                hit_telemetry.to_record(self.fps) if hit_telemetry else None
            ),
            machine=(
                machine_telemetry.to_record(self.fps) if machine_telemetry else None
            ),
        )
        self.active_attempt.bounces.append(event)
        self.active_attempt.bounce_track_keys.add(self.track_key(path))
        if event.hit_table and event.outcome == "far_table" and self.on_confirmed_hit:
            self.on_confirmed_hit(event)

    def process_tracks(self, tracks: Sequence[Track], draw_frame: int) -> None:
        for path in tracks:
            if len(path) < self.settings.min_track_points:
                self.diagnose_track(
                    path, "rejected", draw_frame,
                    f"too short ({len(path)}/{self.settings.min_track_points})",
                )
                continue
            if self.is_launcher_track(path):
                self.diagnose_track(path, "launcher", draw_frame)
                self.start_attempt(path, draw_frame)
                continue
            attempt = self.active_attempt
            if attempt is None:
                reason = self.launcher_rejection_reason(path) or "no active launch"
                self.diagnose_track(path, "rejected", draw_frame, reason)
                continue
            returned = self.return_segment(path, attempt)
            if returned is None:
                self.diagnose_track(
                    path, "rejected", draw_frame,
                    self.return_rejection_reason(path, attempt)
                    or "not a plausible return",
                )
                continue
            path = returned
            self.diagnose_track(path, "return", draw_frame)
            attempt.returns.append(path)
            bounce = find_bounce(path, self.table, self.net_line, self.settings)
            if bounce:
                self.add_bounce(path, *bounce, draw_frame)
                continue
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
                    )

    def process_active_tracks(
        self, tracks: Sequence[Track], draw_frame: int,
    ) -> None:
        """Report a visible bounce without waiting for its track to disappear.

        Completed-track processing remains authoritative for attempts, misses,
        and diagnostics. Here we only act on a return associated with the
        current launch and require post-contact evidence for a shadow peak;
        an apparent contact on the newest frame may still become a plateau.
        """
        if self.active_attempt is None:
            return
        for path in tracks:
            returned = self.return_segment(path, self.active_attempt)
            if returned is None:
                continue
            bounce = find_bounce(
                returned, self.table, self.net_line, self.settings,
                allow_terminal_shadow=False,
            )
            if bounce:
                self.add_bounce(returned, *bounce, draw_frame)


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
    events: Sequence[BounceEvent], total_frames: int, fps: float,
) -> Tuple[Optional[float], List[Tuple[int, BounceEvent]]]:
    """Build the canonical cadence slots used by live and final output."""
    hits = [event for event in events if event.hit_table and event.outcome == "far_table"]
    period = infer_attempt_period([event.frame_number for event in hits], fps)
    if period is None:
        return None, []

    phase = hits[0].frame_number
    signed = [
        (event.frame_number - phase + period / 2) % period - period / 2
        for event in hits
    ]
    phase += sorted(signed)[len(signed) // 2]
    while phase - period >= period * .5:
        phase -= period
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
        if current is None or event.confidence > current.confidence:
            slots[slot] = replace(event, outcome="hit")
        hit_slots[id(event)] = slot

    cursor = -1
    for event in events:
        if id(event) in hit_slots:
            cursor = max(cursor, hit_slots[id(event)])
            continue
        if event.outcome != "off_table":
            continue
        candidates = [index for index in range(cursor + 1, len(slots)) if slots[index] is None]
        if not candidates:
            continue
        slot = candidates[0]
        slots[slot] = replace(event, outcome="out")
        cursor = slot

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
        return list(events)
    return [event for _, event in slots]


class LiveAttemptNormalizer:
    """Emit settled canonical slots while retaining batch finalization.

    The first three confirmed opponent-table contacts are buffered because
    fewer observations cannot establish a cadence. Once warm, a slot is held
    until a result in a later inferred slot is observed, allowing the following
    launch to finish the preceding attempt before a miss can be emitted.
    """

    def __init__(self, fps: float, on_event: Callable[[BounceEvent], None]) -> None:
        self.fps = fps
        self.on_event = on_event
        self.events: List[BounceEvent] = []
        self.period: Optional[float] = None
        self.emitted_anchors: List[int] = []
        self.immediate_event_frames: List[int] = []
        self.pending_attempt_events: List[BounceEvent] = []
        self.settlement_frame: Optional[float] = None

    def observe(self, event: BounceEvent) -> None:
        self.events.append(event)
        self.pending_attempt_events.append(event)

    def publish_finished_attempt(self) -> None:
        """Publish a non-hit as soon as the next launch closes its attempt."""
        pending = self.pending_attempt_events
        self.pending_attempt_events = []
        if not pending or any(
            event.hit_table and event.outcome == "far_table"
            for event in pending
        ):
            return
        event = max(
            pending,
            key=lambda item: (
                item.outcome == "off_table",
                item.outcome == "net",
                item.confidence,
            ),
        )
        outcome = "out" if event.outcome == "off_table" else "miss"
        self.on_event(replace(event, outcome=outcome))
        self.immediate_event_frames.append(event.frame_number)

    def observe_confirmed_hit(self, event: BounceEvent) -> None:
        """Publish direct visual evidence without waiting for cadence."""
        if any(
            abs(event.frame_number - frame) <= self.fps * .5
            for frame in self.immediate_event_frames
        ):
            return
        if self.period is not None and any(
            abs(event.frame_number - anchor) <= self.period * .3
            for anchor in self.emitted_anchors
        ):
            return
        self.on_event(replace(event, outcome="hit"))
        self.immediate_event_frames.append(event.frame_number)

    def settle_attempt(self) -> None:
        """Advance once after a detected launch closes the prior attempt."""
        self.publish_finished_attempt()
        if not self.events:
            return
        hits = [
            event for event in self.events
            if event.hit_table and event.outcome == "far_table"
        ]
        self.period = infer_attempt_period(
            [event.frame_number for event in hits], self.fps,
        )
        if self.period is None:
            return
        newest_evidence = max(event.frame_number for event in self.events)
        if self.settlement_frame is None:
            self.settlement_frame = newest_evidence + self.period
        else:
            self.settlement_frame = max(
                self.settlement_frame + self.period,
                newest_evidence + self.period,
            )
        self.period, slots = attempt_event_slots(
            self.events, round(self.settlement_frame) + 1, self.fps,
        )
        assert self.period is not None
        for anchor, event in slots:
            if any(
                abs(anchor - frame) <= self.period * .3
                or abs(event.frame_number - frame) <= self.fps * .5
                for frame in self.immediate_event_frames
            ) and not any(
                abs(anchor - emitted) <= self.period * .5
                for emitted in self.emitted_anchors
            ):
                self.emitted_anchors.append(anchor)
        for anchor, event in slots:
            if anchor + self.period > self.settlement_frame:
                continue
            if any(
                abs(anchor - emitted) <= self.period * .5
                for emitted in self.emitted_anchors
            ):
                continue
            self.on_event(event)
            self.emitted_anchors.append(anchor)

    def finalize(self, total_frames: int) -> List[BounceEvent]:
        return normalize_attempt_events(self.events, total_frames, self.fps)


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
    path: PathLike, fps: float, size: Tuple[int, int]
) -> cv2.VideoWriter:
    """Create an annotated-video writer or fail before processing begins."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
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
    on_attempt_finished: Optional[Callable[[], None]] = None,
    on_confirmed_hit: Optional[Callable[[BounceEvent], None]] = None,
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
        if writer is not None else None
    )
    telemetry = TelemetryReader()
    classifier = AttemptClassifier(
        fps, calibration, contact_polygon, net_line, occlusion, homography,
        video_width, video_height, scale, settings, on_event,
        on_attempt_finished, on_confirmed_hit,
        diagnostics.completed_track if diagnostics is not None else None,
    )
    previous_gray = None
    next_frame = first_frame
    frame_number = first_frame.number if first_frame is not None else 0
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
            if frame_number % 3 == 0:
                reading = telemetry.update(original, frame_number)
                if reading is not None:
                    classifier.observe_telemetry(reading)
            frame = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
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
            if writer is not None:
                writer.write(draw_overlay(
                    frame, table, net_line, tracker.visible_points, classifier.events,
                    homography, calibration["table_surface_y"],
                    diagnostics, frame_number,
                ))
            frame_number += 1
    except KeyboardInterrupt:
        # A live source normally ends when the user stops it. Preserve and
        # flush the completed session instead of discarding every event.
        pass
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
    parser.add_argument("--no-annotated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-seconds", type=float, default=0, help="seek point; useful when reviewing a short interval")
    parser.add_argument("--end-seconds", type=float, help="stop after this video timestamp")
    args = parser.parse_args()
    annotated_path = None if args.no_annotated else args.annotated
    if not args.extract_calibration_frame:
        # Live SRT sources can block while waiting for the sender. Clear stale
        # events before opening the source so observers immediately see that a
        # new server session has begun.
        reset_output_file(args.output)
    try:
        source = open_video_source(args.video)
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
            calibration, homography, table = calibration_geometry(
                detected, video_width, video_height, scale,
            )
        if annotated_path is not None:
            writer = create_video_writer(annotated_path, fps, (width, height))
        with open(args.output, "w", encoding="utf-8") as output:
            def write_event(event: BounceEvent) -> None:
                serialized = json.dumps(event.to_record())
                output.write(serialized + "\n")
                output.flush()
                if args.live_stdout:
                    print(serialized, flush=True)

            live_normalizer = LiveAttemptNormalizer(fps, write_event)
            events = process_video(
                source, scale, calibration, homography, table,
                args.end_seconds, writer, first_frame,
                live_normalizer.observe, live_normalizer.settle_attempt,
                live_normalizer.observe_confirmed_hit,
            )

            # Cadence-based normalization can rename events and infer missed
            # launches only after enough of the session is known. Replace the
            # live snapshot with that canonical final result before exiting.
            output.seek(0)
            output.truncate()
            for event in events:
                output.write(json.dumps(event.to_record()) + "\n")
            output.flush()
    finally:
        source.close()
        if writer is not None:
            writer.release()
    print(json.dumps({
        "events": len(events),
        "output": args.output,
        "annotated": annotated_path,
    }, indent=2), file=sys.stderr if args.live_stdout else sys.stdout)


if __name__ == "__main__":
    main()

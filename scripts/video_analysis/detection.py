"""Geometry helpers, bounce qualification, and multi-ball tracking."""

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

from .models import (
    ActiveTrack,
    Bounce,
    Calibration,
    Candidate,
    DetectorSettings,
    PathLike,
    Point,
    Track,
    TrackPoint,
)

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
        image = np.asarray(
            [point["image"] for point in data["control_points"]], dtype=np.float32,
        ) * scale
        log = np.asarray(
            [point["log"] for point in data["control_points"]], dtype=np.float32,
        )
    else:
        names = ("far_left", "far_right", "near_right", "near_left")
        image = np.asarray(
            [data["image_corners"][name] for name in names], dtype=np.float32,
        ) * scale
        log = np.asarray(
            [data["log_corners"][name] for name in names], dtype=np.float32,
        )
    table_polygon = np.asarray(data["table_polygon"], dtype=np.float32) * scale
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

"""Frame candidate extraction, coordinate mapping, and diagnostic drawing."""

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
    BounceEvent,
    Candidate,
    DetectorDiagnostics,
    DetectorSettings,
    Point,
    TrackPoint,
)
from .detection import point_in_polygon

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
        center: Point = (float(centers[i][0]), float(centers[i][1]))
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
    source = np.asarray([[image_point]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(source, homography)[0][0]
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
    poly = table.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(view, [poly], True, (0, 255, 255), 3)
    cv2.line(view, tuple(map(int, net_line[0])), tuple(map(int, net_line[1])), (255, 0, 255), 3)
    # Calibration grid: x is across the table width; z is player(-) to
    # opponent(+). This makes a bad corner/axis calibration obvious before
    # any bounce coordinates are trusted.
    inverse_homography = np.linalg.inv(homography)
    for z in (-1.37, -0.685, 0.0, 0.685, 1.37):
        line = np.asarray([[[-0.7625, z]], [[0.7625, z]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(line, inverse_homography).reshape(-1, 2)
        cv2.line(view, tuple(map(int, projected[0])), tuple(map(int, projected[1])), (80, 160, 255), 1)
    for x in (-0.7625, -0.38125, 0.0, 0.38125, 0.7625):
        line = np.asarray([[[x, -1.37]], [[x, 1.37]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(line, inverse_homography).reshape(-1, 2)
        cv2.line(view, tuple(map(int, projected[0])), tuple(map(int, projected[1])), (80, 160, 255), 1)
    origin = np.asarray([[[0.0, 0.0]]], dtype=np.float32)
    center = cv2.perspectiveTransform(origin, inverse_homography)[0][0]
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
            points = np.asarray(
                [(point[1], point[2]) for point in path], dtype=np.int32,
            )
            if len(points) >= 2:
                cv2.polylines(view, [points], False, (255, 255, 0), 1)
        current_frame = frame_number if frame_number is not None else 0
        for completed in diagnostics.visible_completed_tracks(current_frame)[-8:]:
            color = colors[completed.kind]
            points = np.asarray(
                [(point[1], point[2]) for point in completed.points], dtype=np.int32,
            )
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

#!/usr/bin/env python3
"""Explicitly export a per-camera table calibration from one video frame.

The normal analyser detects this geometry in memory. This separate utility is
for visual diagnostics and manually reviewed overrides.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from video_source import VideoSourceError, open_video_source


TABLE_HALF_WIDTH = 0.7625
TABLE_HALF_LENGTH = 1.37
CALIBRATION_WIDTH = 1024

PathLike = Union[str, Path]
Line = Tuple[float, float, float, float]
Segment = Tuple[float, float, Line]
Geometry = Tuple[List[List[float]], Line, Tuple[Line, Line]]
CalibrationReport = dict[str, Any]
Calibration = dict[str, Any]


def hue_distance(hues: np.ndarray, center: float) -> np.ndarray:
    """Circular OpenCV-HSV hue distance (the hue axis wraps at 180)."""
    difference = np.abs(hues.astype(np.float32) - center)
    return np.minimum(difference, 180.0 - difference)


def colored_table_extent(
    frame: np.ndarray,
) -> Tuple[int, int, np.ndarray, dict[str, float]]:
    """Locate the dominant large, saturated surface without assuming its hue."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    legacy_mask = cv2.inRange(
        hsv, np.array([35, 45, 30]), np.array([95, 255, 255]),
    )
    legacy_count, legacy_labels, legacy_stats, _ = cv2.connectedComponentsWithStats(
        legacy_mask,
    )
    if legacy_count > 1:
        component_areas = legacy_stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(np.max(component_areas))
        substantial = 1 + np.flatnonzero(component_areas >= largest_area * .05)
        centered_lower_surface = any(
            legacy_stats[label, cv2.CC_STAT_LEFT] <= frame.shape[1] / 2
            <= legacy_stats[label, cv2.CC_STAT_LEFT] + legacy_stats[label, cv2.CC_STAT_WIDTH]
            and legacy_stats[label, cv2.CC_STAT_TOP] + legacy_stats[label, cv2.CC_STAT_HEIGHT]
            >= frame.shape[0] * .45
            for label in substantial
        )
        if centered_lower_surface:
            selected = np.uint8(np.isin(legacy_labels, substantial)) * 255
            row_counts = np.count_nonzero(selected, axis=1)
            rows = np.flatnonzero(row_counts >= frame.shape[1] * .05)
            if len(rows) >= 20:
                return int(rows[0]), int(rows[-1]), selected, {
                    "hue_center": 65.0,
                    "hue_tolerance": 23.0,
                    "min_saturation": 80.0,
                    "min_value": 30.0,
                }

    eligible = (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 30)
    if np.count_nonzero(eligible) < frame.shape[0] * frame.shape[1] * .01:
        raise ValueError("could not find a sufficiently large colored table surface")

    histogram = np.bincount(hsv[:, :, 0][eligible], minlength=180)
    smoothed = np.zeros(180, dtype=np.int64)
    for offset in range(-6, 7):
        smoothed += np.roll(histogram, offset)

    best: Optional[Tuple[float, np.ndarray, float]] = None
    # Trying several distinct peaks prevents a large blue window or wall accent
    # from displacing a differently colored tabletop.
    peaks: List[int] = []
    for candidate in np.argsort(smoothed)[::-1]:
        hue = int(candidate)
        if all(min(abs(hue - prior), 180 - abs(hue - prior)) > 10 for prior in peaks):
            peaks.append(hue)
        if len(peaks) == 8:
            break
    kernel = np.ones((5, 5), np.uint8)
    for hue in peaks:
        mask = np.uint8(
            eligible & (hue_distance(hsv[:, :, 0], hue) <= 10)
        ) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            if (
                area < frame.shape[0] * frame.shape[1] * .005
                or width < frame.shape[1] * .15
                or height < frame.shape[0] * .05
            ):
                continue
            component_mask = labels == component
            # The table is a broad coherent surface centered in a deliberate
            # spectator composition. A strong center-crossing preference
            # rejects large same-colored walls, windows, floors, and furniture.
            crosses_center = x <= frame.shape[1] / 2 <= x + width
            score = area * (2.5 if crosses_center else 1.0)
            if x <= 1 or y <= 1 or x + width >= frame.shape[1] - 1 or y + height >= frame.shape[0] - 1:
                score *= .25
            score += round(area * (y + height / 2) / frame.shape[0] * .1)
            if best is None or score > best[0]:
                component_hues = hsv[:, :, 0][component_mask]
                best = (score, component_mask, float(np.median(component_hues)))
    if best is None:
        raise ValueError("could not find a sufficiently large colored table surface")

    _, component_mask, center = best
    component_pixels = hsv[component_mask]
    profile = {
        "hue_center": round(center, 1),
        "hue_tolerance": 14.0,
        "min_saturation": float(max(35, np.percentile(component_pixels[:, 1], 5) - 20)),
        "min_value": float(max(20, np.percentile(component_pixels[:, 2], 5) - 30)),
    }
    mask = np.uint8(
        (hue_distance(hsv[:, :, 0], center) <= profile["hue_tolerance"])
        & (hsv[:, :, 1] >= profile["min_saturation"])
        & (hsv[:, :, 2] >= profile["min_value"])
    ) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    seed_labels = np.unique(labels[component_mask])
    selected = [int(label) for label in seed_labels if label]
    mask = np.uint8(np.isin(labels, selected)) * 255
    row_counts = np.count_nonzero(mask, axis=1)
    rows = np.flatnonzero(row_counts >= frame.shape[1] * .05)
    if len(rows) < 20:
        raise ValueError("could not find a sufficiently large colored table surface")
    return int(rows[0]), int(rows[-1]), mask, profile


def infer_ball_color(
    frames: Iterable[np.ndarray], calibration: Calibration,
    diagnostics: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, float]]:
    """Infer a saturated moving ball hue from a short calibrated video prefix."""
    source_width, source_height = calibration["image_size"]
    scale = min(1.0, CALIBRATION_WIDTH / source_width)
    width, height = round(source_width * scale), round(source_height * scale)
    tracking = np.asarray(calibration["tracking_polygon"], dtype=np.float32) * scale
    table = np.asarray(calibration["table_polygon"], dtype=np.float32) * scale
    region = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(region, [tracking.astype(np.int32)], 255)
    # Player hands and paddles occupy the lower exterior of the corridor. Ball
    # calibration needs the tabletop and the air above it, not that foreground.
    table_bottom = int(np.max(table[:, 1]))
    region[min(height, table_bottom + round((np.max(table[:, 1]) - np.min(table[:, 1])) * .2)):] = 0

    table_color = calibration["table_color"]
    table_hue = float(table_color["hue_center"])
    table_tolerance = float(table_color["hue_tolerance"])
    controls = {item["name"]: item["image"] for item in calibration["control_points"]}
    player_control = next(
        value for name, value in controls.items() if name == "x0_player_edge"
    )
    opponent_control = next(
        value for name, value in controls.items() if name == "x0_opponent_edge"
    )
    player = np.asarray(player_control, dtype=np.float64) * scale
    opponent = np.asarray(opponent_control, dtype=np.float64) * scale
    launch_axis = player - opponent
    launch_distance = float(np.linalg.norm(launch_axis))
    launch_direction = launch_axis / max(1.0, launch_distance)
    previous_gray: Optional[np.ndarray] = None
    colored_tracks: List[dict[str, Any]] = []
    white_tracks: List[dict[str, Any]] = []

    def components(
        mask: np.ndarray, hsv: np.ndarray, colored: bool,
    ) -> List[Tuple[Optional[int], float, float, float, float]]:
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        result = []
        for component in range(1, count):
            _, _, component_width, component_height, area = map(int, stats[component])
            if not 2 <= area <= 250:
                continue
            if max(component_width, component_height) / max(1, min(component_width, component_height)) > 2.5:
                continue
            if area / max(1, component_width * component_height) < .35:
                continue
            pixels = hsv[labels == component]
            hue = int(round(float(np.median(pixels[:, 0])))) % 180 if colored else None
            saturation = float(np.median(pixels[:, 1]))
            value = float(np.median(pixels[:, 2]))
            center_x, center_y = map(float, centers[component])
            result.append((hue, center_x, center_y, saturation, value))
        return result

    def update_tracks(
        tracks: List[dict[str, Any]],
        candidates: List[Tuple[Optional[int], float, float, float, float]],
        frame_number: int,
    ) -> None:
        available = {
            index for index, track in enumerate(tracks)
            if frame_number - track["last_frame"] <= 2
        }
        for hue, x, y, saturation, value in candidates:
            compatible = []
            for index in available:
                track = tracks[index]
                if hue is not None and _angle_distance(float(hue), track["hue"]) > 5:
                    continue
                distance = float(np.hypot(x - track["x"], y - track["y"]))
                if distance <= 100:
                    compatible.append((distance, index))
            if compatible:
                _, index = min(compatible)
                available.remove(index)
                track = tracks[index]
            else:
                track = {
                    "hue": float(hue or 0), "points": [], "samples": [],
                }
                tracks.append(track)
            track["points"].append((frame_number, x, y))
            track["samples"].append((saturation, value))
            track.update({"last_frame": frame_number, "x": x, "y": y})

    frame_number = 0

    for original in frames:
        frame = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            previous_gray = gray
            continue
        moving = cv2.threshold(
            cv2.absdiff(gray, previous_gray), 18, 255, cv2.THRESH_BINARY,
        )[1]
        previous_gray = gray
        colored = (
            (hsv[:, :, 1] >= 160)
            & (hsv[:, :, 2] >= 140)
            & (hue_distance(hsv[:, :, 0], table_hue) > table_tolerance + 5)
        )
        mask = cv2.bitwise_and(np.uint8(colored) * 255, moving)
        mask = cv2.bitwise_and(mask, region)
        update_tracks(colored_tracks, components(mask, hsv, True), frame_number)

        white_mask = np.uint8(
            (hsv[:, :, 1] <= 145) & (hsv[:, :, 2] >= 210)
        ) * 255
        white_mask = cv2.bitwise_and(white_mask, moving)
        white_mask = cv2.bitwise_and(white_mask, region)
        update_tracks(white_tracks, components(white_mask, hsv, False), frame_number)
        frame_number += 1

    def qualified(track: dict[str, Any]) -> Optional[float]:
        points = track["points"]
        if len(points) < 4:
            return None
        xs = [point[1] for point in points]
        ys = [point[2] for point in points]
        span = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
        steps = [
            float(np.hypot(right[1] - left[1], right[2] - left[2]))
            for left, right in zip(points, points[1:])
        ]
        median_step = float(np.median(steps)) if steps else 0.0
        start = np.asarray([points[0][1], points[0][2]], dtype=np.float64)
        progress = max(
            float(np.dot(
                np.asarray([point[1], point[2]], dtype=np.float64) - start,
                launch_direction,
            ))
            for point in points
        )
        if (
            span < 40
            or median_step < 2.5
            or np.linalg.norm(start - opponent) > max(120.0, launch_distance * .55)
            or progress < 30
        ):
            return None
        return span * min(len(points), 30) * min(median_step, 30) * min(progress / 30, 3)

    colored_ranked = [
        (score, track) for track in colored_tracks
        if (score := qualified(track)) is not None
    ]
    white_ranked = [
        score for track in white_tracks
        if (score := qualified(track)) is not None
    ]
    if not colored_ranked:
        if diagnostics is not None:
            diagnostics.update({"result": "no_colored_track"})
        return None
    colored_score, best = max(colored_ranked, key=lambda item: item[0])
    white_score = max(white_ranked, default=0.0)
    if diagnostics is not None:
        diagnostics.update({
            "result": "colored" if colored_score > white_score * 1.1 else "white",
            "colored_hue": round(float(best["hue"]), 1),
            "colored_score": round(float(colored_score), 1),
            "white_score": round(float(white_score), 1),
            "colored_tracks": len(colored_ranked),
            "white_tracks": len(white_ranked),
        })
    if colored_score <= white_score * 1.1:
        return None
    center = round(float(best["hue"])) % 180
    supporting = best["samples"]
    saturations = np.asarray([item[0] for item in supporting])
    values = np.asarray([item[1] for item in supporting])
    return {
        "hue_center": float(center),
        "hue_tolerance": 9.0,
        "min_saturation": float(max(120, np.percentile(saturations, 25) - 20)),
        "min_value": float(max(120, np.percentile(values, 25) - 20)),
        "confidence": round(float(colored_score / max(1.0, colored_score + white_score)), 3),
        "evidence_score": round(float(colored_score), 1),
        "white_evidence_score": round(float(white_score), 1),
    }


def calibrated_tracking_regions(
    image_size: List[int],
    table_polygon: List[List[float]],
    player_edge: Tuple[float, float],
    opponent_edge: Tuple[float, float],
) -> dict[str, Any]:
    """Build conservative ball-flight regions from camera-specific geometry.

    The centre stripe endpoints establish which image side belongs to the
    player and launcher.  Region sizes are proportional to the visible table,
    so a wide room view does not admit windows, screens, walls, or floor merely
    because they happen to contain small moving white blobs.
    """
    width, height = image_size
    table = np.asarray(table_polygon, dtype=np.float32)
    _, table_top = np.min(table, axis=0)
    _, table_bottom = np.max(table, axis=0)
    table_height = max(1.0, float(table_bottom - table_top))
    direction = 1.0 if opponent_edge[0] >= player_edge[0] else -1.0

    # Start classification already requires a long directed path. Keep the
    # side zones vertically permissive so high-spin arcs and low edge paths
    # are not lost; the flight corridor performs the room-background trim.
    start_top = 0.0
    start_bottom = float(height - 1)

    # Preserve generous start coverage at both ends of the flight. Which end
    # receives each role comes from calibration, rather than assuming that
    # every spectator camera puts the launcher on image-right.
    if direction > 0:
        launcher = [width * .58, start_top, float(width - 1), start_bottom]
        returned = [0.0, start_top, width * .28, start_bottom]
    else:
        launcher = [0.0, start_top, width * .42, start_bottom]
        returned = [width * .72, start_top, float(width - 1), start_bottom]

    # The flight can extend beyond either rail in x, especially on wide-angle
    # returns. Vertically, however, calibrated table scale gives a reliable
    # way to reject ceiling/window shimmer and floor motion without clipping
    # high arcs or paths that continue briefly below an edge contact.
    corridor_top = max(0.0, float(table_top - table_height * 1.5))
    corridor_bottom = min(float(height - 1), float(table_bottom + table_height * .5))
    corridor = [
        [0.0, corridor_top], [float(width - 1), corridor_top],
        [float(width - 1), corridor_bottom], [0.0, corridor_bottom],
    ]
    return {
        "launcher_region": [float(value) for value in launcher],
        "return_region": [float(value) for value in returned],
        "tracking_polygon": corridor,
        "table_contact_polygon": [
            [float(x), float(y)] for x, y in table_polygon
        ],
    }


def line_at_y(line: Line, y: float) -> Optional[float]:
    x1, y1, x2, y2 = line
    if abs(y2 - y1) < 1e-6:
        return None
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


def hough_segments(frame: np.ndarray) -> List[Segment]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 50, minLineLength=80, maxLineGap=20)
    result = []
    for raw in lines if lines is not None else []:
        x1, y1, x2, y2 = map(float, raw.reshape(-1))
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        result.append((length, angle, (x1, y1, x2, y2)))
    return result


def _quad(points: np.ndarray) -> Optional[np.ndarray]:
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    approximation = cv2.approxPolyDP(hull, .02 * perimeter, True).reshape(-1, 2)
    if len(approximation) != 4:
        return None
    center = approximation.mean(axis=0)
    angles = np.arctan2(
        approximation[:, 1] - center[1], approximation[:, 0] - center[0],
    )
    return approximation[np.argsort(angles)].astype(np.float32)


def _edge_angle(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))) % 180


def _angle_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 180
    return min(difference, 180 - difference)


def perspective_table_geometry(
    table_mask: np.ndarray,
) -> Optional[Tuple[List[List[float]], Tuple[float, float], Tuple[float, float], Line]]:
    """Recover an end-view table quad, center axis, and net from color islands."""
    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    substantial = [
        contour for contour in contours
        if cv2.contourArea(contour) >= table_mask.size * .003
    ]
    if not substantial:
        return None
    full = _quad(np.concatenate(substantial))
    if full is None:
        return None
    edges = [(full[index], full[(index + 1) % 4]) for index in range(4)]
    angles = [_edge_angle(*edge) for edge in edges]
    first_pair_difference = _angle_distance(angles[0], angles[2])
    second_pair_difference = _angle_distance(angles[1], angles[3])
    end_indexes = (0, 2) if first_pair_difference <= second_pair_difference else (1, 3)
    end_edges = [edges[index] for index in end_indexes]
    end_lengths = [float(np.linalg.norm(edge[1] - edge[0])) for edge in end_edges]
    player_index = int(np.argmax(end_lengths))
    player_center = np.mean(end_edges[player_index], axis=0)
    opponent_center = np.mean(end_edges[1 - player_index], axis=0)
    player_edge = (float(player_center[0]), float(player_center[1]))
    opponent_edge = (float(opponent_center[0]), float(opponent_center[1]))
    # Side-view fixtures have an almost horizontal player-to-launcher axis and
    # retain the established calibrated path. This fallback is for elevated
    # end/three-quarter views where depth occupies substantial image height.
    axis_x = opponent_edge[0] - player_edge[0]
    axis_y = opponent_edge[1] - player_edge[1]
    if abs(axis_y) <= abs(axis_x) * .6:
        return None

    component_quads = []
    for contour in sorted(substantial, key=cv2.contourArea, reverse=True)[:3]:
        candidate = _quad(contour)
        if candidate is not None:
            component_quads.append(candidate)
    if len(component_quads) < 2:
        return None
    full_center = full.mean(axis=0)
    end_angle = float(np.mean([angles[index] for index in end_indexes]))
    net_edges = []
    for component in component_quads[:2]:
        candidates = []
        for index in range(4):
            edge = (component[index], component[(index + 1) % 4])
            angle_cost = _angle_distance(_edge_angle(*edge), end_angle)
            center_cost = float(np.linalg.norm(np.mean(edge, axis=0) - full_center))
            candidates.append((angle_cost * 10 + center_cost, edge))
        net_edges.append(min(candidates, key=lambda item: item[0])[1])
    left, right = net_edges
    direct = float(np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[1] - right[1]))
    crossed = float(np.linalg.norm(left[0] - right[1]) + np.linalg.norm(left[1] - right[0]))
    if crossed < direct:
        right = (right[1], right[0])
    net_start_array = (left[0] + right[0]) / 2
    net_end_array = (left[1] + right[1]) / 2
    net_start = (float(net_start_array[0]), float(net_start_array[1]))
    net_end = (float(net_end_array[0]), float(net_end_array[1]))
    table_polygon = [[float(x), float(y)] for x, y in full]
    return table_polygon, player_edge, opponent_edge, (
        net_start[0], net_start[1], net_end[0], net_end[1],
    )


def green_table_extent(frame: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """Backward-compatible name for hue-independent table detection."""
    top, bottom, mask, _ = colored_table_extent(frame)
    return top, bottom, mask


def table_edge_x(mask: np.ndarray, boundary_y: float, inward: int) -> Tuple[float, float]:
    """Fit the two table sides near one boundary and extrapolate to its rail."""
    height, width = mask.shape
    start = round(boundary_y + inward * 3)
    stop = round(boundary_y + inward * 25)
    rows = range(max(0, min(start, stop)), min(height, max(start, stop)) + 1)
    samples = []
    for y in rows:
        xs = np.flatnonzero(mask[y])
        if len(xs) >= width * .05:
            samples.append((y, int(xs[0]), int(xs[-1])))
    if len(samples) < 4:
        raise ValueError("could not trace the table side rails")
    values = np.asarray(samples, dtype=np.float64)
    left = float(np.polyval(np.polyfit(values[:, 0], values[:, 1], 1), boundary_y))
    right = float(np.polyval(np.polyfit(values[:, 0], values[:, 2], 1), boundary_y))
    return max(0.0, left), min(float(width - 1), right)


def detect_geometry(
    frame: np.ndarray,
    table_surface: Optional[Tuple[int, int, np.ndarray]] = None,
) -> Geometry:
    """Return the safely visible table area and visual x=0 line.

    The left rail may be covered by the net, so this deliberately returns a
    triangle rather than inventing an unseen lower-left corner. A bounce
    outside that triangle is reported as unknown.
    """
    height, width = frame.shape[:2]
    lines = hough_segments(frame)
    table_top, table_bottom, table_mask = (
        table_surface if table_surface is not None else green_table_extent(frame)
    )
    horizontals = []
    for length, angle, line in lines:
        y = (line[1] + line[3]) / 2
        # Perspective and foreground objects can split a rail into several
        # shorter Hough segments. Geometry is anchored to the independently
        # detected green surface, so accepting a shorter segment here is less
        # error-prone than falling back to a long room/furniture edge nearby.
        if (abs(angle) <= 8 and length >= width * .075
                and table_top - 15 <= y <= table_bottom + 15):
            horizontals.append((y, length, line))
    if len(horizontals) < 3:
        raise ValueError("could not find the table's horizontal boundaries and centre line")
    # Room geometry can contribute much longer horizontal lines than the
    # table in a wide spectator view. Anchor the rail search to the green
    # surface instead of treating the outermost Hough lines as table rails.
    top = min(horizontals, key=lambda item: (abs(item[0] - table_top), -item[1]))
    bottom = min(horizontals, key=lambda item: (abs(item[0] - table_bottom), -item[1]))
    if bottom[0] - top[0] < height * .15:
        raise ValueError("could not separate the table's near and far rails")
    between = [item for item in horizontals if top[0] + 35 < item[0] < bottom[0] - 35]
    if not between:
        raise ValueError("could not separate the white centre stripe from table rails")
    center = max(between, key=lambda item: item[1])

    top_x = table_edge_x(table_mask, top[0], 1)
    bottom_x = table_edge_x(table_mask, bottom[0], -1)

    # The outer left rail can be hidden by the net or clipped by the frame.
    # Preserve that uncertainty as a triangle, but use all four corners when
    # the green surface proves that the near-left corner is visible.
    if bottom_x[0] <= 2:
        visible_table = [[top_x[0], top[0]], [top_x[1], top[0]], [bottom_x[1], bottom[0]]]
    else:
        visible_table = [
            [top_x[0], top[0]], [top_x[1], top[0]],
            [bottom_x[1], bottom[0]], [bottom_x[0], bottom[0]],
        ]

    # Extend the detected centre stripe to the fitted table sides. Hough often
    # returns only one half of the stripe when the net or painted text splits it.
    center_y = center[0]
    fraction = (center_y - top[0]) / (bottom[0] - top[0])
    center_left = top_x[0] + fraction * (bottom_x[0] - top_x[0])
    center_right = top_x[1] + fraction * (bottom_x[1] - top_x[1])
    center_line = (center_left, center_y, center_right, center_y)

    # Find the visible right rail by requiring it to meet the right ends of
    # both horizontal table boundaries.
    right_candidates = []
    for length, angle, line in lines:
        if length < height * .25 or not (35 < abs(angle) < 85):
            continue
        at_top, at_bottom = line_at_y(line, top[0]), line_at_y(line, bottom[0])
        if at_top is None or at_bottom is None or at_bottom <= at_top:
            continue
        score = abs(at_top - top_x[1]) + abs(at_bottom - bottom_x[1])
        right_candidates.append((score, at_top, at_bottom))
    if not right_candidates:
        raise ValueError("could not find the visible right table rail")
    return visible_table, center_line, (top[2], bottom[2])


def calibration_from_frame(
    original: np.ndarray,
    frame: int = 0,
    diagnostic: Optional[PathLike] = None,
) -> Tuple[Calibration, List[int]]:
    """Detect camera geometry from one frame and keep it in memory."""
    scale = min(1.0, CALIBRATION_WIDTH / original.shape[1])
    small = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    inverse_scale = 1 / scale
    table_top, table_bottom, table_mask, table_color = colored_table_extent(small)
    # Preserve the established side-view geometry for the original green-table
    # fixtures. The perspective path is selected when a differently colored
    # setup also presents a substantial depth axis.
    perspective = (
        None if 35 <= table_color["hue_center"] <= 95
        else perspective_table_geometry(table_mask)
    )
    if perspective is not None:
        visible_table, player_edge_small, opponent_edge_small, net = perspective
        player_edge = (
            player_edge_small[0] * inverse_scale,
            player_edge_small[1] * inverse_scale,
        )
        opponent_edge = (
            opponent_edge_small[0] * inverse_scale,
            opponent_edge_small[1] * inverse_scale,
        )
        net_start = (net[0] * inverse_scale, net[1] * inverse_scale)
        net_end = (net[2] * inverse_scale, net[3] * inverse_scale)
        table_polygon = [
            [x * inverse_scale, y * inverse_scale] for x, y in visible_table
        ]
        control = [
            {"name": "x0_player_edge", "image": list(player_edge), "log": [0.0, -TABLE_HALF_LENGTH]},
            {"name": "x0_opponent_edge", "image": list(opponent_edge), "log": [0.0, TABLE_HALF_LENGTH]},
            {"name": "net_base_left", "image": list(net_start), "log": [-TABLE_HALF_WIDTH, 0.0]},
            {"name": "net_base_right", "image": list(net_end), "log": [TABLE_HALF_WIDTH, 0.0]},
        ]
        regions = calibrated_tracking_regions(
            [int(original.shape[1]), int(original.shape[0])],
            table_polygon, player_edge, opponent_edge,
        )
        image_width = float(original.shape[1])
        image_height = float(original.shape[0] - 1)
        regions["launcher_region"] = [
            max(0.0, opponent_edge[0] - image_width * .12), 0.0,
            min(image_width - 1, opponent_edge[0] + image_width * .18), image_height,
        ]
        regions["return_region"] = [
            max(0.0, player_edge[0] - image_width * .18), 0.0,
            min(image_width - 1, player_edge[0] + image_width * .12), image_height,
        ]
        data = {
            "description": "Automatically detected in memory from the first usable frame.",
            "image_size": [int(original.shape[1]), int(original.shape[0])],
            "table_surface_y": 0.7786086,
            "control_points": control,
            "table_polygon": table_polygon,
            "net_line": [list(net_start), list(net_end)],
            "auto_calibrated": True,
            "calibration_frame": frame,
            "camera_geometry": "elevated_end_view",
            "table_color": table_color,
            **regions,
        }
        if diagnostic is not None:
            diagnostic_path = Path(diagnostic)
            view = small.copy()
            cv2.polylines(
                view,
                [np.asarray(visible_table, dtype=np.int32).reshape((-1, 1, 2))],
                True, (0, 255, 255), 2,
            )
            cv2.line(
                view, tuple(map(round, player_edge_small)),
                tuple(map(round, opponent_edge_small)), (255, 255, 255), 2,
            )
            cv2.line(
                view, (round(net[0]), round(net[1])),
                (round(net[2]), round(net[3])), (255, 0, 255), 2,
            )
            cv2.putText(
                view, "auto perspective table + colors", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2,
            )
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(diagnostic_path), view):
                raise ValueError(f"Could not write diagnostic image to {diagnostic_path}")
            data["diagnostic"] = str(diagnostic_path)
        origin = (
            round((net_start[0] + net_end[0]) / 2),
            round((net_start[1] + net_end[1]) / 2),
        )
        return data, [origin[0], origin[1]]

    visible_table, center_line, _ = detect_geometry(
        small, (table_top, table_bottom, table_mask),
    )

    # The white table stripe is x=0. Its visible endpoints establish z=-/+.
    y = (center_line[1] + center_line[3]) / 2
    left_x = min(center_line[0], center_line[2])
    right_x = line_at_y((visible_table[1][0], visible_table[0][1], visible_table[2][0], visible_table[2][1]), y)
    if right_x is None:
        raise ValueError("could not locate the right table rail at the centre stripe")

    # The net's *bottom* edge is a long dark diagonal inside the table.  The
    # bright elevated rail is deliberately rejected by selecting the inward,
    # lower candidate rather than the strongest/brightest diagonal.  It is
    # found relative to the detected rails, so no coordinates are baked in.
    candidates = []
    for length, angle, line in hough_segments(small):
        # In a low side view the net spans roughly the table's visible depth,
        # which can be much less than 35% of the full video height. Scale the
        # requirement to the detected table instead of the surrounding room.
        if length < (visible_table[2][1] - visible_table[0][1]) * .75 or abs(angle) < 60:
            continue
        segment_top, segment_bottom = sorted((line[1], line[3]))
        table_top, table_bottom = visible_table[0][1], visible_table[2][1]
        overlap = min(segment_bottom, table_bottom) - max(segment_top, table_top)
        # A long room line above the table can extrapolate through both rails
        # and look like a net mathematically. Require the observed segment,
        # not merely its infinite extension, to span the tabletop itself.
        if overlap < (table_bottom - table_top) * .5:
            continue
        top_x, bottom_x = line_at_y(line, visible_table[0][1]), line_at_y(line, visible_table[2][1])
        if top_x is None or bottom_x is None or top_x <= bottom_x:
            continue
        # It must cross the visible table interior at both boundary levels.
        if not (visible_table[0][0] < top_x < visible_table[1][0]):
            continue
        if bottom_x >= visible_table[2][0]:
            continue
        center_x = line_at_y(line, y)
        if center_x is None:
            continue
        candidates.append((center_x, top_x, bottom_x, line))
    if not candidates:
        raise ValueError("could not find the physical bottom edge of the net")
    # At a shared image row, the table-side edge of this mesh is the rightward
    # member of its parallel-edge family; the other member is the elevated
    # white rail. This is the key distinction between net base and net top.
    _, net_top_x, net_bottom_x, net_line = max(candidates, key=lambda item: item[0])
    control = [
        {"name": "x0_player_edge", "image": [left_x * inverse_scale, y * inverse_scale], "log": [0.0, -TABLE_HALF_LENGTH]},
        {"name": "x0_opponent_edge", "image": [right_x * inverse_scale, y * inverse_scale], "log": [0.0, TABLE_HALF_LENGTH]},
        {"name": "net_base_top", "image": [net_top_x * inverse_scale, visible_table[0][1] * inverse_scale], "log": [-TABLE_HALF_WIDTH, 0.0]},
        {"name": "net_base_bottom", "image": [net_bottom_x * inverse_scale, visible_table[2][1] * inverse_scale], "log": [TABLE_HALF_WIDTH, 0.0]},
    ]
    table_polygon = [[x * inverse_scale, y * inverse_scale] for x, y in visible_table]
    player_edge = (left_x * inverse_scale, y * inverse_scale)
    opponent_edge = (right_x * inverse_scale, y * inverse_scale)
    regions = calibrated_tracking_regions(
        [int(original.shape[1]), int(original.shape[0])],
        table_polygon,
        player_edge,
        opponent_edge,
    )
    data = {
        "description": "Automatically detected in memory from the first usable frame.",
        "image_size": [int(original.shape[1]), int(original.shape[0])],
        "table_surface_y": 0.7786086,
        "control_points": control,
        "table_polygon": table_polygon,
        "net_line": [[net_top_x * inverse_scale, visible_table[0][1] * inverse_scale], [net_bottom_x * inverse_scale, visible_table[2][1] * inverse_scale]],
        "auto_calibrated": True,
        "calibration_frame": frame,
        "table_color": table_color,
        **regions,
    }
    if diagnostic is not None:
        diagnostic_path = Path(diagnostic)
        view = small.copy()
        cv2.polylines(
            view,
            [np.asarray(visible_table, dtype=np.int32).reshape((-1, 1, 2))],
            True,
            (0, 255, 255),
            2,
        )
        cv2.line(view, tuple(map(int, center_line[:2])), tuple(map(int, center_line[2:])), (255, 255, 255), 2)
        cv2.line(view, (round(net_top_x), round(visible_table[0][1])), (round(net_bottom_x), round(visible_table[2][1])), (255, 0, 255), 2)
        cv2.putText(view, "auto table + x=0 line", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(diagnostic_path), view):
            raise ValueError(f"Could not write diagnostic image to {diagnostic_path}")
        data["diagnostic"] = str(diagnostic_path)
    center_x = line_at_y(net_line, y)
    if center_x is None:
        raise ValueError("could not locate the net at the centre stripe")
    return data, [round(center_x * inverse_scale), round(y * inverse_scale)]


def create_calibration(
    video: PathLike,
    output: PathLike,
    diagnostic: Optional[PathLike] = None,
    frame: int = 0,
) -> CalibrationReport:
    """Explicitly export detected calibration and its visual diagnostic."""
    output = Path(output)
    diagnostic_path = Path(diagnostic) if diagnostic else output.with_suffix(".png")
    source = open_video_source(video)
    try:
        source.seek_frame(frame)
        video_frame = source.read()
        if video_frame is None:
            raise ValueError("Could not read the requested calibration frame")
        data, table_center = calibration_from_frame(
            video_frame.image, video_frame.number, diagnostic_path,
        )

        def color_frames() -> Iterable[np.ndarray]:
            yield video_frame.image
            for _ in range(max(0, round(source.fps * 8) - 1)):
                sampled = source.read()
                if sampled is None:
                    break
                yield sampled.image

        ball_color = infer_ball_color(color_frames(), data)
        if ball_color is not None:
            data["ball_color"] = ball_color
    finally:
        source.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    return {
        "calibration": str(output),
        "diagnostic": str(diagnostic_path),
        "table_center": table_center,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--output", required=True, help="exported calibration JSON")
    parser.add_argument("--diagnostic", help="annotated detection PNG")
    parser.add_argument("--frame", type=int, default=0, help="first usable frame (default: 0)")
    args = parser.parse_args()
    try:
        report = create_calibration(args.video, args.output, args.diagnostic, args.frame)
    except (ValueError, VideoSourceError) as exc:
        raise SystemExit(f"Automatic calibration failed: {exc}.") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

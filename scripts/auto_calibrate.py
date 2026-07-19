#!/usr/bin/env python3
"""Explicitly export a per-camera table calibration from one video frame.

The normal analyser detects this geometry in memory. This separate utility is
for visual diagnostics and manually reviewed overrides.
"""
import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

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


def green_table_extent(frame: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """Locate the table's vertical span without assuming it fills the view."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 45, 30]), np.array([95, 255, 255]))
    row_counts = np.count_nonzero(mask, axis=1)
    rows = np.flatnonzero(row_counts >= frame.shape[1] * .05)
    if len(rows) < 20:
        raise ValueError("could not find a sufficiently large green table surface")
    return int(rows[0]), int(rows[-1]), mask


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


def detect_geometry(frame: np.ndarray) -> Geometry:
    """Return the safely visible table area and visual x=0 line.

    The left rail may be covered by the net, so this deliberately returns a
    triangle rather than inventing an unseen lower-left corner. A bounce
    outside that triangle is reported as unknown.
    """
    height, width = frame.shape[:2]
    lines = hough_segments(frame)
    green_top, green_bottom, green_mask = green_table_extent(frame)
    horizontals = []
    for length, angle, line in lines:
        y = (line[1] + line[3]) / 2
        # Perspective and foreground objects can split a rail into several
        # shorter Hough segments. Geometry is anchored to the independently
        # detected green surface, so accepting a shorter segment here is less
        # error-prone than falling back to a long room/furniture edge nearby.
        if (abs(angle) <= 8 and length >= width * .075
                and green_top - 15 <= y <= green_bottom + 15):
            horizontals.append((y, length, line))
    if len(horizontals) < 3:
        raise ValueError("could not find the table's horizontal boundaries and centre line")
    # Room geometry can contribute much longer horizontal lines than the
    # table in a wide spectator view. Anchor the rail search to the green
    # surface instead of treating the outermost Hough lines as table rails.
    top = min(horizontals, key=lambda item: (abs(item[0] - green_top), -item[1]))
    bottom = min(horizontals, key=lambda item: (abs(item[0] - green_bottom), -item[1]))
    if bottom[0] - top[0] < height * .15:
        raise ValueError("could not separate the table's near and far rails")
    between = [item for item in horizontals if top[0] + 35 < item[0] < bottom[0] - 35]
    if not between:
        raise ValueError("could not separate the white centre stripe from table rails")
    center = max(between, key=lambda item: item[1])

    top_x = table_edge_x(green_mask, top[0], 1)
    bottom_x = table_edge_x(green_mask, bottom[0], -1)

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
    visible_table, center_line, _ = detect_geometry(small)

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
    data = {
        "description": "Automatically detected in memory from the first usable frame.",
        "image_size": [int(original.shape[1]), int(original.shape[0])],
        "table_surface_y": 0.7786086,
        "control_points": control,
        "table_polygon": [[x * inverse_scale, y * inverse_scale] for x, y in visible_table],
        "tracking_polygon": [[0, 0], [int(original.shape[1]), 0], [int(original.shape[1]), int(original.shape[0])], [0, int(original.shape[0])],],
        "net_line": [[net_top_x * inverse_scale, visible_table[0][1] * inverse_scale], [net_bottom_x * inverse_scale, visible_table[2][1] * inverse_scale]],
        "auto_calibrated": True,
        "calibration_frame": frame,
    }
    if diagnostic is not None:
        diagnostic_path = Path(diagnostic)
        view = small.copy()
        cv2.polylines(view, [np.int32(visible_table).reshape((-1, 1, 2))], True, (0, 255, 255), 2)
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
    finally:
        source.close()
    if video_frame is None:
        raise ValueError("Could not read the requested calibration frame")
    data, table_center = calibration_from_frame(
        video_frame.image, video_frame.number, diagnostic_path,
    )
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

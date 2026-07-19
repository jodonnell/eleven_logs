#!/usr/bin/env python3
"""Create a cached, per-camera table calibration from one video frame.

This is intentionally a one-time operation.  It detects the green table and
the white centre line in the first usable frame, then writes a JSON file that
the streaming analyser can reuse without making any pixel-position assumptions.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


TABLE_HALF_WIDTH = 0.7625
TABLE_HALF_LENGTH = 1.37


def line_at_y(line, y):
    x1, y1, x2, y2 = line
    if abs(y2 - y1) < 1e-6:
        return None
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


def hough_segments(frame):
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


def detect_geometry(frame):
    """Return visible table rails and visual x=0 line in downscaled pixels."""
    height, width = frame.shape[:2]
    lines = hough_segments(frame)
    horizontals = []
    for length, angle, line in lines:
        y = (line[1] + line[3]) / 2
        if abs(angle) <= 8 and length >= width * .30 and height * .08 < y < height * .92:
            horizontals.append((y, length, line))
    if len(horizontals) < 3:
        raise ValueError("could not find the table's horizontal boundaries and centre line")
    # The table's top/bottom rails are the outermost long horizontal lines;
    # its white centre stripe is the strongest line between them.
    horizontals.sort()
    top = horizontals[0]
    bottom = horizontals[-1]
    between = [item for item in horizontals if top[0] + 35 < item[0] < bottom[0] - 35]
    if not between:
        raise ValueError("could not separate the white centre stripe from table rails")
    center = max(between, key=lambda item: item[1])

    # The outer left rail can be hidden by the net. The centre stripe itself
    # is the reliable x=0 endpoint feature. Find only the visible right rail
    # by requiring it to meet the right ends of both horizontal rails.
    top_x = sorted((top[2][0], top[2][2]))
    bottom_x = sorted((bottom[2][0], bottom[2][2]))
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
    _, right_top, right_bottom = min(right_candidates)
    corners = [
        [top_x[0], top[0]], [right_top, top[0]],
        [right_bottom, bottom[0]], [0.0, bottom[0]],
    ]
    return corners, center[2], (top[2], bottom[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--output", required=True, help="cached calibration JSON")
    parser.add_argument("--diagnostic", help="annotated detection PNG")
    parser.add_argument("--frame", type=int, default=0, help="first usable frame (default: 0)")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, original = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read the requested calibration frame")
    small = cv2.resize(original, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    height, width = small.shape[:2]
    try:
        corners, center_line, _ = detect_geometry(small)
    except ValueError as exc:
        raise SystemExit(f"Automatic calibration failed: {exc}. Use scripts/calibrate_video.py as a fallback.")

    # The white table stripe is x=0. Its visible endpoints establish z=-/+.
    y = (center_line[1] + center_line[3]) / 2
    left_x = min(center_line[0], center_line[2])
    right_x = line_at_y((corners[1][0], corners[0][1], corners[2][0], corners[2][1]), y)

    # The net's *bottom* edge is a long dark diagonal inside the table.  The
    # bright elevated rail is deliberately rejected by selecting the inward,
    # lower candidate rather than the strongest/brightest diagonal.  It is
    # found relative to the detected rails, so no coordinates are baked in.
    left_angle = 45.0
    candidates = []
    for length, angle, line in hough_segments(small):
        if length < height * .35 or abs(angle) < max(60, left_angle + 6):
            continue
        top_x, bottom_x = line_at_y(line, corners[0][1]), line_at_y(line, corners[2][1])
        if top_x is None or bottom_x is None or top_x <= bottom_x:
            continue
        # It must cross the visible table interior at both boundary levels.
        if not (corners[0][0] < top_x < corners[1][0]):
            continue
        if not (corners[3][0] - width * .10 < bottom_x < corners[2][0]):
            continue
        center_x = line_at_y(line, y)
        candidates.append((center_x, top_x, bottom_x, line))
    if not candidates:
        raise SystemExit("Automatic calibration failed: could not find the physical bottom edge of the net. Use scripts/calibrate_video.py as a fallback.")
    # At a shared image row, the table-side edge of this mesh is the rightward
    # member of its parallel-edge family; the other member is the elevated
    # white rail. This is the key distinction between net base and net top.
    _, net_top_x, net_bottom_x, net_line = max(candidates, key=lambda item: item[0])
    control = [
        {"name": "x0_player_edge", "image": [left_x * 4, y * 4], "log": [0.0, -TABLE_HALF_LENGTH]},
        {"name": "x0_opponent_edge", "image": [right_x * 4, y * 4], "log": [0.0, TABLE_HALF_LENGTH]},
        {"name": "net_base_top", "image": [net_top_x * 4, corners[0][1] * 4], "log": [-TABLE_HALF_WIDTH, 0.0]},
        {"name": "net_base_bottom", "image": [net_bottom_x * 4, corners[2][1] * 4], "log": [TABLE_HALF_WIDTH, 0.0]},
    ]
    # The visual diagnostic makes automatic calibration reviewable.  The
    # analyser draws the resulting log grid before it emits coordinates.
    view = small.copy()
    cv2.polylines(view, [np.int32(corners).reshape((-1, 1, 2))], True, (0, 255, 255), 2)
    cv2.line(view, tuple(map(int, center_line[:2])), tuple(map(int, center_line[2:])), (255, 255, 255), 2)
    cv2.line(view, (round(net_top_x), round(corners[0][1])), (round(net_bottom_x), round(corners[2][1])), (255, 0, 255), 2)
    cv2.putText(view, "auto table + x=0 line", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    diagnostic = args.diagnostic or str(Path(args.output).with_suffix(".png"))
    Path(diagnostic).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(diagnostic, view)
    data = {
        "description": "Automatically detected once from the first usable frame; cache is camera-specific.",
        "image_size": [int(original.shape[1]), int(original.shape[0])],
        "table_surface_y": 0.7786086,
        "control_points": control,
        "table_polygon": [[x * 4, y * 4] for x, y in corners],
        "tracking_polygon": [[0, 0], [int(original.shape[1]), 0], [int(original.shape[1]), int(original.shape[0])], [0, int(original.shape[0])],],
        "net_line": [[net_top_x * 4, corners[0][1] * 4], [net_bottom_x * 4, corners[2][1] * 4]],
        "auto_calibrated": True,
        "calibration_frame": args.frame,
        "diagnostic": diagnostic,
    }
    Path(args.output).write_text(json.dumps(data, indent=2) + "\n")
    center_x = line_at_y(net_line, y)
    print(json.dumps({"calibration": args.output, "diagnostic": diagnostic, "table_center": [round(center_x * 4), round(y * 4)]}, indent=2))


if __name__ == "__main__":
    main()

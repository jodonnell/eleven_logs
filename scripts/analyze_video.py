#!/usr/bin/env python3
"""Streaming bounce analysis for Eleven Table Tennis fixed spectator footage.

Uses only the current/previous frame and bounded trajectory history.  It is
deliberately conservative: an incomplete/occluded trajectory is unknown,
rather than a fabricated table coordinate.
"""
import argparse
import json
import math
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install dependencies first: python3 -m pip install --user opencv-python-headless numpy") from exc

from auto_calibrate import create_calibration


SCALE = 0.25                # process 4K frames at 1K; input remains streaming
MAX_GAP = 3
MIN_TRACK_POINTS = 9
MIN_LAUNCH_TRACK_POINTS = 18


def load_calibration(path, video_width, video_height):
    data = json.loads(Path(path).read_text())
    required = ("image_size", "table_surface_y", "table_polygon", "tracking_polygon", "net_line")
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"Calibration {path} is missing: {', '.join(missing)}")
    expected_size = [video_width, video_height]
    if data["image_size"] != expected_size:
        raise SystemExit(
            f"Calibration is for {data['image_size']}, but this video is {expected_size}. "
            "Create a calibration for this camera/video; do not reuse it."
        )
    if "control_points" in data:
        if len(data["control_points"]) != 4:
            raise SystemExit("Calibration needs exactly four image/log control points")
        image = np.float32([point["image"] for point in data["control_points"]]) * SCALE
        log = np.float32([point["log"] for point in data["control_points"]])
    else:
        names = ("far_left", "far_right", "near_right", "near_left")
        image = np.float32([data["image_corners"][name] for name in names]) * SCALE
        log = np.float32([data["log_corners"][name] for name in names])
    table_polygon = np.float32(data["table_polygon"]) * SCALE
    return data, cv2.getPerspectiveTransform(image, log), table_polygon


def fmt_timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def point_in_rectangle(point, rectangle):
    x, y = point[0] / SCALE, point[1] / SCALE
    left, top, right, bottom = rectangle
    return left <= x <= right and top <= y <= bottom


def signed_distance_to_line(point, line):
    """Signed perpendicular pixel distance from point to a calibrated line."""
    start, end = line
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    return ((dx * (point[1] - start[1])) - (dy * (point[0] - start[0]))) / length


def find_bounce(points, table_polygon, net_line=None):
    """Find a visible table-plane turn in one completed candidate track."""
    if len(points) < MIN_TRACK_POINTS:
        return None
    # A rendered ball casts a compact, moving shadow on the green table. At
    # contact the ball/shadow separation collapses, even when perspective
    # makes the bright ball's screen-space path continue in one direction.
    # This catches the clear 17s sample bounce that has no vertical reversal.
    for index in range(2, len(points) - 2):
        score = points[index][3] if len(points[index]) > 3 else 0
        if score < 25:
            continue
        pixel = (points[index][1], points[index][2])
        if not point_in_polygon(pixel, table_polygon):
            continue
        if net_line is not None and abs(signed_distance_to_line(pixel, net_line)) < 70:
            continue  # net mesh creates a false dark "shadow"
        neighbours = [points[index - 1][3], points[index + 1][3]]
        if score >= max(neighbours):
            return points[index], points[index - 2:index], points[index + 1:index + 3]
    # Two post-contact frames are enough for a terminal turn when the ball
    # disappears behind the launcher immediately afterwards.
    best = None
    for index in range(3, len(points) - 2):
        before = [p[2] for p in points[index - 3:index]]
        after = [p[2] for p in points[index + 1:index + 3]]
        y = points[index][2]
        before_mean, after_mean = sum(before) / len(before), sum(after) / len(after)
        maximum = y - before_mean >= 1 and y - after_mean >= 1
        minimum = before_mean - y >= 1 and after_mean - y >= 1
        if not point_in_polygon((points[index][1], points[index][2]), table_polygon):
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
            if before_speed >= 12 and after_speed <= before_speed * 0.35:
                flattening = (before_speed - after_speed) * 0.6
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
    def __init__(self):
        self.tracks = []

    def update(self, frame_number, candidates):
        pairs = []
        for track_index, track in enumerate(self.tracks):
            points = track["points"]
            if len(points) >= 2:
                a, b = points[-2], points[-1]
                predicted = (b[1] + (b[1] - a[1]), b[2] + (b[2] - a[2]))
            else:
                predicted = points[-1][1:3]
            for candidate_index, candidate in enumerate(candidates):
                distance = math.dist(predicted, candidate[:2])
                if distance <= 80:
                    pairs.append((distance, track_index, candidate_index))
        pairs.sort()
        used_tracks, used_candidates = set(), set()
        for _, track_index, candidate_index in pairs:
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_index]
            candidate = candidates[candidate_index]
            track["points"].append((frame_number, candidate[0], candidate[1], candidate[2]))
            track["gap"] = 0
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)
        for track_index, track in enumerate(self.tracks):
            if track_index not in used_tracks:
                track["gap"] += 1
        completed, active = [], []
        for track in self.tracks:
            if track["gap"] > MAX_GAP:
                completed.append(track["points"])
            else:
                active.append(track)
        self.tracks = active
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index not in used_candidates:
                self.tracks.append({"points": [(frame_number, candidate[0], candidate[1], candidate[2])], "gap": 0})
        return completed

    @property
    def visible_points(self):
        return [point for track in self.tracks for point in track["points"][-12:]]


def shadow_contact_score(hsv, center):
    """Local green-table darkening directly below a bright-ball candidate."""
    x, y = map(round, center)
    height, width = hsv.shape[:2]
    local = hsv[max(0, y + 5):min(height, y + 28), max(0, x - 18):min(width, x + 19)]
    surrounding = hsv[max(0, y - 35):min(height, y + 36), max(0, x - 35):min(width, x + 36)]
    def green_values(region):
        if region.size == 0:
            return np.array([])
        mask = (region[:, :, 0] >= 42) & (region[:, :, 0] <= 88) & (region[:, :, 1] >= 80)
        return region[:, :, 2][mask]
    dark, baseline = green_values(local), green_values(surrounding)
    if len(dark) < 8 or len(baseline) < 20:
        return 0.0
    return max(0.0, float(np.median(baseline) - np.percentile(dark, 5)))


def candidates_for_frame(frame, previous_gray, tracking_polygon):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # White ball: very bright and low saturation. Difference rejects static
    # white markings/text/net edges without retaining a background frame.
    bright = cv2.inRange(hsv, (0, 0, 210), (180, 145, 255))
    if previous_gray is None:
        return gray, []
    moving = cv2.threshold(cv2.absdiff(gray, previous_gray), 18, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_and(bright, moving)
    # Keep 1--2px distant balls; temporal/trajectory filtering handles noise.
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    choices = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if not 1 <= area <= 500:
            continue
        center = tuple(map(float, centers[i]))
        if point_in_polygon(center, tracking_polygon):
            choices.append((area, center, shadow_contact_score(hsv, center)))
    # At track start, prefer the compact moving ball over single-pixel codec
    # shimmer; once a track exists, motion prediction chooses continuity.
    choices.sort(key=lambda item: item[0], reverse=True)
    return gray, [(center[0], center[1], score) for _, center, score in choices]


def map_log_coordinate(homography, image_point, surface_y):
    mapped = cv2.perspectiveTransform(np.float32([[image_point]]), homography)[0][0]
    return round(float(mapped[0]), 4), round(float(surface_y), 4), round(float(mapped[1]), 4)


def draw_overlay(frame, table, net_line, track, events, homography, surface_y):
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
    for event in events:
        if event["frame_number"] != event.get("_draw_frame"):
            continue
        p = event["_pixel"]
        cv2.drawMarker(view, (round(p[0]), round(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
        label = f"{event['outcome']} {event['confidence']:.2f}"
        if event["posx"] is not None:
            label += f" x={event['posx']:.2f} z={event['posz']:.2f}"
        cv2.putText(view, label, (round(p[0]) + 12, round(p[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    return view


class AttemptClassifier:
    """Turn completed ball tracks into one result for each launcher cycle."""

    def __init__(self, fps, calibration, table, net_line, occlusion, homography,
                 video_width, video_height, write_event):
        self.fps = fps
        self.calibration = calibration
        self.table = table
        self.net_line = net_line
        self.occlusion = occlusion
        self.homography = homography
        self.write_event = write_event
        self.events = []
        self.emitted = set()
        self.active_attempt = None
        self.launcher_tracks_seen = 0
        self.launcher_region = calibration.get(
            "launcher_region", [video_width * .58, 0, video_width, video_height]
        )
        self.return_region = calibration.get(
            "return_region", [0, 0, video_width * .28, video_height]
        )
        self.warmup_launcher_tracks = calibration.get("warmup_launcher_tracks", 0)

    def emit(self, event):
        self.write_event(event)
        self.events.append(event)

    def no_bounce_event(self, attempt, draw_frame):
        """Describe a launcher cycle without a confirmed returned bounce."""
        if attempt["returns"]:
            returned = max(attempt["returns"], key=lambda path: math.dist(path[0][1:3], path[-1][1:3]))
            terminal = returned[-1]
            terminal_pixel = (terminal[1], terminal[2])
            start_pixel = (returned[0][1], returned[0][2])
            start_side = signed_distance_to_line(start_pixel, self.net_line)
            end_side = signed_distance_to_line(terminal_pixel, self.net_line)
            crossed_net = start_side * end_side <= 0
            net_distance = abs(end_side)
            if crossed_net:
                outcome, confidence = "unknown", 0.5
            elif net_distance <= self.calibration.get("net_proximity_fraction", 0.2) * math.dist(self.net_line[0], self.net_line[1]):
                outcome, confidence = "net", 0.55
            elif not point_in_polygon(terminal_pixel, self.table):
                outcome, confidence = "off_table", 0.58
            else:
                outcome, confidence = "unknown", 0.35
        else:
            outcome, confidence = "unknown", 0.2
        event = {"video_time_seconds": round(attempt["frame"] / self.fps, 3), "video_timestamp": fmt_timestamp(attempt["frame"] / self.fps), "hit_table": False, "is_in": False, "outcome": outcome, "posx": None, "posy": None, "posz": None, "confidence": confidence, "frame_number": attempt["frame"], "_pixel": attempt["pixel"], "_draw_frame": draw_frame}
        if attempt["returns"]:
            event["return_crossed_net"] = bool(crossed_net)
        return event

    def finish_attempt(self, draw_frame):
        if self.active_attempt is None:
            return
        if self.active_attempt["bounces"]:
            self.emit(self.active_attempt["bounces"][0])
        else:
            self.emit(self.no_bounce_event(self.active_attempt, draw_frame))
        self.active_attempt = None

    def add_bounce(self, path, hit, approach, departure, draw_frame):
        key = (path[0][0], hit[0])
        if key in self.emitted:
            return
        self.emitted.add(key)
        pixel = (hit[1], hit[2])
        in_occlusion = len(self.occlusion) > 2 and point_in_polygon(pixel, self.occlusion)
        posx, posy, posz = map_log_coordinate(self.homography, pixel, self.calibration["table_surface_y"])
        far = posz > 0.03
        continuity = min(1.0, len(approach + departure) / 6)
        confidence = round((0.82 if far else 0.72) * continuity * (0.45 if in_occlusion else 1.0), 2)
        outcome = "unknown" if in_occlusion else ("far_table" if far else "near_table")
        self.active_attempt["bounces"].append({"video_time_seconds": round(hit[0] / self.fps, 3), "video_timestamp": fmt_timestamp(hit[0] / self.fps), "hit_table": not in_occlusion, "is_in": bool(far and not in_occlusion), "outcome": outcome, "posx": posx if not in_occlusion else None, "posy": posy if not in_occlusion else None, "posz": posz if not in_occlusion else None, "confidence": confidence, "frame_number": hit[0], "_pixel": pixel, "_draw_frame": draw_frame})

    def process_tracks(self, tracks, draw_frame):
        for path in tracks:
            if len(path) < MIN_TRACK_POINTS:
                continue
            start_pixel = (path[0][1], path[0][2])
            dx = path[-1][1] - path[0][1]
            is_launcher = (
                point_in_rectangle(start_pixel, self.launcher_region)
                and len(path) >= MIN_LAUNCH_TRACK_POINTS
                and dx <= -120
            )
            if is_launcher:
                self.launcher_tracks_seen += 1
                if self.launcher_tracks_seen > self.warmup_launcher_tracks:
                    self.finish_attempt(draw_frame)
                    self.active_attempt = {"frame": path[0][0], "pixel": start_pixel, "returns": [], "bounces": []}
                continue
            is_return = self.active_attempt and point_in_rectangle(start_pixel, self.return_region) and dx >= 120
            if not is_return:
                continue
            self.active_attempt["returns"].append(path)
            bounce = find_bounce(path, self.table, self.net_line)
            if bounce:
                self.add_bounce(path, *bounce, draw_frame)


def create_video_writer(path, fps, size):
    """Create an annotated-video writer or fail before processing begins."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise SystemExit(f"Could not create annotated video at {path}")
    return writer


def ensure_calibration(args, fps):
    """Return an existing calibration path or create the requested cache."""
    if args.calibration:
        return args.calibration
    cache = Path(args.calibration_cache or f"{Path(args.video).stem}.table-calibration.json")
    if not cache.exists():
        diagnostic = cache.with_suffix(".png")
        print("No calibration supplied; detecting and caching table geometry once.")
        try:
            create_calibration(args.video, cache, diagnostic, round(args.start_seconds * fps))
        except ValueError as exc:
            raise SystemExit(
                "Automatic calibration failed. Inspect its diagnostic and "
                "supply a valid calibration with --calibration."
            ) from exc
    return str(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--calibration", help="Cached per-camera JSON calibration")
    parser.add_argument("--calibration-cache", help="where an automatic first-frame calibration is cached")
    parser.add_argument("--extract-calibration-frame", metavar="PNG", help="write a frame for per-camera corner calibration, then exit")
    parser.add_argument("--output", default="video_bounces.jsonl")
    parser.add_argument("--annotated", default="video_bounces_annotated.mp4")
    parser.add_argument("--no-annotated", action="store_true")
    parser.add_argument("--start-seconds", type=float, default=0, help="seek point; useful when reviewing a short interval")
    parser.add_argument("--end-seconds", type=float, help="stop after this video timestamp")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    if args.start_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE)
    video_width, video_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.extract_calibration_frame:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Could not read a calibration frame")
        if not cv2.imwrite(args.extract_calibration_frame, frame):
            raise SystemExit(f"Could not write calibration frame to {args.extract_calibration_frame}")
        print(json.dumps({"calibration_frame": args.extract_calibration_frame, "image_size": [video_width, video_height]}, indent=2))
        return
    calibration_path = ensure_calibration(args, fps)
    calibration, homography, table = load_calibration(calibration_path, video_width, video_height)
    net_line = np.float32(calibration["net_line"]) * SCALE
    occlusion = np.float32(calibration.get("occlusion_polygon", [])) * SCALE
    tracking_polygon = np.float32(calibration["tracking_polygon"]) * SCALE
    tracker = MultiBallTracker()
    writer = None
    if not args.no_annotated:
        writer = create_video_writer(args.annotated, fps, (width, height))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output:
        def write_event(event):
            output.write(json.dumps({k: v for k, v in event.items() if not k.startswith("_")}) + "\n")

        classifier = AttemptClassifier(
            fps, calibration, table, net_line, occlusion, homography,
            video_width, video_height, write_event,
        )
        previous_gray = None
        # Seeking by timestamp is codec-dependent. Read the position OpenCV
        # actually selected so reported frame numbers remain truthful.
        frame_number = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
        while True:
            ok, original = cap.read()
            if not ok:
                break
            if args.end_seconds is not None and frame_number / fps >= args.end_seconds:
                break
            frame = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
            gray, candidates = candidates_for_frame(frame, previous_gray, tracking_polygon)
            previous_gray = gray
            completed_tracks = tracker.update(frame_number, candidates)
            classifier.process_tracks(completed_tracks, frame_number)
            if writer is not None:
                writer.write(draw_overlay(frame, table, net_line, tracker.visible_points, classifier.events, homography, calibration["table_surface_y"]))
            frame_number += 1
        classifier.finish_attempt(frame_number)
    cap.release()
    if writer is not None:
        writer.release()
    print(json.dumps({"events": len(classifier.events), "output": args.output, "annotated": None if args.no_annotated else args.annotated}, indent=2))


if __name__ == "__main__":
    main()

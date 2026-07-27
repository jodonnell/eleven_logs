"""Low-resolution HUD OCR and debounced telemetry readings."""

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

from .models import MAX_SPIN_REVOLUTIONS_PER_SECOND, TelemetryReading


PROCESSING_WIDTH = 1024

DIGIT_TEMPLATES = {
    digit: np.asarray(
        [[pixel == "1" for pixel in row] for row in bitmap.split("/")],
        dtype=np.uint8,
    )
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
        np.asarray(
            [[pixel == "1" for pixel in row] for row in variant.split("/")],
            dtype=np.uint8,
        )
        for variant in bitmap.split("|")
    ]
    for digit, bitmap in {
        "0": "0100/1011/1001/1011|1001/1001/1111|1011/1001/1111|1101/1001/1101|011110/110011/110011/011110|11110/11011/10011/10011/11111",
        "1": "111/001/001|111/011/011|111/111/011/011|111/001/011/001|111/001/001/001",
        "2": "0011/0010/1100|0100/0011/0110/1100|0100/0011/0110/1110|01111/00001/00110/11111|001110/011111/000111/001110/111111|011110/000011/001100/111010",
        "3": "011/110/011|100/011/110/011|11110/00110/00011/00011",
        "4": "0010/0110/1010/0011|0010/0110/1010/1011|0110/1010/1111|00011/01111/11011/00011|000110/011010/111111/000010",
        "5": "1000/1111/0011|1100/0111/0001|1110/1000/1111/0011|1110/1000/1111/1011|11110/11110/00111/10111|10000/11110/00011/11110|010000/011110/000011/111110|11110/11110/00011/00011|011110/011111/011111/000011/111111|001100/011111/001111/000111/111111",
        "6": "1000/1111/1011|1011/1110/1001|1100/1111/1001|01100/11000/11110/11110/01100|011110/011110/110011/010011|000110/011111/111111/111011/011111|001100/011111/011111/111111/111111",
        "7": "011/010/100|011/010/110|111/001/011/010|00011/00110/01100/11000|111111/000110/001100/001000|11110/00011/01100/11000|11111/11111/00110/01110/11100",
        "8": "1011/1110/1001|1011/1110/1011|1101/0111/1101|011011/011110/110011/111111|010011/011110/110011/111111|011110/011110/011111/110011",
        "9": "0100/1011/1111/0010|1101/1111/0001|1011/1111/0011|010011/110011/001011/011110|011011/110011/001011/011110|011110/110011/011111/000110",
    }.items()
}

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
        native = mask > 0
        exact = [
            digit
            for digit, raw_templates in LOW_RES_DIGIT_TEMPLATES.items()
            if any(
                raw_template.shape == mask.shape
                and np.array_equal(native, raw_template)
                for raw_template in raw_templates
            )
        ]
        if len(exact) == 1:
            return exact[0], 1.0
        scores = {}
        for digit, raw_templates in LOW_RES_DIGIT_TEMPLATES.items():
            candidate_scores = []
            for raw_template in raw_templates:
                template = normalize_digit(raw_template * 255)
                intersection = np.count_nonzero(normalized & template)
                total = np.count_nonzero(normalized) + np.count_nonzero(template)
                score = 2 * intersection / total if total else 0.0
                if raw_template.shape == mask.shape:
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
        ranked = sorted(scores, key=lambda item: scores[item], reverse=True)
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
    digit = max(scores, key=lambda item: scores[item])
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
            # In the profile-side view the physical net post crosses the
            # leading ``1`` of a three-digit player spin. The suffix remains
            # visible in the wider antialiased mask, so infer only that
            # occluded leading digit and classify the two visible cells.
            if digit_count == 3 and index == 0:
                digits.append("1")
                continue
            cell_mask = white if digit_count == 3 else core
            cell = cell_mask[:, left:right_edge]
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
    boxes: List[Tuple[int, int, int, int, int]] = sorted(
        [
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]))
            for box in stats[1:]
            if box[4] >= minimum_area
        ],
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

"""Stateful launch, return, contact, and attempt classification."""

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
    Attempt,
    Bounce,
    BounceEvent,
    Calibration,
    ContactCandidate,
    DetectorSettings,
    Point,
    TelemetryReading,
    Track,
    TrackDiagnostic,
    TrackPoint,
)
from .detection import (
    bounce_signal,
    bounce_strength,
    find_bounce,
    find_bounces,
    fmt_timestamp,
    point_in_polygon,
    point_in_rectangle,
    point_near_polygon,
    signed_distance_to_line,
)
from .telemetry import TelemetryReader
from .vision import map_log_coordinate

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
        if self.calibration.get("camera_geometry") == "profile_side_view":
            # In the canonical side view the projected table contact and the
            # player-return HUD update share essentially the same timestamp.
            # Launcher tracking can therefore begin after that HUD state was
            # published, causing it to be mistaken for machine telemetry.
            # Anchor the player result to the contact, then recover the most
            # recent distinct preceding state as the machine setting.
            hit = self.telemetry_near(frame)
            if hit is not None:
                preceding = [
                    item for item in self.telemetry_history
                    if item.frame_number < hit.frame_number
                    and not TelemetryReader.same_values(item, hit)
                ]
                return hit, (preceding[-1] if preceding else machine)
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

        xs = np.asarray([point[1] for point in path], dtype=np.float64)
        ys = np.asarray([point[2] for point in path], dtype=np.float64)
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
        hit_telemetry, machine_telemetry = self.telemetry_pair_for_attempt(
            attempt, hit[0],
        )
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
            hit=(
                hit_telemetry.to_player_record(self.fps)
                if hit_telemetry else None
            ),
            machine=(
                machine_telemetry.to_record(self.fps)
                if machine_telemetry else None
            ),
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

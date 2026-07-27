"""Offline and streaming normalization of detected attempt events."""

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

from .models import BounceEvent
from .detection import fmt_timestamp

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

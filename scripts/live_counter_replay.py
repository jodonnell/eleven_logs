#!/usr/bin/env python3
"""Replay a clean video through the live counter and verify labeled outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_video.py"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def is_hit(outcome: str) -> bool:
    return outcome == "hit"


def streak_transitions(records: Sequence[Dict[str, Any]]) -> List[int]:
    """Mirror the browser's chronological-prefix streak calculation."""
    transitions = []
    received: List[Dict[str, Any]] = []
    for record in records:
        received.append(record)
        ordered = sorted(
            received,
            key=lambda item: item.get(
                "attempt_frame_number", item["frame_number"],
            ),
        )
        streak = 0
        for item in ordered:
            if item["outcome"] == "hit":
                streak += 1
            elif item["outcome"] in ("miss", "out"):
                streak = 0
        transitions.append(streak)
    return transitions


def expected_streaks(outcomes: Sequence[str]) -> List[int]:
    streak = 0
    transitions = []
    for outcome in outcomes:
        streak = streak + 1 if outcome == "hit" else 0
        transitions.append(streak)
    return transitions


def reconcile_live_messages(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply the same shot/snapshot state updates as the browser."""
    shots: List[Dict[str, Any]] = []
    for message in messages:
        if message.get("type") == "snapshot":
            shots = list(message["shots"])
        elif message.get("type") == "reset":
            continue
        else:
            shots.append(message)
    return shots


def mismatch_line(
    index: int,
    expected: str,
    actual: str,
    record: Dict[str, Any] | None,
) -> str:
    timestamp = record.get("video_timestamp", "-") if record else "-"
    delay = record.get("publication_delay_seconds", "-") if record else "-"
    return (
        f"#{index + 1} expected={expected} actual={actual} "
        f"shot={timestamp} delay={delay}s"
    )


def verify_records(
    fixture: Dict[str, Any],
    canonical: Sequence[Dict[str, Any]],
    live: Sequence[Dict[str, Any]],
) -> List[str]:
    expected = fixture["outcomes"]
    mismatches = []
    if len(canonical) != len(expected):
        mismatches.append(
            f"launch-count expected={len(expected)} actual={len(canonical)}"
        )
    for index in range(max(len(expected), len(canonical))):
        wanted = expected[index] if index < len(expected) else "<none>"
        record = canonical[index] if index < len(canonical) else None
        actual = record["outcome"] if record else "<missing>"
        if record is None or is_hit(wanted) != is_hit(actual):
            mismatches.append(mismatch_line(index, wanted, actual, record))

    logical_frames = [
        record.get("attempt_frame_number", record["frame_number"])
        for record in canonical
    ]
    duplicates = sorted({frame for frame in logical_frames if logical_frames.count(frame) > 1})
    if duplicates:
        mismatches.append(f"duplicate-logical-frames actual={duplicates}")

    actual_streaks = streak_transitions(canonical)
    wanted_streaks = expected_streaks(expected)
    for index, (wanted, actual) in enumerate(zip(wanted_streaks, actual_streaks)):
        if wanted != actual:
            record = canonical[index]
            mismatches.append(mismatch_line(
                index, f"streak:{wanted}", f"streak:{actual}", record,
            ))

    browser_shots = reconcile_live_messages(live)
    if len(browser_shots) != len(expected):
        mismatches.append(
            f"browser-launch-count expected={len(expected)} actual={len(browser_shots)}"
        )
    for index in range(min(len(expected), len(browser_shots))):
        wanted = expected[index]
        record = browser_shots[index]
        actual = record["outcome"]
        if is_hit(wanted) != is_hit(actual):
            mismatches.append(mismatch_line(index, wanted, actual, record))
    browser_streaks = streak_transitions(browser_shots)
    for index, (wanted, actual) in enumerate(zip(wanted_streaks, browser_streaks)):
        if wanted != actual:
            mismatches.append(mismatch_line(
                index,
                f"browser-streak:{wanted}",
                f"browser-streak:{actual}",
                browser_shots[index],
            ))

    shot_publications = [
        record for record in live if record.get("type") not in ("snapshot", "reset")
    ]
    for index, record in enumerate(shot_publications):
        required = ("frame_number", "publication_frame_number", "publication_delay_seconds")
        missing = [name for name in required if name not in record]
        if missing:
            mismatches.append(
                mismatch_line(index, "publication metadata", f"missing:{','.join(missing)}", record)
            )
        if (
            record.get("outcome") == "miss"
            and record.get("publication_delay_seconds", 0)
            > fixture["max_no_swing_publication_delay_seconds"]
        ):
            mismatches.append(mismatch_line(
                index,
                f"delay<={fixture['max_no_swing_publication_delay_seconds']}s",
                f"delay={record['publication_delay_seconds']}s",
                record,
            ))
    return mismatches


def run_replay(fixture_path: Path) -> List[str]:
    fixture = json.loads(fixture_path.read_text())
    video = ROOT / fixture["video"]
    with tempfile.TemporaryDirectory() as directory:
        canonical_path = Path(directory) / "canonical.jsonl"
        live_path = Path(directory) / "live.jsonl"
        command = [
            sys.executable, str(ANALYZER), str(video),
            "--output", str(canonical_path),
            "--live-events", str(live_path),
            "--no-annotated",
        ]
        if fixture.get("calibration"):
            command.extend(["--calibration", str(ROOT / fixture["calibration"])])
        if fixture.get("start_seconds") is not None:
            command.extend(["--start-seconds", str(fixture["start_seconds"])])
        try:
            subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            return [f"analyzer-error actual={detail}"]
        return verify_records(
            fixture, read_jsonl(canonical_path), read_jsonl(live_path),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    mismatches = run_replay(args.fixture)
    if mismatches:
        print("live counter replay mismatch:")
        print("\n".join(f"  {line}" for line in mismatches[:20]))
        if len(mismatches) > 20:
            print(f"  ... {len(mismatches) - 20} more")
        raise SystemExit(1)
    fixture = json.loads(args.fixture.read_text())
    print(f"live counter replay OK: {len(fixture['outcomes'])} launches")


if __name__ == "__main__":
    main()

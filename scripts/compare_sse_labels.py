#!/usr/bin/env python3
"""Compare browser-reconciled SSE attempt upserts with human video labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluate_detector import align_outcomes, evaluate, read_json, read_jsonl
from live_counter_replay import reconcile_live_messages


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def publication_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    values = [
        float(item["attempt_publication_delay_seconds"])
        for item in records
        if item.get("attempt_publication_delay_seconds") is not None
    ]
    return {
        "count": len(values),
        "median_seconds": round(statistics.median(values), 3) if values else None,
        "p95_seconds": percentile(values, .95),
        "maximum_seconds": round(max(values), 3) if values else None,
    }


def alignment_rows(
    truth: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    labels = truth["labels"]
    alignment = align_outcomes(
        [item["outcome"] for item in labels],
        [item["outcome"] for item in records],
    )
    rows = []
    for item in alignment:
        label = labels[item.expected_index] if item.expected_index is not None else None
        record = records[item.actual_index] if item.actual_index is not None else None
        rows.append({
            "kind": item.kind,
            "human_number": (
                item.expected_index + 1 if item.expected_index is not None else None
            ),
            "human_time_seconds": label.get("time_seconds") if label else None,
            "human_outcome": label.get("outcome") if label else None,
            "sse_number": (
                item.actual_index + 1 if item.actual_index is not None else None
            ),
            "sse_sequence": record.get("sequence") if record else None,
            "attempt_id": record.get("attempt_id") if record else None,
            "sse_outcome": record.get("outcome") if record else None,
            "anchor_time_seconds": (
                record.get("anchor_frame_number", 0) / 60 if record else None
            ),
            "evidence_time_seconds": (
                record.get("frame_number", 0) / 60 if record else None
            ),
            "publication_time_seconds": (
                record.get("publication_video_time_seconds") if record else None
            ),
            "publication_delay_seconds": (
                record.get("attempt_publication_delay_seconds") if record else None
            ),
            "confidence": record.get("confidence") if record else None,
            "revision": record.get("revision", 0) if record else None,
        })
    return rows


def add_streak_comparison(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay aligned outcomes as the user-visible browser streak."""
    human_streak = 0
    sse_streak = 0
    compared = 0
    exact = 0
    human_maximum = 0
    sse_maximum = 0
    for row in rows:
        if row["human_outcome"] is not None:
            human_streak = (
                human_streak + 1 if row["human_outcome"] == "hit" else 0
            )
            human_maximum = max(human_maximum, human_streak)
        if row["sse_outcome"] is not None:
            sse_streak = sse_streak + 1 if row["sse_outcome"] == "hit" else 0
            sse_maximum = max(sse_maximum, sse_streak)
        row["human_streak"] = human_streak
        row["sse_streak"] = sse_streak
        if row["human_outcome"] is not None:
            compared += 1
            exact += human_streak == sse_streak
    return {
        "compared_human_attempts": compared,
        "exact_after_attempt": exact,
        "accuracy": exact / compared if compared else 0,
        "human_maximum": human_maximum,
        "sse_maximum": sse_maximum,
    }


def compare(
    truth: Dict[str, Any],
    messages: Sequence[Dict[str, Any]],
    canonical: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    finalized = reconcile_live_messages(messages)
    rows = alignment_rows(truth, finalized)
    pending_ids = {
        item["attempt_id"]
        for item in messages
        if item.get("type") == "attempt_upsert" and item.get("state") == "pending"
    }
    finalized_ids = {
        item["attempt_id"]
        for item in messages
        if item.get("type") == "attempt_upsert" and item.get("state") == "finalized"
    }
    report: Dict[str, Any] = {
        "human": {
            "attempts": len(truth["labels"]),
            "hits": sum(item["outcome"] == "hit" for item in truth["labels"]),
            "misses": sum(item["outcome"] == "miss" for item in truth["labels"]),
        },
        "raw_sse": {
            "messages": len(messages),
            "pending_messages": sum(
                item.get("state") == "pending" for item in messages
            ),
            "finalized_messages": sum(
                item.get("state") == "finalized" for item in messages
            ),
            "revisions": sum(item.get("revision", 0) > 0 for item in messages),
            "attempt_ids": len(pending_ids | finalized_ids),
            "left_pending": len(pending_ids - finalized_ids),
        },
        "browser_reconciled": {
            "attempts": len(finalized),
            "hits": sum(item.get("outcome") == "hit" for item in finalized),
            "misses": sum(item.get("outcome") != "hit" for item in finalized),
            "evaluation": evaluate(truth, finalized),
            "publication": publication_summary(finalized),
            "streak": add_streak_comparison(rows),
        },
        "alignment": rows,
    }
    if canonical is not None:
        report["canonical"] = {
            "attempts": len(canonical),
            "hits": sum(item.get("outcome") == "hit" for item in canonical),
            "misses": sum(item.get("outcome") != "hit" for item in canonical),
            "evaluation": evaluate(truth, canonical),
        }
    return report


def markdown(report: Dict[str, Any]) -> str:
    human = report["human"]
    raw = report["raw_sse"]
    browser = report["browser_reconciled"]
    sequence = browser["evaluation"]["sequence"]
    hit = browser["evaluation"]["hit_classification"]
    publication = browser["publication"]
    streak = browser["streak"]
    lines = [
        "# Human labels vs browser SSE",
        "",
        "## Summary",
        "",
        f"- Human attempts: {human['attempts']} "
        f"({human['hits']} hits, {human['misses']} misses)",
        f"- Browser-reconciled SSE attempts: {browser['attempts']} "
        f"({browser['hits']} hits, {browser['misses']} misses)",
        f"- Correct aligned outcomes: {sequence['match']}",
        f"- Wrong outcomes: {sequence['wrong_outcome']}",
        f"- Missing attempts: {sequence['missing']}",
        f"- Extra attempts: {sequence['extra']}",
        f"- Sequence accuracy: {sequence['accuracy']:.1%}",
        f"- Hit precision: {hit['precision']:.1%}",
        f"- Hit recall: {hit['recall']:.1%}",
        f"- Visible streak correct after human attempts: "
        f"{streak['exact_after_attempt']}/{streak['compared_human_attempts']} "
        f"({streak['accuracy']:.1%})",
        f"- Maximum streak: human {streak['human_maximum']}, "
        f"SSE {streak['sse_maximum']}",
        f"- Raw SSE messages: {raw['messages']} "
        f"({raw['pending_messages']} pending, "
        f"{raw['finalized_messages']} finalized, {raw['revisions']} revisions)",
        f"- Attempts left pending: {raw['left_pending']}",
        f"- Publication delay: median {publication['median_seconds']}s, "
        f"p95 {publication['p95_seconds']}s, "
        f"max {publication['maximum_seconds']}s",
    ]
    canonical = report.get("canonical")
    if canonical is not None:
        canonical_sequence = canonical["evaluation"]["sequence"]
        lines.extend([
            "",
            "## Canonical detector comparison",
            "",
            f"- Canonical attempts: {canonical['attempts']}",
            f"- Correct outcomes: {canonical_sequence['match']}",
            f"- Wrong outcomes: {canonical_sequence['wrong_outcome']}",
            f"- Missing attempts: {canonical_sequence['missing']}",
            f"- Extra attempts: {canonical_sequence['extra']}",
            f"- Sequence accuracy: {canonical_sequence['accuracy']:.1%}",
        ])
    errors = [row for row in report["alignment"] if row["kind"] != "match"]
    lines.extend([
        "",
        "## SSE mismatches",
        "",
        "| Kind | Human # | Human time | Expected | SSE # | SSE anchor | Actual | Delay |",
        "|---|---:|---:|---|---:|---:|---|---:|",
    ])
    for row in errors:
        lines.append(
            f"| {row['kind']} | {row['human_number'] or '-'} | "
            f"{row['human_time_seconds'] if row['human_time_seconds'] is not None else '-'} | "
            f"{row['human_outcome'] or '-'} | {row['sse_number'] or '-'} | "
            f"{round(row['anchor_time_seconds'], 3) if row['anchor_time_seconds'] is not None else '-'} | "
            f"{row['sse_outcome'] or '-'} | "
            f"{row['publication_delay_seconds'] if row['publication_delay_seconds'] is not None else '-'} |"
        )
    if not errors:
        lines.append("| None | - | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("sse", type=Path)
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--alignment-output", type=Path)
    args = parser.parse_args()

    report = compare(
        read_json(args.labels),
        read_jsonl(args.sse),
        read_jsonl(args.canonical) if args.canonical else None,
    )
    rendered = markdown(report)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.write_text(rendered, encoding="utf-8")
    if args.alignment_output:
        args.alignment_output.write_text(
            "".join(json.dumps(row) + "\n" for row in report["alignment"]),
            encoding="utf-8",
        )
    print(rendered, end="")


if __name__ == "__main__":
    main()

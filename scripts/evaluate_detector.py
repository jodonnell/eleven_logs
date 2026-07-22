#!/usr/bin/env python3
"""Compare ordered detector attempts with timestamped hit/miss ground truth."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AlignmentItem:
    kind: str
    expected_index: Optional[int]
    actual_index: Optional[int]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def binary_outcome(value: str) -> str:
    return "hit" if value == "hit" else "miss"


def align_outcomes(
    expected: Sequence[str], actual: Sequence[str],
) -> List[AlignmentItem]:
    """Levenshtein alignment with diagonal ties preferred for stable locality."""
    rows, columns = len(expected) + 1, len(actual) + 1
    costs = [[0] * columns for _ in range(rows)]
    steps: List[List[Optional[str]]] = [[None] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        steps[row][0] = "missing"
    for column in range(1, columns):
        costs[0][column] = column
        steps[0][column] = "extra"
    for row in range(1, rows):
        for column in range(1, columns):
            same = expected[row - 1] == actual[column - 1]
            choices: List[Tuple[int, int, str]] = [
                (costs[row - 1][column - 1] + (0 if same else 1), 0,
                 "match" if same else "wrong_outcome"),
                (costs[row - 1][column] + 1, 1, "missing"),
                (costs[row][column - 1] + 1, 2, "extra"),
            ]
            cost, _, step = min(choices)
            costs[row][column] = cost
            steps[row][column] = step
    alignment = []
    row, column = len(expected), len(actual)
    while row or column:
        step = steps[row][column]
        if step in ("match", "wrong_outcome"):
            alignment.append(AlignmentItem(step, row - 1, column - 1))
            row -= 1
            column -= 1
        elif step == "missing":
            alignment.append(AlignmentItem(step, row - 1, None))
            row -= 1
        elif step == "extra":
            alignment.append(AlignmentItem(step, None, column - 1))
            column -= 1
        else:
            raise AssertionError("alignment traceback failed")
    return list(reversed(alignment))


def evaluate(
    truth: Dict[str, Any], predictions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    labels = truth["labels"]
    unresolved = [item for item in labels if item["outcome"] not in ("hit", "miss")]
    if unresolved:
        raise ValueError(f"ground truth has {len(unresolved)} unresolved labels")
    expected = [item["outcome"] for item in labels]
    actual = [binary_outcome(item["outcome"]) for item in predictions]
    alignment = align_outcomes(expected, actual)
    counts = {kind: 0 for kind in ("match", "wrong_outcome", "missing", "extra")}
    confusion = {key: 0 for key in ("true_hit", "false_hit", "true_miss", "false_miss")}
    errors = []
    for item in alignment:
        counts[item.kind] += 1
        wanted = labels[item.expected_index] if item.expected_index is not None else None
        predicted = predictions[item.actual_index] if item.actual_index is not None else None
        wanted_outcome = wanted["outcome"] if wanted else None
        predicted_outcome = binary_outcome(predicted["outcome"]) if predicted else None
        if wanted_outcome == "hit" and predicted_outcome == "hit":
            confusion["true_hit"] += 1
        elif wanted_outcome == "miss" and predicted_outcome == "hit":
            confusion["false_hit"] += 1
        elif wanted_outcome == "miss" and predicted_outcome == "miss":
            confusion["true_miss"] += 1
        elif wanted_outcome == "hit" and predicted_outcome == "miss":
            confusion["false_miss"] += 1
        elif wanted_outcome == "hit" and predicted_outcome is None:
            confusion["false_miss"] += 1
        elif wanted_outcome is None and predicted_outcome == "hit":
            confusion["false_hit"] += 1
        if item.kind != "match":
            errors.append({
                "kind": item.kind,
                "expected_number": (
                    item.expected_index + 1 if item.expected_index is not None else None
                ),
                "expected_outcome": wanted_outcome,
                "label_time_seconds": wanted.get("time_seconds") if wanted else None,
                "predicted_number": (
                    item.actual_index + 1 if item.actual_index is not None else None
                ),
                "predicted_outcome": predicted_outcome,
                "prediction_time_seconds": (
                    predicted.get("attempt_frame_number", predicted.get("frame_number", 0)) / 60
                    if predicted else None
                ),
            })
    compared = sum(counts.values())
    tp, fp = confusion["true_hit"], confusion["false_hit"]
    fn = confusion["false_miss"]
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        "expected_attempts": len(expected),
        "predicted_attempts": len(actual),
        "sequence": {
            **counts,
            "accuracy": counts["match"] / compared if compared else 0,
        },
        "hit_classification": {
            **confusion,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        },
        "errors": errors,
    }


def format_report(report: Dict[str, Any]) -> str:
    sequence = report["sequence"]
    hit = report["hit_classification"]
    lines = [
        "# Detector evaluation",
        "",
        f"- Expected attempts: {report['expected_attempts']}",
        f"- Predicted attempts: {report['predicted_attempts']}",
        f"- Sequence accuracy: {sequence['accuracy']:.1%}",
        f"- Correct: {sequence['match']}",
        f"- Wrong outcome: {sequence['wrong_outcome']}",
        f"- Missing launches: {sequence['missing']}",
        f"- Extra launches: {sequence['extra']}",
        f"- Hit precision: {hit['precision']:.1%}",
        f"- Hit recall: {hit['recall']:.1%}",
        "",
        "## Errors",
        "",
    ]
    if not report["errors"]:
        lines.append("None.")
    for error in report["errors"]:
        label_time = error["label_time_seconds"]
        prediction_time = error["prediction_time_seconds"]
        lines.append(
            f"- {error['kind']}: expected #{error['expected_number']} "
            f"{error['expected_outcome']} at {label_time if label_time is not None else '-'}s; "
            f"predicted #{error['predicted_number']} {error['predicted_outcome']} "
            f"at {round(prediction_time, 3) if prediction_time is not None else '-'}s"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = evaluate(read_json(args.truth), read_jsonl(args.predictions))
    markdown = format_report(report)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()

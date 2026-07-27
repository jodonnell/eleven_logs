#!/usr/bin/env python3
"""Report or remove old disposable artifact directories."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SCOPES = ("runs", "scratch")
PIN_FILE = "PINNED"


def directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def cleanup_candidates(
    artifacts: Path,
    older_than_days: float,
    now: float | None = None,
) -> Iterable[Path]:
    cutoff = (time.time() if now is None else now) - older_than_days * 86400
    for scope in SCOPES:
        parent = artifacts / scope
        if not parent.exists():
            continue
        for path in sorted(parent.iterdir()):
            if not path.is_dir() or path.name == "archive":
                continue
            if (path / PIN_FILE).exists():
                continue
            if path.stat().st_mtime < cutoff:
                yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=14,
        help="select run and scratch directories older than this (default: 14)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="remove selected directories; without this flag, only report them",
    )
    args = parser.parse_args()
    if args.older_than_days < 0:
        parser.error("--older-than-days must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    candidates = list(cleanup_candidates(ARTIFACTS, args.older_than_days))
    if not candidates:
        print("No disposable artifact directories matched.")
        return

    total = 0
    action = "Removing" if args.apply else "Would remove"
    for path in candidates:
        size = directory_size(path)
        total += size
        print(f"{action} {path.relative_to(ROOT)} ({size / 1048576:.1f} MiB)")
        if args.apply:
            shutil.rmtree(path)

    qualifier = "Freed" if args.apply else "Recoverable"
    print(f"{qualifier}: {total / 1048576:.1f} MiB")
    if not args.apply:
        print("Dry run only. Re-run with --apply to remove these directories.")


if __name__ == "__main__":
    main()

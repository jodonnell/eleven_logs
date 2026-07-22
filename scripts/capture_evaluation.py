#!/usr/bin/env python3
"""Record an SRT evaluation stream without running the video classifier."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRT_URL = "srt://192.168.1.197:9000?mode=caller&latency=120000"


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return ROOT / "artifacts" / f"evaluation-{stamp}.mkv"


def capture_command(source: str, output: Path, seconds: float) -> List[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-i", source,
        "-map", "0:v:0",
        "-c:v", "copy",
        "-an",
        "-t", str(seconds),
        str(output),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SRT_URL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--seconds", type=float, default=1200,
        help="safety limit in seconds (default: 1200 / 20 minutes)",
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    output = (args.output or default_output()).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing recording: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording directly to {output}", flush=True)
    print("Press Ctrl-C when the session is finished.", flush=True)
    command = capture_command(args.source, output, args.seconds)
    try:
        os.execvp(command[0], command)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg is required but was not found") from exc


if __name__ == "__main__":
    main()

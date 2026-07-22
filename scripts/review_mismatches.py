#!/usr/bin/env python3
"""Build raw/annotated review clips and an evidence manifest for detector errors."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


CATEGORIES = (
    "own-side-first bounce",
    "net interaction",
    "off-table return",
    "tracker handoff",
    "delayed/overlapping ball",
    "missed return",
    "missed launch",
    "extra launch",
)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def error_anchor(error: Dict[str, Any]) -> float:
    """Prefer launch time, falling back to the visible ground-truth outcome."""
    prediction_time = error.get("prediction_time_seconds")
    if prediction_time is not None:
        return float(prediction_time)
    return float(error["label_time_seconds"])


def nearby_diagnostics(
    diagnostics: Sequence[Dict[str, Any]], anchor: float, fps: float, window: float,
) -> List[Dict[str, Any]]:
    start_frame = round((anchor - window) * fps)
    end_frame = round((anchor + window) * fps)
    nearby = []
    for diagnostic in diagnostics:
        points = diagnostic.get("points") or []
        if not points:
            continue
        hit_frame = next(
            (
                int(part.split("=", 1)[1])
                for part in diagnostic.get("reason", "").split()
                if part.startswith("hit_frame=")
            ),
            int(points[-1][0]),
        )
        if start_frame <= hit_frame <= end_frame:
            nearby.append({
                "hit_frame": hit_frame,
                "time_seconds": round(hit_frame / fps, 3),
                "signal": diagnostic.get("reason", "").split()[0] or None,
                "track_start_frame": int(points[0][0]),
                "track_end_frame": int(points[-1][0]),
                "points": points,
            })
    return nearby


def build_manifest(
    report: Dict[str, Any], diagnostics: Sequence[Dict[str, Any]], fps: float,
    before: float, after: float,
) -> Dict[str, Any]:
    reviews = []
    for number, error in enumerate(report["errors"], start=1):
        anchor = error_anchor(error)
        stem = f"{number:02d}-{error['kind']}-expected-{error.get('expected_number') or 'none'}"
        reviews.append({
            "review_number": number,
            **error,
            "anchor_time_seconds": round(anchor, 3),
            "clip_start_seconds": round(max(0, anchor - before), 3),
            "clip_duration_seconds": round(before + after, 3),
            "raw_clip": f"raw/{stem}.mp4",
            "annotated_clip": f"annotated/{stem}.mp4",
            "category": None,
            "review_notes": "",
            "nearby_bounce_diagnostics": nearby_diagnostics(
                diagnostics, anchor, fps, before + after,
            ),
        })
    return {
        "categories": list(CATEGORIES),
        "instructions": (
            "Review the same numbered raw and annotated clips, set category to one "
            "listed value, and record the visible contact order/track failure in review_notes."
        ),
        "scorecard": {
            "expected_attempts": report["expected_attempts"],
            "predicted_attempts": report["predicted_attempts"],
            **report["sequence"],
            **report["hit_classification"],
        },
        "reviews": reviews,
    }


def run(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def render_clip(
    ffmpeg: str, source: Path, output: Path, start: float, duration: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", str(source), "-t", str(duration),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(output),
    ])


def concat_reel(ffmpeg: str, clips: Iterable[Path], output: Path) -> None:
    clips = list(clips)
    if not clips:
        return
    concat_file = output.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{clip.resolve()}'\n" for clip in clips), encoding="utf-8",
    )
    try:
        run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(output),
        ])
    finally:
        concat_file.unlink(missing_ok=True)


def render_media(
    manifest: Dict[str, Any], output_dir: Path, raw_video: Path,
    annotated_video: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required with --render")
    raw_clips, annotated_clips = [], []
    for review in manifest["reviews"]:
        raw_clip = output_dir / review["raw_clip"]
        annotated_clip = output_dir / review["annotated_clip"]
        render_clip(
            ffmpeg, raw_video, raw_clip, review["clip_start_seconds"],
            review["clip_duration_seconds"],
        )
        render_clip(
            ffmpeg, annotated_video, annotated_clip, review["clip_start_seconds"],
            review["clip_duration_seconds"],
        )
        raw_clips.append(raw_clip)
        annotated_clips.append(annotated_clip)
    concat_reel(ffmpeg, raw_clips, output_dir / "raw-mismatches.mp4")
    concat_reel(ffmpeg, annotated_clips, output_dir / "annotated-mismatches.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--raw-video", type=Path)
    parser.add_argument("--annotated-video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--before-seconds", type=float, default=2.5)
    parser.add_argument("--after-seconds", type=float, default=1.5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render and (args.raw_video is None or args.annotated_video is None):
        parser.error("--render requires --raw-video and --annotated-video")
    manifest = build_manifest(
        read_json(args.report), read_jsonl(args.diagnostics), args.fps,
        args.before_seconds, args.after_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "mismatch-review.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.render:
        render_media(
            manifest, args.output_dir, args.raw_video, args.annotated_video,
        )
    print(json.dumps({
        "errors": len(manifest["reviews"]),
        "manifest": str(manifest_path),
        "rendered": args.render,
    }, indent=2))


if __name__ == "__main__":
    main()

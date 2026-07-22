#!/usr/bin/env python3
"""Serve a local timestamped hit/miss labeling tool for one video."""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import subprocess
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "evaluation-labeler"
VALID_OUTCOMES = {"hit", "miss", "uncertain"}


def parse_byte_range(value: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """Parse one HTTP byte range, returning inclusive bounds."""
    if not value:
        return None
    unit, separator, requested = value.partition("=")
    if separator != "=" or unit.strip().lower() != "bytes" or "," in requested:
        raise ValueError("unsupported range")
    start_text, separator, end_text = requested.strip().partition("-")
    if separator != "-":
        raise ValueError("invalid range")
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("invalid suffix range")
        return max(0, size - length), size - 1
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("range outside file")
    return start, min(end, size - 1)


class LabelStore:
    def __init__(self, path: Path, video: Path) -> None:
        self.path = path
        self.video = video
        self.lock = threading.Lock()

    def empty(self) -> Dict[str, Any]:
        try:
            video_name = str(self.video.relative_to(ROOT))
        except ValueError:
            video_name = str(self.video)
        return {"version": 1, "video": video_name, "labels": []}

    def read(self) -> Dict[str, Any]:
        with self.lock:
            if not self.path.exists():
                return self.empty()
            return json.loads(self.path.read_text(encoding="utf-8"))

    def validate(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict) or not isinstance(value.get("labels"), list):
            raise ValueError("body must contain a labels array")
        labels = []
        for item in value["labels"]:
            if not isinstance(item, dict) or item.get("outcome") not in VALID_OUTCOMES:
                raise ValueError("every label needs a valid outcome")
            time_seconds = item.get("time_seconds")
            if (
                isinstance(time_seconds, bool)
                or not isinstance(time_seconds, (int, float))
                or time_seconds < 0
            ):
                raise ValueError("every label needs a non-negative timestamp")
            labels.append({
                "time_seconds": round(float(time_seconds), 3),
                "outcome": item["outcome"],
            })
        labels.sort(key=lambda item: item["time_seconds"])
        result = self.empty()
        result["labels"] = labels
        return result

    def write(self, value: Any) -> Dict[str, Any]:
        checked = self.validate(value)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(checked, indent=2) + "\n", encoding="utf-8",
            )
            temporary.replace(self.path)
        return checked


def browser_video(source: Path) -> Path:
    """Create a CFR browser proxy that recovers from live-stream damage."""
    output = source.with_suffix(".browser-cfr.mp4")
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return output
    temporary = output.with_name(output.stem + ".tmp.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:v:0", "-vf", "fps=60",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-profile:v", "high", "-pix_fmt", "yuv420p", "-tag:v", "avc1",
        "-an", "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg is required to prepare browser playback") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Could not prepare browser video: {exc}") from exc
    temporary.replace(output)
    return output


def handler_for(video: Path, store: LabelStore):
    class LabelHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send_asset(UI_ROOT / "index.html")
            elif path == "/app.js":
                self.send_asset(UI_ROOT / "app.js")
            elif path == "/styles.css":
                self.send_asset(UI_ROOT / "styles.css")
            elif path == "/api/labels":
                self.send_json(store.read())
            elif path == "/video":
                self.send_video(False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:
            if urlparse(self.path).path == "/video":
                self.send_video(True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_PUT(self) -> None:
            if urlparse(self.path).path != "/api/labels":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("invalid body length")
                value = json.loads(self.rfile.read(length))
                saved = store.write(value)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_json(saved)

        def send_json(self, value: Dict[str, Any]) -> None:
            content = json.dumps(value).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_asset(self, path: Path) -> None:
            content = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_video(self, head_only: bool) -> None:
            size = video.stat().st_size
            try:
                bounds = parse_byte_range(self.headers.get("Range"), size)
            except (ValueError, TypeError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = bounds if bounds is not None else (0, size - 1)
            self.send_response(
                HTTPStatus.PARTIAL_CONTENT if bounds is not None else HTTPStatus.OK,
            )
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if bounds is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head_only:
                return
            remaining = end - start + 1
            with video.open("rb") as source:
                source.seek(start)
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)

        def log_message(self, format: str, *args: Any) -> None:
            if urlparse(self.path).path not in ("/video", "/api/labels"):
                super().log_message(format, *args)

    return LabelHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.video.resolve()
    if not source.is_file():
        raise SystemExit(f"Video does not exist: {source}")
    labels = (args.labels or source.with_suffix(".labels.json")).resolve()
    prepared = browser_video(source)
    store = LabelStore(labels, source)
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_for(prepared, store),
    )
    print(f"Evaluation labeler: http://{args.host}:{args.port}")
    print(f"Video: {source}")
    print(f"Autosave: {labels}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

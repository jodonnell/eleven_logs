#!/usr/bin/env python3
"""Serve raw live-analyzer shot events to a local browser page."""

import argparse
import json
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
COUNTER_PAGE = ROOT / "live-counter" / "index.html"
COUNTER_SCRIPT = ROOT / "live-counter" / "counter.js"
ANALYZER = ROOT / "scripts" / "analyze_video.py"


class ShotEventBroker:
    """Thread-safe raw analyzer event history and browser fan-out."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.session_id = uuid.uuid4().hex
        self._events: List[tuple[int, Dict[str, Any]]] = []
        self._subscribers: List[queue.Queue[tuple[int, Dict[str, Any]]]] = []
        self._subscriber_connected = threading.Event()
        self._source_done = threading.Event()

    def publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            event_id = len(self._events) + 1
            item = (event_id, event)
            self._events.append(item)
            for updates in self._subscribers:
                updates.put_nowait(item)

    def subscribe(
        self, after_event_id: int = 0,
    ) -> queue.Queue[tuple[int, Dict[str, Any]]]:
        updates: queue.Queue[tuple[int, Dict[str, Any]]] = queue.Queue()
        with self._lock:
            self._subscribers.append(updates)
            self._subscriber_connected.set()
            for item in self._events:
                if item[0] > after_event_id:
                    updates.put_nowait(item)
        return updates

    def wait_for_subscriber(self) -> None:
        self._subscriber_connected.wait()

    def mark_source_done(self) -> None:
        self._source_done.set()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"done": self._source_done.is_set(), "messages": len(self._events)}

    def resume_index(self, last_event_id: Optional[str]) -> int:
        """Resume only when the browser's event ID belongs to this process."""
        if not last_event_id:
            return 0
        session_id, separator, index = last_event_id.rpartition(":")
        if separator != ":" or session_id != self.session_id:
            return 0
        try:
            return max(0, int(index))
        except ValueError:
            return 0

    def stream_id(self, event_id: int) -> str:
        return f"{self.session_id}:{event_id}"

    def unsubscribe(
        self, updates: queue.Queue[tuple[int, Dict[str, Any]]],
    ) -> None:
        with self._lock:
            if updates in self._subscribers:
                self._subscribers.remove(updates)


def handler_for(events: ShotEventBroker):
    class CounterHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_file(COUNTER_PAGE, "text/html; charset=utf-8")
            elif path == "/counter.js":
                self._send_file(COUNTER_SCRIPT, "text/javascript; charset=utf-8")
            elif path == "/events":
                self._send_events()
            elif path == "/status":
                self._send_json(events.status())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                content = path.read_bytes()
            except OSError as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, value: Dict[str, Any]) -> None:
            content = json.dumps(value).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _send_events(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_event_id = events.resume_index(self.headers.get("Last-Event-ID"))
            updates = events.subscribe(last_event_id)
            try:
                while True:
                    try:
                        event_id, event = updates.get(timeout=15)
                        payload = (
                            f"id: {events.stream_id(event_id)}\n"
                            f"data: {json.dumps(event)}\n\n"
                        )
                    except queue.Empty:
                        payload = ": keepalive\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                events.unsubscribe(updates)

        def log_message(self, format: str, *args: Any) -> None:
            if self.path.split("?", 1)[0] not in ("/events", "/status"):
                super().log_message(format, *args)

    return CounterHandler


def analyzer_command(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(ANALYZER),
        args.video,
        "--live-stdout",
        "--output",
        args.output,
    ]
    if args.calibration:
        command.extend(["--calibration", args.calibration])
    if args.annotated:
        command.extend(["--annotated", args.annotated])
    if args.clean_recording:
        command.extend([
            "--clean-recording", args.clean_recording,
            "--clean-recording-seconds", str(args.clean_recording_seconds),
            "--clean-recording-start", args.clean_recording_start,
        ])
    if args.live_events:
        command.extend(["--live-events", args.live_events])
    return command


def read_analyzer(
    process: subprocess.Popen[str], events: ShotEventBroker,
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"Ignoring non-JSON analyzer output: {line.rstrip()}", file=sys.stderr)
            continue
        if isinstance(event, dict):
            events.publish(event)
    process.wait()


def run_analyzer(
    args: argparse.Namespace,
    events: ShotEventBroker,
    process_holder: List[subprocess.Popen[str]],
) -> None:
    if args.wait_for_subscriber:
        events.wait_for_subscriber()
    process = subprocess.Popen(
        analyzer_command(args),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    process_holder.append(process)
    try:
        read_analyzer(process, events)
    finally:
        events.mark_source_done()


def replay_events(path: Path, interval_seconds: float, events: ShotEventBroker) -> None:
    """Publish a deterministic JSONL session after a browser subscribes."""
    events.wait_for_subscriber()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        time.sleep(interval_seconds)
        events.publish(json.loads(line))
    events.mark_source_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", nargs="?", help="video file or srt:// URL passed to the analyzer",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--calibration", help="optional analyzer calibration JSON")
    parser.add_argument(
        "--annotated",
        nargs="?",
        const="video_bounces_annotated.mp4",
        help="write analyzer diagnostics, optionally to a custom MP4 path",
    )
    parser.add_argument(
        "--clean-recording",
        help="bounded clean detector-input MP4 forwarded to the analyzer",
    )
    parser.add_argument(
        "--clean-recording-seconds",
        type=float,
        default=120,
        help="maximum clean recording length (default: 120 seconds)",
    )
    parser.add_argument(
        "--clean-recording-start",
        choices=("launch", "immediate"),
        default="launch",
        help="when the analyzer starts the clean recording",
    )
    parser.add_argument(
        "--live-events",
        help="append-only live publication JSONL forwarded to the analyzer",
    )
    parser.add_argument("--output", default="video_bounces.jsonl")
    parser.add_argument(
        "--replay-events",
        type=Path,
        help="serve deterministic JSONL messages instead of running the analyzer",
    )
    parser.add_argument(
        "--replay-interval-ms",
        type=float,
        default=200,
        help="delay between replayed browser messages (default: 200ms)",
    )
    parser.add_argument(
        "--wait-for-subscriber",
        action="store_true",
        help="open the analyzer source only after a browser connects",
    )
    args = parser.parse_args()
    if args.video is None and args.replay_events is None:
        parser.error("video is required unless --replay-events is supplied")
    if args.replay_interval_ms < 0:
        parser.error("--replay-interval-ms cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    events = ShotEventBroker()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(events))
    process_holder: List[subprocess.Popen[str]] = []
    if args.replay_events is not None:
        reader = threading.Thread(
            target=replay_events,
            args=(args.replay_events, args.replay_interval_ms / 1000, events),
            daemon=True,
        )
    else:
        reader = threading.Thread(
            target=run_analyzer,
            args=(args, events, process_holder),
            daemon=True,
        )
    reader.start()
    print(f"Hit counter: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        process = process_holder[0] if process_holder else None
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()


if __name__ == "__main__":
    main()

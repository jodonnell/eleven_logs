#!/usr/bin/env python3
"""Serve live analyzer attempt-ledger upserts to a local browser page."""

import argparse
import base64
import json
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import cv2

from video_source import open_video_source


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

    def reset(self) -> None:
        """Begin a new source run while keeping browser subscribers connected."""
        with self._lock:
            self.session_id = uuid.uuid4().hex
            self._events.clear()
            self._source_done.clear()
            for updates in self._subscribers:
                while True:
                    try:
                        updates.get_nowait()
                    except queue.Empty:
                        break

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


class PreviewFrameBroker:
    """Keep only the newest annotated JPEG for browser MJPEG clients."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0
        self._jpeg: Optional[bytes] = None

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._version += 1
            self._condition.notify_all()

    def reset(self) -> None:
        with self._condition:
            self._jpeg = None
            self._version += 1
            self._condition.notify_all()

    def next_frame(
        self, after_version: int, timeout: float = 15,
    ) -> tuple[int, Optional[bytes]]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._version > after_version,
                timeout=timeout,
            )
            return self._version, self._jpeg


def handler_for(
    events: ShotEventBroker,
    preview: Optional[PreviewFrameBroker] = None,
    restart: Optional[Callable[[], None]] = None,
):
    class CounterHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_file(COUNTER_PAGE, "text/html; charset=utf-8")
            elif path == "/counter.js":
                self._send_file(COUNTER_SCRIPT, "text/javascript; charset=utf-8")
            elif path == "/events":
                self._send_events()
            elif path == "/preview.mjpg" and preview is not None:
                self._send_preview()
            elif path == "/status":
                self._send_json(events.status())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/restart" or restart is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                restart()
            except Exception as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self._send_json({"status": "restarted"})

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

        def _send_preview(self) -> None:
            assert preview is not None
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            version = 0
            try:
                while True:
                    next_version, jpeg = preview.next_frame(version)
                    if next_version == version or jpeg is None:
                        continue
                    version = next_version
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

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
            "--clean-recording-codec", args.clean_recording_codec,
        ])
    if args.live_events:
        command.extend(["--live-events", args.live_events])
    if args.realtime:
        command.append("--realtime")
    if getattr(args, "preview", False):
        command.extend([
            "--preview-stdout",
            "--preview-fps",
            str(getattr(args, "preview_fps", 12)),
        ])
    return command


def lan_address_for(video: Optional[str]) -> Optional[str]:
    """Return the local address routed toward the video sender."""
    peer = urlparse(video).hostname if video else None
    if peer is None:
        return None
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect selects a route without sending a packet.
        connection.connect((peer, 9))
        address = connection.getsockname()[0]
        return address if address and address != "0.0.0.0" else None
    except OSError:
        return None
    finally:
        connection.close()


def counter_urls(host: str, port: int, video: Optional[str]) -> List[str]:
    if host not in ("0.0.0.0", "::", ""):
        return [f"http://{host}:{port}"]
    urls = [f"http://127.0.0.1:{port}"]
    lan_address = lan_address_for(video)
    if lan_address is not None and lan_address != "127.0.0.1":
        urls.append(f"http://{lan_address}:{port}")
    return urls


def read_analyzer(
    process: subprocess.Popen[str],
    events: ShotEventBroker,
    preview: PreviewFrameBroker,
) -> int:
    assert process.stdout is not None
    for line in process.stdout:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"Ignoring non-JSON analyzer output: {line.rstrip()}", file=sys.stderr)
            continue
        if isinstance(event, dict):
            if event.get("type") == "_preview_frame":
                encoded = event.get("jpeg_base64")
                if isinstance(encoded, str):
                    try:
                        preview.publish(base64.b64decode(encoded, validate=True))
                    except ValueError:
                        print("Ignoring invalid browser preview frame", file=sys.stderr)
            else:
                events.publish(event)
    return process.wait()


def run_analyzer(
    args: argparse.Namespace,
    events: ShotEventBroker,
    process_holder: List[subprocess.Popen[str]],
    preview: Optional[PreviewFrameBroker] = None,
) -> None:
    if preview is None:
        preview = PreviewFrameBroker()
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
    returncode: Optional[int] = None
    try:
        returncode = read_analyzer(process, events, preview)
    finally:
        if returncode is None:
            returncode = process.poll()
        exit_record = {
            "type": "analyzer_exit",
            "returncode": returncode,
            "logged_at_unix_seconds": round(time.time(), 3),
        }
        if args.live_events:
            with open(args.live_events, "a", encoding="utf-8") as output:
                output.write(json.dumps(exit_record) + "\n")
        events.publish(exit_record)
        print(
            f"Analyzer exited with status {returncode}",
            file=sys.stderr,
            flush=True,
        )
        events.mark_source_done()


def stop_analyzer(
    process: Optional[subprocess.Popen[str]],
    timeout_seconds: float = 10,
) -> None:
    """Interrupt the analyzer and escalate only if it misses the deadline."""
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait()


def replay_events(path: Path, interval_seconds: float, events: ShotEventBroker) -> None:
    """Publish a deterministic JSONL session after a browser subscribes."""
    events.wait_for_subscriber()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        time.sleep(interval_seconds)
        events.publish(json.loads(line))
    events.mark_source_done()


def preview_video(
    path: str,
    realtime: bool,
    preview_fps: float,
    wait_for_subscriber: bool,
    events: ShotEventBroker,
    preview: PreviewFrameBroker,
    stop: threading.Event,
) -> None:
    """Stream an unannotated local video without starting the detector."""
    if wait_for_subscriber:
        events.wait_for_subscriber()
    source = None
    try:
        source = open_video_source(path, realtime=realtime)
        interval = max(1, round(source.fps / preview_fps))
        events.publish({
            "type": "preview_only",
            "message": "Local video preview — detector disabled",
        })
        while not stop.is_set():
            frame = source.read()
            if frame is None:
                break
            if frame.number % interval:
                continue
            image = frame.image
            if image.shape[1] > 1280:
                scale = 1280 / image.shape[1]
                image = cv2.resize(
                    image,
                    (1280, max(1, round(image.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            encoded, jpeg = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 72],
            )
            if encoded:
                preview.publish(jpeg.tobytes())
    except Exception as exc:
        print(f"Video preview failed: {exc}", file=sys.stderr, flush=True)
        events.publish({"type": "preview_error", "message": str(exc)})
    finally:
        if source is not None:
            source.close()
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
        "--clean-recording-codec",
        choices=("ffv1", "mjpeg"),
        default="ffv1",
        help="lossless FFV1 or lower-overhead MJPEG capture (default: ffv1)",
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
        "--preview-only",
        action="store_true",
        help="stream a local video without detector analysis or calibration",
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
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace prerecorded analysis against wall-clock time",
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="disable the annotated browser video preview",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=12,
        help="maximum annotated browser preview rate (default: 12 FPS)",
    )
    parser.set_defaults(preview=True)
    args = parser.parse_args()
    if args.video is None and args.replay_events is None:
        parser.error("video is required unless --replay-events is supplied")
    if args.preview_only and args.replay_events is not None:
        parser.error("--preview-only cannot be combined with --replay-events")
    if (
        args.preview_only
        and args.video is not None
        and args.video.lower().startswith("srt://")
    ):
        parser.error("--preview-only accepts a local video file, not an SRT URL")
    if args.replay_interval_ms < 0:
        parser.error("--replay-interval-ms cannot be negative")
    if args.preview_fps <= 0:
        parser.error("--preview-fps must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    events = ShotEventBroker()
    preview = PreviewFrameBroker()
    process_holder: List[subprocess.Popen[str]] = []
    reader_holder: List[threading.Thread] = []
    stop_holder: List[threading.Event] = []
    restart_lock = threading.Lock()

    def start_source(reset: bool = False) -> None:
        with restart_lock:
            if stop_holder:
                stop_holder[-1].set()
            existing_process = process_holder[-1] if process_holder else None
            stop_analyzer(existing_process)
            if reader_holder and reader_holder[-1].is_alive():
                reader_holder[-1].join(timeout=10)
            if reset:
                events.reset()
                preview.reset()
                events.publish({
                    "type": "session_reset",
                    "logged_at_unix_seconds": round(time.time(), 3),
                })
            process_holder.clear()
            if args.replay_events is not None:
                reader = threading.Thread(
                    target=replay_events,
                    args=(
                        args.replay_events,
                        args.replay_interval_ms / 1000,
                        events,
                    ),
                    daemon=True,
                )
            elif args.preview_only:
                stop = threading.Event()
                stop_holder[:] = [stop]
                reader = threading.Thread(
                    target=preview_video,
                    args=(
                        args.video,
                        args.realtime,
                        args.preview_fps,
                        args.wait_for_subscriber,
                        events,
                        preview,
                        stop,
                    ),
                    daemon=True,
                )
            else:
                reader = threading.Thread(
                    target=run_analyzer,
                    args=(args, events, process_holder, preview),
                    daemon=True,
                )
            reader_holder[:] = [reader]
            reader.start()

    server = ThreadingHTTPServer(
        (args.host, args.port), handler_for(events, preview, lambda: start_source(True)),
    )
    start_source()
    urls = counter_urls(args.host, args.port, args.video)
    print(f"Hit counter on this Mac: {urls[0]}", flush=True)
    if len(urls) > 1:
        print(f"Hit counter on Quest/LAN: {urls[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if stop_holder:
            stop_holder[-1].set()
        process = process_holder[0] if process_holder else None
        stop_analyzer(process)
        # serve_forever() ran on this thread and has already returned. Calling
        # shutdown() here would wait forever for a loop that is no longer
        # running.
        server.server_close()


if __name__ == "__main__":
    main()

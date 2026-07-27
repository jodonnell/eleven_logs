"""Video processing orchestration and command-line argument handling."""

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

from .models import (
    BounceEvent,
    Calibration,
    DetectorDiagnostics,
    DetectorSettings,
    PROCESSING_WIDTH,
    TrackDiagnostic,
)
from .detection import MultiBallTracker, calibration_geometry, load_calibration
from .telemetry import TelemetryReader
from .vision import candidates_for_frame, draw_overlay
from .classifier import AttemptClassifier
from .normalization import LiveAttemptNormalizer, normalize_attempt_events
from .live import (
    DirectLiveAttemptPublisher,
    LiveCounterHealthMonitor,
    LivePipelineLogger,
    attach_missing_machine_telemetry,
    create_video_writer,
    reset_output_file,
)

def process_video(
    source: VideoSource,
    scale: float,
    calibration: Calibration,
    homography: np.ndarray,
    table: np.ndarray,
    end_seconds: Optional[float] = None,
    writer: Optional[cv2.VideoWriter] = None,
    first_frame: Optional[VideoFrame] = None,
    on_event: Optional[Callable[[BounceEvent], None]] = None,
    on_attempt_finished: Optional[Callable[[Optional[int]], None]] = None,
    on_confirmed_hit: Optional[Callable[[BounceEvent], None]] = None,
    on_confirmed_non_hit: Optional[Callable[[BounceEvent], None]] = None,
    on_processing_frame: Optional[Callable[[int], None]] = None,
    clean_writer: Optional[cv2.VideoWriter] = None,
    clean_frame_limit: Optional[int] = None,
    clean_start_on_launch: bool = False,
    on_attempt_started: Optional[Callable[[int], None]] = None,
    on_track_diagnostic: Optional[
        Callable[[TrackDiagnostic, int], None]
    ] = None,
    on_frame_processed: Optional[
        Callable[[int, Dict[str, Any]], None]
    ] = None,
    on_preview_frame: Optional[Callable[[int, np.ndarray], None]] = None,
) -> List[BounceEvent]:
    """Process an already-open source and return its detected bounce events."""
    fps = source.fps
    video_width, video_height = source.width, source.height
    width, height = round(video_width * scale), round(video_height * scale)
    settings = DetectorSettings.from_calibration(calibration)
    net_line = np.asarray(calibration["net_line"], dtype=np.float32) * scale
    occlusion = np.asarray(
        calibration.get("occlusion_polygon", []), dtype=np.float32,
    ) * scale
    tracking_polygon = np.asarray(
        calibration["tracking_polygon"], dtype=np.float32,
    ) * scale
    contact_polygon = np.asarray(
        calibration.get("table_contact_polygon", calibration["table_polygon"]),
        dtype=np.float32,
    ) * scale
    tracker = MultiBallTracker(settings)
    diagnostics = (
        DetectorDiagnostics(track_lifetime_frames=max(1, round(fps * .5)))
        if writer is not None or on_preview_frame is not None else None
    )
    telemetry = TelemetryReader()
    clean_recording_started = not clean_start_on_launch

    def mark_clean_recording_started(frame_number: int) -> None:
        nonlocal clean_recording_started
        clean_recording_started = True
        if on_attempt_started is not None:
            on_attempt_started(frame_number)

    def report_track(diagnostic: TrackDiagnostic, frame_number: int) -> None:
        if diagnostics is not None:
            diagnostics.completed_track(diagnostic, frame_number)
        if on_track_diagnostic is not None:
            on_track_diagnostic(diagnostic, frame_number)

    classifier = AttemptClassifier(
        fps, calibration, contact_polygon, net_line, occlusion, homography,
        video_width, video_height, scale, settings, on_event,
        on_attempt_finished, on_confirmed_hit, on_confirmed_non_hit,
        report_track if diagnostics is not None or on_track_diagnostic is not None else None,
        mark_clean_recording_started,
    )
    previous_gray = None
    next_frame = first_frame
    frame_number = first_frame.number if first_frame is not None else 0
    clean_frames_written = 0
    clean_seed_written = False
    try:
        while True:
            video_frame = next_frame if next_frame is not None else source.read()
            next_frame = None
            if video_frame is None or (
                end_seconds is not None
                and video_frame.time_seconds >= end_seconds
            ):
                break
            frame_number = video_frame.number
            original = video_frame.image
            if on_processing_frame is not None:
                on_processing_frame(frame_number)
            if (
                clean_writer is not None
                and clean_start_on_launch
                and not clean_seed_written
            ):
                # Keep one stable setup frame for automatic calibration. The
                # continuous bounded recording still waits for detector
                # activity, so a long headset setup does not consume it.
                clean_seed_written = True
            if frame_number % 3 == 0:
                reading = telemetry.update(original, frame_number)
                if reading is not None:
                    classifier.observe_telemetry(reading)
            frame = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
            wrote_clean_seed = (
                clean_writer is not None
                and clean_start_on_launch
                and clean_frames_written == 0
                and clean_seed_written
            )
            if wrote_clean_seed and clean_writer is not None:
                clean_writer.write(frame)
                clean_frames_written += 1
            elif (
                clean_writer is not None
                and clean_recording_started
                and (
                    clean_frame_limit is None
                    or clean_frames_written < clean_frame_limit
                )
            ):
                # Store the exact resized detector input losslessly before any
                # diagnostics are drawn. This makes offline replay pixel-equivalent.
                clean_writer.write(frame)
                clean_frames_written += 1
            if diagnostics is not None:
                diagnostics.begin_frame()
            gray, candidates = candidates_for_frame(
                frame, previous_gray, tracking_polygon, settings, diagnostics,
            )
            previous_gray = gray
            completed_tracks = tracker.update(frame_number, candidates)
            classifier.process_tracks(completed_tracks, frame_number)
            classifier.process_active_tracks(tracker.confirmed_tracks, frame_number)
            if diagnostics is not None:
                diagnostics.set_unconfirmed_tracks(tracker.tracks)
            if writer is not None or on_preview_frame is not None:
                annotated_frame = draw_overlay(
                    frame, table, net_line, tracker.visible_points, classifier.events,
                    homography, calibration["table_surface_y"],
                    diagnostics, frame_number,
                )
                if writer is not None:
                    writer.write(annotated_frame)
                if on_preview_frame is not None:
                    on_preview_frame(frame_number, annotated_frame)
            if on_frame_processed is not None:
                on_frame_processed(frame_number, {
                    "candidate_count": len(candidates),
                    "active_track_count": len(tracker.tracks),
                    "confirmed_track_count": len(tracker.confirmed_tracks),
                    "completed_track_count": len(completed_tracks),
                    "detected_event_count": len(classifier.events),
                })
            frame_number += 1
    except KeyboardInterrupt:
        # A live source normally ends when the user stops it. Preserve and
        # flush the completed session instead of discarding every event.
        pass
    if on_processing_frame is not None:
        on_processing_frame(frame_number)
    classifier.finish_attempt(frame_number)
    normalized = normalize_attempt_events(classifier.events, frame_number, fps)
    return attach_missing_machine_telemetry(
        normalized, classifier.telemetry_history, fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="video file or srt:// URL")
    parser.add_argument("--calibration", help="Optional manually reviewed JSON calibration")
    parser.add_argument("--extract-calibration-frame", metavar="PNG", help="write a frame for per-camera corner calibration, then exit")
    parser.add_argument("--output", default="video_bounces.jsonl")
    parser.add_argument(
        "--live-stdout",
        action="store_true",
        help="print each detected event immediately instead of watching the JSONL file",
    )
    parser.add_argument(
        "--annotated",
        nargs="?",
        const="video_bounces_annotated.mp4",
        metavar="MP4",
        help="write annotated video, optionally to a custom path",
    )
    parser.add_argument(
        "--clean-recording",
        metavar="MP4",
        help="record clean decoded detector input before overlays are drawn",
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
        help="start clean capture at the first detected launch (default) or immediately",
    )
    parser.add_argument(
        "--clean-recording-codec",
        choices=("ffv1", "mjpeg"),
        default="ffv1",
        help="lossless FFV1 or lower-overhead MJPEG capture (default: ffv1)",
    )
    parser.add_argument(
        "--live-events",
        metavar="JSONL",
        help="append-only live publication log with shot and publication frames",
    )
    parser.add_argument(
        "--bounce-diagnostics",
        metavar="JSONL",
        help="write observation-only confirmed-bounce paths and signal kinds",
    )
    parser.add_argument(
        "--contact-diagnostics",
        metavar="JSONL",
        help="write ordered contact candidates without changing classification",
    )
    parser.add_argument(
        "--track-diagnostics",
        metavar="JSONL",
        help="write every completed track decision and rejection reason",
    )
    parser.add_argument("--no-annotated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-seconds", type=float, default=0, help="seek point; useful when reviewing a short interval")
    parser.add_argument("--end-seconds", type=float, help="stop after this video timestamp")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace prerecorded input against wall-clock time",
    )
    parser.add_argument(
        "--preview-stdout",
        action="store_true",
        help="emit throttled JPEG detector previews on stdout for the counter server",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=12,
        help="maximum browser preview rate (default: 12 FPS)",
    )
    args = parser.parse_args()
    if args.clean_recording_seconds <= 0:
        parser.error("--clean-recording-seconds must be greater than zero")
    if args.preview_fps <= 0:
        parser.error("--preview-fps must be greater than zero")
    annotated_path = None if args.no_annotated else args.annotated
    if not args.extract_calibration_frame:
        # Live SRT sources can block while waiting for the sender. Clear stale
        # events before opening the source so observers immediately see that a
        # new server session has begun.
        reset_output_file(args.output)
        if args.live_events is not None:
            reset_output_file(args.live_events)
    try:
        source = open_video_source(args.video, realtime=args.realtime)
        try:
            source.seek_seconds(args.start_seconds)
        except VideoSourceError:
            source.close()
            raise
    except VideoSourceError as exc:
        raise SystemExit(str(exc)) from exc
    fps = source.fps
    video_width, video_height = source.width, source.height
    scale = min(1.0, PROCESSING_WIDTH / video_width)
    width, height = round(video_width * scale), round(video_height * scale)
    if args.extract_calibration_frame:
        video_frame = source.read()
        source.close()
        if video_frame is None:
            raise SystemExit("Could not read a calibration frame")
        if not cv2.imwrite(args.extract_calibration_frame, video_frame.image):
            raise SystemExit(f"Could not write calibration frame to {args.extract_calibration_frame}")
        print(json.dumps({"calibration_frame": args.extract_calibration_frame, "image_size": [video_width, video_height]}, indent=2))
        return
    writer = None
    clean_writer = None
    live_events_output = None
    bounce_diagnostics_output = None
    contact_diagnostics_output = None
    track_diagnostics_output = None
    first_frame = None
    try:
        if args.calibration:
            calibration, homography, table = load_calibration(
                args.calibration, video_width, video_height, scale,
            )
        else:
            first_frame = source.read()
            if first_frame is None:
                raise SystemExit("Could not read the first frame for automatic calibration")
            try:
                detected, _ = calibration_from_frame(
                    first_frame.image, first_frame.number,
                )
            except ValueError as exc:
                raise SystemExit(f"Automatic calibration failed: {exc}.") from exc
            calibration_start_frame = first_frame.number
            calibration_frame_image = first_frame.image

            def color_calibration_frames() -> Iterable[np.ndarray]:
                yield calibration_frame_image
                for _ in range(max(0, round(fps * 8) - 1)):
                    sampled = source.read()
                    if sampled is None:
                        break
                    yield sampled.image

            # Ball hue needs motion evidence and therefore cannot come from the
            # single setup frame used for table geometry. Saved files rewind
            # after this prefix; a live stream intentionally treats it as a
            # short color-calibration warm-up.
            ball_color = infer_ball_color(color_calibration_frames(), detected)
            if ball_color is not None:
                detected["ball_color"] = ball_color
            if source.seekable:
                source.seek_frame(calibration_start_frame)
                first_frame = source.read()
                if first_frame is None:
                    raise SystemExit("Could not rewind after automatic color calibration")
            else:
                first_frame = None
            calibration, homography, table = calibration_geometry(
                detected, video_width, video_height, scale,
            )
        if annotated_path is not None:
            writer = create_video_writer(annotated_path, fps, (width, height))
        if args.clean_recording is not None:
            if Path(args.clean_recording).suffix.lower() != ".mkv":
                raise SystemExit("--clean-recording must use an .mkv path")
            clean_codec = "FFV1" if args.clean_recording_codec == "ffv1" else "MJPG"
            clean_writer = create_video_writer(
                args.clean_recording, fps, (width, height), codec=clean_codec,
            )
        if args.live_events is not None:
            live_events_output = open(args.live_events, "a", encoding="utf-8")
        if args.bounce_diagnostics is not None:
            reset_output_file(args.bounce_diagnostics)
            bounce_diagnostics_output = open(
                args.bounce_diagnostics, "a", encoding="utf-8",
            )
        if args.contact_diagnostics is not None:
            reset_output_file(args.contact_diagnostics)
            contact_diagnostics_output = open(
                args.contact_diagnostics, "a", encoding="utf-8",
            )
        if args.track_diagnostics is not None:
            reset_output_file(args.track_diagnostics)
            track_diagnostics_output = open(
                args.track_diagnostics, "a", encoding="utf-8",
            )
        with open(args.output, "w", encoding="utf-8") as output:
            processing_frame = first_frame.number if first_frame is not None else 0
            live_normalizer: Optional[Any] = None
            health_monitor: Optional[LiveCounterHealthMonitor] = None
            live_stdout_open = args.live_stdout
            attempt_states: Dict[str, str] = {}
            attempt_upsert_count = 0

            def observe_processing_frame(frame_number: int) -> None:
                nonlocal processing_frame
                processing_frame = frame_number
                if live_normalizer is not None:
                    live_normalizer.advance(frame_number)

            def write_live_record(record: Dict[str, Any]) -> None:
                nonlocal live_stdout_open
                serialized = json.dumps(record)
                if live_events_output is not None:
                    live_events_output.write(serialized + "\n")
                    live_events_output.flush()
                if live_stdout_open:
                    try:
                        print(serialized, flush=True)
                    except BrokenPipeError:
                        live_stdout_open = False
                        sys.stdout = open("/dev/null", "w", encoding="utf-8")

            if source.live:
                source.set_event_callback(write_live_record)
                health_monitor = LiveCounterHealthMonitor(
                    write_live_record,
                    publisher_event_threshold=12,
                )

            def write_attempt(attempt: Dict[str, Any]) -> None:
                nonlocal attempt_upsert_count
                record = {"type": "attempt_upsert", **attempt}
                record["publication_frame_number"] = processing_frame
                record["publication_video_time_seconds"] = round(
                    processing_frame / fps, 3,
                )
                if attempt["state"] == "finalized":
                    evidence_frame = attempt.get(
                        "frame_number", attempt["anchor_frame_number"],
                    )
                    decision_frame = attempt.get(
                        "decision_frame_number", evidence_frame,
                    )
                    record["publication_delay_frames"] = (
                        processing_frame - evidence_frame
                    )
                    record["publication_delay_seconds"] = round(
                        (processing_frame - evidence_frame) / fps, 3,
                    )
                    record["attempt_publication_delay_seconds"] = round(
                        (processing_frame - attempt["anchor_frame_number"]) / fps,
                        3,
                    )
                    record["decision_publication_delay_seconds"] = round(
                        (processing_frame - decision_frame) / fps,
                        3,
                    )
                    record["feedback_delay_seconds"] = (
                        record["publication_delay_seconds"]
                        if attempt.get("outcome") == "hit"
                        else record["decision_publication_delay_seconds"]
                    )
                attempt_upsert_count += 1
                attempt_states[record["attempt_id"]] = record["state"]
                write_live_record(record)
                if health_monitor is not None:
                    health_monitor.observe_attempt(record)

            # Detector-native launch fragments create extra misses and omit
            # fully occluded launches. Establish cadence from several distinct
            # contacts before publishing a stable live ledger; status messages
            # keep startup visible while evidence accumulates.
            live_normalizer = LiveAttemptNormalizer(
                fps,
                write_attempt,
                minimum_cadence_hits=6,
                on_status=write_live_record if source.live else None,
            )
            pipeline_logger = None
            if source.live and live_events_output is not None:
                def write_pipeline_record(record: Dict[str, Any]) -> None:
                    live_events_output.write(json.dumps(record) + "\n")
                    live_events_output.flush()

                pipeline_logger = LivePipelineLogger(
                    fps, write_pipeline_record,
                )

            def write_pipeline_heartbeat(
                frame_number: int,
                metrics: Dict[str, Any],
            ) -> None:
                live_metrics = {
                    **metrics,
                    "attempt_upsert_count": attempt_upsert_count,
                    "attempt_count": len(attempt_states),
                    "pending_attempt_count": sum(
                        state == "pending" for state in attempt_states.values()
                    ),
                    "finalized_attempt_count": sum(
                        state == "finalized" for state in attempt_states.values()
                    ),
                }
                if pipeline_logger is not None:
                    pipeline_logger.observe(frame_number, live_metrics)
                if health_monitor is not None:
                    health_monitor.observe_pipeline(live_metrics)

            preview_interval_frames = max(1, round(fps / args.preview_fps))

            def write_preview_frame(
                frame_number: int,
                frame: np.ndarray,
            ) -> None:
                nonlocal live_stdout_open
                if (
                    not args.preview_stdout
                    or not live_stdout_open
                    or frame_number % preview_interval_frames
                ):
                    return
                encoded, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72],
                )
                if not encoded:
                    return
                record = {
                    "type": "_preview_frame",
                    "frame_number": frame_number,
                    "video_time_seconds": round(frame_number / fps, 3),
                    "jpeg_base64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                }
                try:
                    print(json.dumps(record, separators=(",", ":")), flush=True)
                except BrokenPipeError:
                    live_stdout_open = False
                    sys.stdout = open("/dev/null", "w", encoding="utf-8")

            def write_bounce_diagnostic(
                diagnostic: TrackDiagnostic, draw_frame: int,
            ) -> None:
                if track_diagnostics_output is not None:
                    track_diagnostics_output.write(json.dumps({
                        "draw_frame": draw_frame,
                        "kind": diagnostic.kind,
                        "reason": diagnostic.reason,
                        "points": diagnostic.points,
                    }) + "\n")
                    track_diagnostics_output.flush()
                if diagnostic.kind in (
                    "contact_candidate", "rejected_contact_candidate",
                ):
                    if contact_diagnostics_output is None:
                        return
                    record = json.loads(diagnostic.reason)
                    record["draw_frame"] = draw_frame
                    contact_diagnostics_output.write(
                        json.dumps(record) + "\n"
                    )
                    contact_diagnostics_output.flush()
                    return
                if bounce_diagnostics_output is None or diagnostic.kind not in (
                    "confirmed_bounce", "association",
                ):
                    return
                bounce_diagnostics_output.write(json.dumps({
                    "draw_frame": draw_frame,
                    "kind": diagnostic.kind,
                    "reason": diagnostic.reason,
                    "points": diagnostic.points,
                }) + "\n")
                bounce_diagnostics_output.flush()

            events = process_video(
                source, scale, calibration, homography, table,
                args.end_seconds, writer, first_frame,
                live_normalizer.observe, live_normalizer.settle_attempt,
                live_normalizer.observe_confirmed_hit,
                live_normalizer.observe_confirmed_non_hit,
                observe_processing_frame,
                clean_writer,
                round(args.clean_recording_seconds * fps),
                args.clean_recording_start == "launch",
                on_attempt_started=getattr(
                    live_normalizer, "observe_attempt_started", None,
                ),
                on_track_diagnostic=write_bounce_diagnostic,
                on_frame_processed=write_pipeline_heartbeat,
                on_preview_frame=(
                    write_preview_frame if args.preview_stdout else None
                ),
            )
            live_normalizer.finish_session(processing_frame)
            if pipeline_logger is not None:
                pipeline_logger.finish(processing_frame, "processing_ended")

            # Cadence-based normalization can rename events and infer missed
            # launches only after enough of the session is known. Preserve the
            # canonical analysis file independently of the live ledger log.
            output.seek(0)
            output.truncate()
            for event in events:
                output.write(json.dumps(event.to_record()) + "\n")
            output.flush()
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if clean_writer is not None:
            clean_writer.release()
        if live_events_output is not None:
            live_events_output.close()
        if bounce_diagnostics_output is not None:
            bounce_diagnostics_output.close()
        if contact_diagnostics_output is not None:
            contact_diagnostics_output.close()
        if track_diagnostics_output is not None:
            track_diagnostics_output.close()
    print(json.dumps({
        "events": len(events),
        "output": args.output,
        "annotated": annotated_path,
        "clean_recording": args.clean_recording,
        "live_events": args.live_events,
        "bounce_diagnostics": args.bounce_diagnostics,
        "contact_diagnostics": args.contact_diagnostics,
    }, indent=2), file=sys.stderr if args.live_stdout else sys.stdout)

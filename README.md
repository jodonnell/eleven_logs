# Eleven Practice

## Run tests

npm run test

## Format JavaScript tests

npm run prettier

## Lint

npm run lint

## Ball-machine video analysis

### Capture evaluation footage

Record long, previously unseen evaluation footage without running the detector
or losslessly re-encoding frames in the live path:

```sh
npm run capture:evaluation
```

This copies the incoming compressed SRT video directly into a timestamped MKV
under `artifacts/`, stops after 20 minutes as a safety bound, and can be stopped
earlier with Ctrl-C. Analyze the saved file offline after recording. Do not use
the live counter's `--clean-recording` option for long evaluation sessions: the
detector can process more slowly than real time and eventually overflow SRT's
receive buffer.

Label a saved evaluation recording with timestamped ground truth:

```sh
npm run label:evaluation -- artifacts/evaluation-session.mkv
```

Open the printed local URL. Press `H`, `M`, or `U` as each attempt finishes;
`Z` undoes the last edit, arrow keys seek, and `A` advances through uncertain
labels. Labels autosave beside the source video as
`*.labels.json`. The labeler prepares a constant-frame-rate browser proxy so
isolated damage or irregular timestamps in a captured live stream cannot stall
browser playback. The original detector input remains untouched. Outcome times
are not assumed to follow launcher cadence: net clips and other delayed contacts
can resolve close to a later ball. Missing and duplicate labels must instead be
checked by matching the ordered ledger to independently located launches.

Compare detector JSONL with a timestamped ground-truth export and preserve both
machine-readable and review-friendly reports:

```sh
npm run evaluate:detector -- ground-truth.json detector-output.jsonl \
  --json-output evaluation-report.json \
  --markdown-output evaluation-report.md
```

The report separates wrong hit/miss decisions from missing and extra launches,
so one dropped launch does not shift every later comparison.

### Live hit counter

Start the analyzer and its local browser counter together by passing a video
file or live SRT URL:

```sh
npm run counter -- 'srt://OBS_IP:9000?mode=caller&latency=120000'
```

Then open <http://127.0.0.1:8000>. The server streams keyed attempt upserts to
the page, where the session count increases for every finalized `hit` and
resets to zero for a finalized `miss` or `out`. After six distinct contacts
establish a stable launcher rhythm, every inferred launch receives
a stable attempt ID. The newest slot remains pending until direct evidence, the
next credible launch, or a conservative cadence deadline finalizes it.
Refreshing the page replays these keyed attempt upserts so the browser can
reconstruct the same finalized ledger and streak.
The page also keeps the all-time best streak in that browser's local storage,
so it survives refreshes and server restarts without requiring a database.
Session stats show the hit percentage as hits over finalized attempts. Average
player-return speed and spin include only successful hits whose TV telemetry
was read; misses, outs, and attempts without trusted on-screen OCR values are
excluded. A separate last-ball row shows the newest finalized hit or miss and
its player-return speed and spin when available.

The first six distinct contacts remain buffered long enough to infer a stable
cadence, so startup can publish those initial attempts several seconds late.
The page reports calibration progress during this warm-up.
After warm-up, confirmed hits publish at detection time; unseen misses still
wait for a credible following launch or the conservative cadence deadline.

When OBS is available at `192.168.1.197:9000` and the counter should be
available to a Quest on the same local network, use the shortcut below, then
open `http://MAC_LAN_IP:8000` in the Quest browser:

```sh
npm run counter:quest
```

This shortcut uses the current 1920x1080 profile-side calibration at
`artifacts/live-2026-07-24-side-calibration.json`, fitted to the detector input
captured from the live Quest/OBS framing. The live OBS output must keep that
resolution, crop, and camera placement. Configure the SRT sender/OBS
output for 30 FPS. The labeled 153-second evaluation processes more than four
times faster than real time at 30 FPS on the development Mac, leaving enough
headroom to avoid the growing SRT backlog and H.264 corruption seen at 60 FPS.
Reducing FPS only after frames reach the receiver does not reduce SRT bandwidth
or decoder load, so make this change at the sender.
The analyzer prints a warning if its processed-video clock falls at least two
seconds behind wall-clock time, and reports when that backlog recovers.
The browser also keeps semantic health warnings visible when detector events
are not reaching SSE or when eight finalized attempts produce no confirmed
table contact. These warnings clear automatically after publication or contact
detection recovers; they do not reset the current score.
Every normal Quest run also resets and continuously writes
`artifacts/live-counter-events.jsonl`, plus a bounded 30-second MJPEG detector
input at `artifacts/live-counter-clean.mkv` beginning with the first detected
launch. This makes a failed live session directly replayable without imposing
an unbounded recording workload. In addition to `attempt_upsert` records,
the file receives one `pipeline_heartbeat` per wall-clock second with the
processed frame and video time, effective processing FPS, estimated lag,
candidate and track counts, detector event count, and pending/finalized
attempt counts. A final `pipeline_end` record distinguishes an orderly stop
from a process that disappeared or stopped advancing between heartbeats.
The Quest shortcut uses a 400 ms SRT latency buffer. An individual frame read
times out after three seconds; the analyzer then reconnects without resetting
its frame or attempt sequence. The live log records `source_stalled`,
`source_reconnecting`, and `source_reconnected` transitions, followed by an
`analyzer_exit` record when the analyzer process ends.

For a short detector-diagnostic session, use `npm run counter:quest:debug`.
It writes two complementary artifacts:

- `artifacts/live-counter-clean.mkv` is lower-overhead MJPEG detector input
  captured before overlays. Recording begins immediately and is capped at 120
  seconds by the Quest debug shortcut, so it still captures sessions when
  launch detection is broken. Direct analyzer captures remain lossless FFV1 by
  default when exact pixel replay is required.
- `artifacts/live-counter-events.jsonl` preserves every live publication with
  its shot frame, publication frame, and publication delay.

The event stream carries `attempt_upsert` records with a `pending -> finalized`
lifecycle. Live SRT and prerecorded evaluation use the cadence-aware ledger to
reject launcher fragments and recover fully occluded launches. Overlapping
tracks for one physical contact are deduplicated before cadence inference, and
cadence is locked once established so attempt IDs cannot shift. A later
confirmed hit can revise a previous launch miss, and the browser accepts only
explicitly higher revisions. Finalized records distinguish the physical
evidence `frame_number` from `decision_frame_number`, the frame where enough
evidence existed to commit the result. `publication_delay_seconds` measures
contact/evidence-to-publication latency,
`decision_publication_delay_seconds` measures decision-to-publication latency,
and `feedback_delay_seconds` uses contact latency for hits and decision latency
for misses. This keeps miss delivery latency meaningful even when cadence must
wait for a later launch to prove that no return occurred.

Compare a captured browser SSE ledger with timestamped human labels using the
same reconciliation rules as the counter page:

```sh
python3 scripts/compare_sse_labels.py LABELS.json SSE.jsonl \
  --canonical CANONICAL.jsonl \
  --json-output human-vs-sse.json \
  --markdown-output human-vs-sse.md \
  --alignment-output human-vs-sse-alignment.jsonl
```

Stop it with Ctrl-C after the labeled sequence. Change the bound with
`--clean-recording-seconds`, or force recording from stream startup with
`--clean-recording-start immediate`. Clean recording remains optional for
normal use. Render verbose overlays afterward, without burdening live capture:

```sh
python3 scripts/analyze_video.py artifacts/live-counter-clean.mkv \
  --annotated artifacts/live-counter-debug.mp4 \
  --output artifacts/live-counter-replay.jsonl
```

Run the checked-in clean sample through the detector and browser-order
regression with:

```sh
npm run counter:replay
npm run test:e2e:counter
npm run test:e2e:counter:side-view
```

Failures print a compact expected/actual result with shot timestamp and live
publication delay. The Playwright checks exercise the served page and SSE
stream with both deterministic messages and checked-in
`side-view-regression.mkv`, and verify that the visible counter recovers after
a reset. The 60-second fixture is a lossless trim of the user-confirmed
canonical profile-side evaluation and covers 43 human-labeled attempts: 24
hits and 19 misses. It enforces feedback latency after cadence warm-up while
still checking all warm-up outcomes. The shorter structured
unit fixture covers 5 hits, 2
no-swings, 3 hits, 1 out, and 3 hits for fast normalizer iteration.

Press Ctrl-C in the terminal to stop both the server and analyzer. Use
`--calibration PATH.json`, `--output PATH.jsonl`, `--host`, or `--port` after
the video argument when needed.

For the quickest lifelike feedback loop with a prerecorded Quest/OBS capture,
pace the file against its original media clock and wait to start until the
browser is connected:

```sh
npm run counter:video -- VIDEO.mkv \
  --calibration artifacts/live-2026-07-24-side-calibration.json
```

To inspect a local recording without running the detector, requiring a
calibration, writing analyzer output, or connecting to OBS, use:

```sh
npm run counter:preview -- VIDEO.mkv
```

This mode labels the debug page as **Video preview · detector disabled** so it
cannot be mistaken for analyzed output.

Open <http://127.0.0.1:8000/?debug=true> before playback begins to show the
annotated detector replay and its restart control. Without `?debug=true`, the
browser shows only the normal counter and does not connect to the video
preview. The detector still processes every source frame; the debug browser
preview is throttled to 12 FPS to keep JPEG transport from affecting detector
timing. Change that with `--preview-fps 20`, or disable it with `--no-preview`.
Use the browser's **Restart video** button to stop the current analysis, rewind
the file, clear the session streak, and resume without restarting the server.
This reproduces frame cadence and live publication timing, but not SRT packet loss, decoder
reconnects, encoder latency, or network jitter. Use the real SRT command when
those transport effects are specifically under test.

`scripts/analyze_video.py` reads either a fixed spectator-view video file or a
live OBS SRT stream and writes one finalized `hit`, `out`, or `miss` JSONL
record for each inferred ball-machine launch. A visually confirmed opponent-
side contact is emitted as `hit` from the live return track once two
post-contact frames establish the bounce; it does not wait for the track to
disappear, for cadence, or for the next launch. A terminal shadow contact with
no visible departure still waits for the track to end. Cadence-based `out`
and `miss` slots have a six-contact live warm-up and remain held until a later
launch settles them, preventing a temporarily occluded return from becoming a
premature miss. Merely receiving more video does not infer trailing misses
after the machine stops. Every live record is flushed immediately. Use
`--live-stdout` to see it directly without the roughly one-second update delay
that `tail -f` can add on macOS. At shutdown the file is rewritten using the
analyzer's existing canonical batch normalization. It never loads the full
video into memory. The supported profile-side camera placement requires its
matching calibration: table corners, coordinate orientation, net, and the
launcher region are deliberately not inferred from old-angle fixtures or
reused across setups.

The strict profile view captured in
`artifacts/evaluation-2026-07-23-184555.mkv` uses the reviewed
`artifacts/evaluation-2026-07-23-side-calibration.json`. In this mode the
counter deliberately reports no table coordinate because table width is not
visible. A hit is the directly observable downward-to-upward ball turn on the
calibrated opponent/right side of the net:

```sh
python3 scripts/analyze_video.py VIDEO_OR_SRT_URL \
  --calibration artifacts/evaluation-2026-07-23-side-calibration.json
```

Automatic calibration also derives four camera-relative detector regions: a
launcher-side start zone, a player-side return zone, a vertically bounded
flight corridor, and the visible table-contact polygon. The start-zone roles
follow the calibrated player/opponent orientation even when a camera view is
mirrored. The corridor stays horizontally wide for edge-of-table and
wide-angle paths while excluding room motion well above or below the table.

Create an isolated Python environment and install the pinned dependencies. For
the supported profile-side view, pass its reviewed calibration explicitly:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/analyze_video.py side-view-regression.mkv \
  --calibration artifacts/live-2026-07-24-side-calibration.json
```

For live input, use an OpenCV build whose FFmpeg backend supports SRT. When OBS
is the SRT server/listener, connect the analyzer as the caller:

```sh
python3 scripts/analyze_video.py \
  'srt://OBS_IP:9000?mode=caller&latency=120000' \
  --live-stdout
```

The inverse arrangement also works: use `mode=listener` in the analyzer URL
when OBS is configured as the caller.

The analyzer waits for OBS, uses the first received frame for automatic
calibration, and continues until OBS disconnects. Pressing Ctrl-C ends the
session cleanly and finalizes the events already written. `--end-seconds` can
bound a live session; `--start-seconds` is available only for seekable files.
SRT is opened explicitly with OpenCV's FFmpeg backend.

Annotated video is disabled by default, which avoids an indefinitely growing
recording during live analysis. Pass `--annotated` to write
`video_bounces_annotated.mp4`, or `--annotated PATH.mp4` to choose its path. It
shows the detected geometry and is the visual check: the yellow table polygon,
magenta physical net-base line, and projected log-space grid must align with
the table. An explicitly reviewed calibration can still be exported with
`scripts/auto_calibrate.py` and supplied with `--calibration` for diagnostic
work, but automatic analysis does not cache one. The geometry
uses a player-relative convention: `posz > 0` is the far/opponent side and
`posy` is the 0.7786m table surface. The physical image
direction of each axis is per-camera calibration data, never a global rule.
An explicitly supplied JSON is rejected if its `image_size` does not match the
input video. When requested, the generated annotated video shows
the table, net, tracked path, markers, coordinates, and confidence.
Its diagnostic legend distinguishes faint raw candidates, orange rejected
candidates and tracks (with rejection reasons), cyan unconfirmed tracks, blue
launcher tracks, green return tracks, and red confirmed bounces. Completed
track annotations remain visible briefly for review; these diagnostics observe
the detector and do not change its classification decisions.

When the in-room TV is visible, the analyzer also reads its speed, spin, and
blue spin arrow. The two alternating TV updates are kept separate: `machine`
is the ball-machine delivery captured for the launch, while `hit` is the later
reading produced by the player's return. Both are nested on that attempt's
landing record, for example:

```json
{
  "outcome": "hit",
  "posx": 0.03,
  "posy": 0.7786,
  "posz": 1.08,
  "hit": {
    "speed_mps": 15.0,
    "spin_revolutions_per_second": 80,
    "spin_direction": {"x": -0.7, "y": 0.7, "angle_degrees": 135, "label": "up-left"},
    "video_time_seconds": 12.1
  },
  "machine": {
    "speed_mps": 10.5,
    "spin_revolutions_per_second": 51,
    "spin_direction": {"x": 0.0, "y": 1.0, "angle_degrees": 90, "label": "up"},
    "video_time_seconds": 11.6
  }
}
```

Spin-arrow vectors use TV screen coordinates with positive `x` to the right
and positive `y` upward. Telemetry is omitted rather than guessed when the TV
or its tiny digits cannot be read conservatively.

For a camera that needs different detection sensitivity, the calibration JSON
may include a `detector_settings` object. It can override named thresholds such
as `motion_threshold`, `track_match_distance`, `min_shadow_contact_score`, or
the candidate appearance limits; omitted settings retain the tested defaults.
Candidate filtering checks brightness, saturation, shape compactness, aspect
ratio, and a size range that grows toward the near end of the calibrated
flight corridor. Single-pixel shimmer is rejected while compact two-pixel
distant balls remain eligible for temporal tracking. Tracks require three
consistent observations before classification, and candidate links are gated
by frame-aware position prediction, speed, acceleration, and direction change.
Launcher tracks must begin in the calibrated machine-side region and make
sustained horizontal progress toward the calibrated player side. This follows
mirrored camera orientation without constraining vertical arcs, delivery speed,
or spin shape, while static shimmer and back-and-forth scene motion cannot open
an attempt. Return tracks likewise must begin near the calibrated player side
and make camera-relative progress toward the opponent. A stale-object prefix
may predate the physical ball, so association requires several observations
after the active credible launch starts; the next credible launch closes that
attempt. This keeps partially occluded returns while rejecting tracks that
finished before the active launch.

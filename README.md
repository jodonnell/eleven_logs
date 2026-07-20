
# Download your logs off your device
For the quest plug you into your computer and click the popup that says allow this pc to connect.
Then navigate in explorer to `\This PC\Quest 3\Internal shared storage\Android\data\quest.eleven.forfunlabs\logs`

These logs get deleted after some period of time so if you want to keep track of all your data you will need to copy them to a folder on your computer.  Click the upload button.  The logs aren't actually uploaded, they are processed locally in your browser.


# run console version
npm run console

# run build
npm run build

# run build dev server
npm run build:watch

# run tests
npm run test

# prettier
npm run prettier

# lint
npm run lint

## Ball-machine video analysis

`scripts/analyze_video.py` reads either a fixed spectator-view video file or a
live OBS SRT stream and writes one finalized `hit`, `out`, or `miss` JSONL
record for each inferred ball-machine launch. A visually confirmed opponent-
side contact is emitted as `hit` as soon as its completed track establishes the
bounce; it does not wait for cadence or the next launch. Cadence-based `out`
and `miss` slots have a three-contact warm-up and remain held until a later
launch settles them, preventing a temporarily occluded return from becoming a
premature miss. Merely receiving more video does not infer trailing misses
after the machine stops. Every live record is flushed immediately, so
`tail -f video_bounces.jsonl` can follow the session. At shutdown the file is
rewritten using the analyzer's existing canonical batch normalization. It
never loads the full video into memory. Every camera
placement requires its own calibration: table corners, coordinate orientation,
net, and the launcher region are deliberately not inferred from `sample.mp4` or
reused across setups.

Automatic calibration also derives four camera-relative detector regions: a
launcher-side start zone, a player-side return zone, a vertically bounded
flight corridor, and the visible table-contact polygon. The start-zone roles
follow the calibrated player/opponent orientation even when a camera view is
mirrored. The corridor stays horizontally wide for edge-of-table and
wide-angle paths while excluding room motion well above or below the table.

Install the local Python dependencies once. Each run detects the green table,
white `x=0` centre stripe, and the table-side (bottom) edge of the net from the
first usable frame. That geometry stays in memory for the duration of the
analysis; the normal workflow does not create or consume a calibration file:

```sh
python3 -m pip install --user opencv-python-headless numpy
python3 scripts/analyze_video.py sample.mp4
```

For live input, use an OpenCV build whose FFmpeg backend supports SRT. When OBS
is the SRT server/listener, connect the analyzer as the caller:

```sh
python3 scripts/analyze_video.py \
  'srt://OBS_IP:9000?mode=caller&latency=120000'
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
maps to the same
player-relative convention used by `src/parser.js`: `posz > 0` is the
far/opponent side and `posy` is the 0.7786m table surface. The physical image
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
distant balls remain eligible for temporal tracking.

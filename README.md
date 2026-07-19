
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

## Offline ball-machine video analysis

`scripts/analyze_video.py` streams a fixed spectator-view MP4 and writes one
JSONL record for each conservative table-bounce candidate. It never loads the
video into memory. Every camera placement requires its own calibration: table
corners, coordinate orientation, net, and the launcher region are deliberately
not inferred from `sample.mp4` or reused across setups.

Install the local Python dependencies once. On a new camera position, the
first run detects the green table, white `x=0` centre stripe, and the
table-side (bottom) edge of the net from the first usable frame. It caches
that per-camera geometry beside the output; later runs reuse it and do not
recalibrate while processing frames:

```sh
python3 -m pip install --user opencv-python-headless numpy
python3 scripts/analyze_video.py sample.mp4 --calibration-cache my_camera.table-calibration.json
```

The cache's adjacent PNG is a required visual check: it shows the yellow table
polygon, white `x=0` stripe, and magenta physical net-base line. If the camera
is unusually obstructed or the automatic diagnostic is wrong, provide a
reviewed per-camera calibration with `--calibration`. The calibration maps to
the same
player-relative convention used by `src/parser.js`: `posz > 0` is the
far/opponent side and `posy` is the 0.7786m table surface. The physical image
direction of each axis is per-camera calibration data, never a global rule.
The JSON is rejected if its `image_size` does not match the input video. The
generated `video_bounces_annotated.mp4` shows
the table, net, tracked path, markers, coordinates, and confidence.

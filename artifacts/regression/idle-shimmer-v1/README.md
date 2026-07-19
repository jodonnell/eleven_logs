# Idle/shimmer detector regression

This fixture preserves the failing detector run from 2026-07-19. Treat the
video and JSONL files as immutable: create a new versioned fixture instead of
overwriting them when the detector changes.

## Files

- `annotated.mp4` is the 104.1-second annotated detector output.
- `detector-output.jsonl` is the corresponding six-event detector output.
- `timestamps.json` records the manually classified points and conservative
  idle intervals used when comparing later detector versions.
- `manifest.json` records byte sizes, media properties, and SHA-256 checksums.

The recording contains one known real table hit at `01:23.933`. The other five
reported events are false detections. In particular, output before the real
attempt and after `01:31.000` must not be interpreted as a stream of attempts.

To verify that the saved artifacts have not changed:

```sh
shasum -a 256 annotated.mp4 detector-output.jsonl
```

Compare the result with `manifest.json`.

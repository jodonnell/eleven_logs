# Ball Detection Improvement Plan

Update each task's status from `TODO` to `DONE` when its implementation and
verification are complete.

## 1. Preserve the failing example — DONE

- Save the current annotated video and JSONL output as regression artifacts.
- Record timestamps for known real hits, false detections, and idle periods.
- Keep the fixtures stable so later detector changes can be compared against
  the same footage.

## 2. Add detector diagnostics — TODO

- Give raw candidates, rejected candidates, unconfirmed tracks, valid launcher
  tracks, valid return tracks, and confirmed bounces distinct annotations.
- Include the rejection reason where practical.
- Keep diagnostic rendering separate from classification behavior.

## 3. Tighten the tracking region — TODO

- Define calibrated regions for launcher-side track starts, player-side return
  starts, the valid flight corridor, and table-contact locations.
- Reject candidates that begin on the windows, TV, floor, or walls.
- Verify that edge-of-table and wide-angle ball paths remain inside the allowed
  regions.

## 4. Validate ball appearance — TODO

- Filter candidates by pixel area, aspect ratio, brightness, saturation, and
  compactness.
- Scale the acceptable apparent ball size according to position and
  perspective.
- Retain small distant balls without accepting single-pixel codec shimmer.

## 5. Require plausible motion — TODO

- Require several consistent observations before promoting a candidate to a
  valid track.
- Check direction, speed, acceleration, and distance from the predicted next
  position.
- Reject implausible jumps between unrelated bright objects.

## 6. Validate launcher tracks — TODO

- Require a launch to begin near the machine and travel toward the player.
- Prevent static shimmer and unrelated scene motion from opening an attempt.
- Verify launch detection across the machine's supported speeds and spins.

## 7. Validate return tracks — TODO

- Require a return to begin near the player side after a credible launch and
  travel toward the opponent side.
- Associate each return with its active launch instead of accepting any
  right-moving track.
- Handle short or partially occluded returns conservatively.

## 8. Improve bounce confirmation — TODO

- Combine trajectory direction change, vertical-speed change, shadow
  convergence, table position, and return timing.
- Avoid confirming a bounce from a single noisy frame or one signal alone.
- Preserve detection of terminal contacts that disappear behind the launcher.

## 9. Fix miss generation — TODO

- Infer a `miss` only after observing a credible later launch that settles the
  previous attempt.
- Suspend cadence inference when credible launcher tracks disappear.
- Never generate trailing misses merely because video continues or the machine
  is idle.
- Ensure every attempt produces either confirmed hit data or one clear
  non-hit outcome: `out`, `net`, or `miss`.

## 10. Reduce annotation clutter — TODO

- Draw raw candidates faintly and briefly.
- Reserve persistent paths for validated ball tracks.
- Use distinct colors and labels for launches, returns, rejected tracks, and
  confirmed bounces.

## 11. Add automated regression tests — TODO

- Cover an in-table hit, an off-table return, a net return, a genuine miss,
  idle footage, window/TV shimmer, a partially occluded ball, and machine
  startup/shutdown.
- Test that live and finalized JSONL output agree on settled attempts.
- Run the detector against the preserved failing recording.

## 12. Meet the completion criteria — TODO

- Report the known hit exactly once and without waiting for cadence.
- Produce no attempts from window, TV, wall, floor, or launcher shimmer.
- Produce no results during idle footage.
- Produce at most one result per real launch.
- Make annotated tracks visibly follow the ball.
- Prevent any unbounded stream of inferred misses.

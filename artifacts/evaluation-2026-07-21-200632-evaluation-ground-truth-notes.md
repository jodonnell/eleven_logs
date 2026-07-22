# Evaluation ground-truth notes

- Video: `evaluation-2026-07-21-200632-evaluation.mkv`
- Raw export: `evaluation-2026-07-21-200632-evaluation-ground-truth-export.json`
- Raw label count: 234 (`172 hit`, `62 miss`)
- The raw export is preserved byte-for-byte from `~/Downloads/evaluation-labels.json`.

## User-reviewed edge cases

- `01:07.291` — The ball bounces on the player's side and then the opponent's
  side. The supplied ground-truth outcome is `miss`.
- `01:21.708` — The ball hits the net and then lands on the opponent's table.
  The supplied ground-truth outcome is `hit`. At least one ball rolled up the
  net and flew high into the air, delaying its table contact until the following
  ball was also close to landing.
- Around `01:17` — The recording contains a visual glitch resembling a ball
  floating near the player. This is not a separate launch or attempt.

## Timestamp interpretation

Each timestamp records when the user observed and entered the outcome. It is not
the corresponding machine-launch timestamp. Delayed net contacts can therefore
produce two legitimate outcome labels very close together, including:

- `00:50.772 miss` → `00:51.162 miss` (0.390 seconds)
- `00:55.759 miss` → `00:56.259 hit` (0.500 seconds)
- `01:21.708 hit` → `01:21.936 hit` (0.228 seconds)
- `01:54.220 miss` → `01:54.801 hit` (0.581 seconds)

These gaps are not evidence of duplicate labels. Missing and duplicate labels
must be checked by matching the ordered ground-truth ledger against independently
located machine launches in the video. The raw export remains unmodified.

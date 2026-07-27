# Ball Classification Architecture Plan

This plan replaces one-off threshold tuning with three independently measured
architectural stages. Complete and score each stage before beginning the next.
Change a task from `[ ]` to `[x]` only after its implementation, tests, and
234-shot evaluation are complete.

## Baseline and guardrails

- Starting commit: `663af19` (`Reject pre-contact tracker handoffs`)
- Evaluation set: 234 labeled attempts
- Starting accuracy: 91.5% (215 correct)
- Starting errors: 17 wrong outcomes, 2 missing launches, 1 extra launch
- Starting hit metrics: 92.8% precision, 97.1% recall
- Current stable video guard: `side-view-regression.mkv`, a lossless 60-second
  trim of the confirmed canonical profile-side evaluation
- Excluded guard: the archived `artifacts/live/archive-2026-07-24/clean.mkv`
  does not match
  its old 11-shot label fixture

Rules for every stage:

- [x] Keep the evaluation video, labels, and existing reports unchanged.
- [x] Add unit coverage for the new behavior before running the full video.
- [x] Produce a named prediction file, JSON report, Markdown report, and
  comparison manifest.
- [x] Compare every changed attempt, not only aggregate accuracy.
- [x] Pass the stable video guard.
- [x] Keep the stage only if accuracy improves without a stable-video
  regression; otherwise record the result and revert the behavior.
- [x] Do not adjust numerical thresholds unless recorded evidence establishes
  a separable mechanism and the adjustment receives its own scorecard.

## Stage 1: Ordered contact history

Goal: represent what happened to a returned ball in time order instead of
reducing its track to one strongest apparent bounce.

### Data model

- [x] Add a `ContactCandidate` record containing frame, pixel/log position,
  table side, signal type, strength/confidence, source track key, and bounded
  approach/departure evidence.
- [x] Add ordered contact candidates to `Attempt` without changing current
  user-facing events.
- [x] Deduplicate the same physical contact observed by active-track and
  completed-track processing.
- [x] Keep rejected candidates available to diagnostics without allowing them
  to affect classification.

### Detection and classification

- [x] Refactor bounce detection so it can return all qualified contacts in
  chronological order while preserving the existing single-contact wrapper.
- [x] Distinguish a physical contact from a small in-flight trajectory turn
  using the already-recorded approach, departure, shadow, and table-position
  evidence—not contact order alone.
- [x] Encode the outcome rule: a confirmed player-side contact followed by an
  opponent-side contact is a miss.
- [x] Encode the outcome rule: a net interaction followed by an opponent-side
  table contact is a hit.
- [x] Preserve off-table, net-only, terminal-shadow, and fully occluded misses.
- [x] Delay final classification when contact history is still ambiguous.

### Tests and evaluation

- [x] Add unit tests for near-then-far, net-then-far, clean far-only, off-table,
  tracker handoff, shadow plateau, and weak pre-contact jitter.
- [x] Add diagnostics that render and serialize ordered contact candidates.
- [x] Review the ten mismatches categorized as own-side-first bounce against
  the new history.
- [x] Run the full evaluation and record the Stage 1 before/after scorecard.
- [x] Run the three stable video guards and decide keep/revert.

## Stage 2: Overlapping attempts and durable ball ownership

Goal: keep a delayed ball associated with its launch after the next machine
launch appears, while preventing one track from belonging to multiple balls.

### Attempt lifecycle

- [x] Replace `active_attempt: Optional[Attempt]` with a bounded ordered set of
  active attempts.
- [x] Define explicit attempt states such as `launched`, `return_seen`,
  `contact_pending`, `settled`, and `expired`.
- [x] Keep a previous attempt alive across a later launch when it has credible
  unresolved return, net, or airborne evidence.
- [x] Bound active-attempt count and lifetime from observed machine cadence so
  streaming memory remains finite.

### Track ownership

- [x] Give each accepted launch, return fragment, reconnection, and contact a
  stable track/ball identity.
- [x] Score track-to-attempt association using temporal order, direction,
  position continuity, and prior ownership.
- [x] Make ownership exclusive unless a tracker fragment is explicitly split.
- [x] Preserve return reconnection across short avatar/net occlusions.
- [x] Prevent the next launch, a rolling old ball, or a static marking from
  taking over an unresolved return.
- [x] Define deterministic tie-breaking and emit association diagnostics.

### Settlement and live behavior

- [x] Settle attempts in launch order even when their visible contacts arrive
  out of order.
- [x] Preserve immediate live publication for unambiguous hits.
- [x] Prevent a later attempt from permanently publishing over an earlier
  delayed net ball.
- [x] Verify that batch output and the live ledger converge to the same ordered
  outcomes.

### Tests and evaluation

- [x] Add unit scenarios with two overlapping airborne balls, delayed net-then-
  table contact, a launch during an unresolved return, and competing fragments.
- [x] Review errors #122/#123 and other delayed/overlapping clips using the new
  ownership diagnostics.
- [x] Run the full evaluation and record the Stage 2 before/after scorecard.
- [x] Run the three stable video guards and decide keep/revert.

## Stage 3: Evidence-aware cadence reconciliation

Goal: use cadence only to reconcile genuinely unseen attempts, not to create or
misalign attempts despite contradictory launch evidence.

Candidate result: rejected and reverted. The visual-anchor-first candidate
scored 91.1% (214 correct) versus Stage 2's 92.7% (217 correct) and reduced the
historical three-view guard from 48 attempts to 47. See
`docs/evaluations/2026-07-21-detector-development/stage3-evidence-aware-cadence-manifest.json`
for the scorecard and
reconciliation review. The unchecked behavior and test items below were not
retained.

### Reconciliation model

- [ ] Separate visually detected launch anchors from cadence-only inferred
  anchors.
- [ ] Associate finalized attempt histories with the nearest compatible launch
  before filling any cadence gap.
- [ ] Require credible surrounding launch evidence before inserting a missing
  attempt.
- [ ] Suppress cadence-only slots when the launcher stream disappears, the
  machine pauses, or visual evidence contradicts a launch.
- [ ] Ensure the 1:17 floating-ball glitch cannot become a launch.
- [ ] Preserve a truly unseen miss when surrounding launches establish that a
  machine cycle occurred.
- [ ] Keep startup and shutdown boundaries from producing leading or trailing
  attempts.

### Batch/live convergence

- [ ] Use the same reconciliation decisions for canonical batch output and the
  live attempt ledger.
- [ ] Record why every attempt anchor is `visual` or `cadence_inferred`.
- [ ] Make reconciliation deterministic when delayed contacts arrive after a
  later launch.

### Tests and evaluation

- [ ] Add tests for one missed visual launch, one false visual launch, a machine
  pause, startup/shutdown, the floating-ball glitch, and delayed contacts.
- [x] Review the two missing launches and one extra launch from the baseline.
- [x] Run the full evaluation and record the Stage 3 before/after scorecard.
- [x] Run the three stable video guards and decide keep/revert.

## Final completion criteria

- [x] Each retained stage has an independent commit and comparison manifest.
- [x] The final detector improves on 91.5% across the same 234 labels.
- [x] Hit precision and recall changes are explicitly reported.
- [x] No stable sample video regresses in launch count or ordered hit/miss
  outcomes.
- [x] Annotated output makes launch ownership and ordered contacts reviewable.
- [x] Live and batch output agree after all attempts settle.
- [x] No large videos, reels, browser proxies, or transient diagnostics are
  committed.

## Evaluation commands

```sh
python3 scripts/analyze_video.py \
  artifacts/runs/archive/2026-07-21/evaluation-2026-07-21-200632-evaluation.mkv \
  --output artifacts/runs/STAGE-CANDIDATE/detector.jsonl \
  --no-annotated

python3 scripts/evaluate_detector.py \
  docs/evaluations/2026-07-21-detector-development/evaluation-2026-07-21-200632-evaluation-ground-truth-export.json \
  artifacts/runs/STAGE-CANDIDATE/detector.jsonl \
  --json-output artifacts/runs/STAGE-CANDIDATE/report.json \
  --markdown-output artifacts/runs/STAGE-CANDIDATE/report.md
```

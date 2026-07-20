import {
  currentHitStreak,
  reduceCounterState,
  reconcileShotMessage,
} from "../live-counter/counter.js"

describe("live hit counter", () => {
  it("calculates the streak in shot order rather than arrival order", () => {
    const shots = [
      { frame_number: 200, outcome: "hit" },
      { frame_number: 300, outcome: "hit" },
      { frame_number: 100, outcome: "miss" },
    ]

    expect(currentHitStreak(shots)).toBe(2)
  })

  it("retroactively resets and retains hits made after a late miss", () => {
    const shots = [
      { frame_number: 100, outcome: "hit" },
      { frame_number: 300, outcome: "hit" },
      { frame_number: 200, outcome: "miss" },
    ]

    expect(currentHitStreak(shots)).toBe(1)
  })

  it("resets when the newest chronological shot is a miss", () => {
    const shots = [
      { frame_number: 100, outcome: "hit" },
      { frame_number: 300, outcome: "miss" },
      { frame_number: 200, outcome: "hit" },
    ]

    expect(currentHitStreak(shots)).toBe(0)
  })

  it("treats an out as a missed streak shot", () => {
    const shots = [
      { frame_number: 100, outcome: "hit" },
      { frame_number: 200, outcome: "out" },
      { frame_number: 300, outcome: "hit" },
    ]

    expect(currentHitStreak(shots)).toBe(1)
  })

  it("uses a reconciled attempt frame while preserving the evidence frame", () => {
    const shots = [
      { frame_number: 100, attempt_frame_number: 100, outcome: "hit" },
      { frame_number: 500, attempt_frame_number: 200, outcome: "out" },
      { frame_number: 300, attempt_frame_number: 300, outcome: "hit" },
    ]

    expect(currentHitStreak(shots)).toBe(1)
  })

  it("replaces provisional shots with a cadence-reconciled snapshot", () => {
    const provisional = [
      { frame_number: 100, outcome: "hit" },
      { frame_number: 150, outcome: "miss" },
    ]
    const snapshot = {
      type: "snapshot",
      shots: [
        { frame_number: 100, attempt_frame_number: 90, outcome: "hit" },
        { frame_number: 200, attempt_frame_number: 150, outcome: "hit" },
      ],
    }

    const reconciled = reconcileShotMessage(provisional, snapshot)

    expect(reconciled).toEqual(snapshot.shots)
    expect(currentHitStreak(reconciled)).toBe(2)
  })

  it("retains display-only reset signals as a streak boundary", () => {
    const shots = [{ frame_number: 100, outcome: "hit" }]

    const reconciled = reconcileShotMessage(shots, {
      type: "reset",
      after_hit_frame_number: 100,
    })

    expect(reconciled).toEqual([
      ...shots,
      expect.objectContaining({ outcome: "miss", display_only: true }),
    ])
    expect(currentHitStreak(reconciled)).toBe(0)
  })

  it("counts hits after a timed reset instead of remaining at zero", () => {
    const messages = [
      { outcome: "hit", frame_number: 100 },
      { outcome: "hit", frame_number: 160 },
      { outcome: "hit", frame_number: 220 },
      { type: "reset", after_hit_frame_number: 220 },
      {
        type: "snapshot",
        shots: [
          { outcome: "hit", frame_number: 100 },
          { outcome: "hit", frame_number: 160 },
          { outcome: "hit", frame_number: 220 },
        ],
      },
      { outcome: "hit", frame_number: 280 },
      { outcome: "hit", frame_number: 340 },
      { outcome: "hit", frame_number: 400 },
    ]
    let state = { shots: [], streak: 0 }

    const visible = messages.map((message) => {
      state = reduceCounterState(state, message)
      return state.streak
    })

    expect(visible).toEqual([1, 2, 3, 0, 0, 1, 2, 3])
  })
})

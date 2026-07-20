import { currentHitStreak } from "../live-counter/counter.js"

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
})

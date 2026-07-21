import {
  HIGH_SCORE_STORAGE_KEY,
  currentHitStreak,
  loadHighScore,
  reconcileAttemptUpsert,
  reduceCounterState,
  saveHighScore,
} from "../live-counter/counter.js"

const attempt = (sequence, state, outcome) => ({
  type: "attempt_upsert",
  attempt_id: `attempt-${String(sequence).padStart(4, "0")}`,
  sequence,
  anchor_frame_number: sequence * 100,
  state,
  ...(outcome ? { outcome } : {}),
})

describe("live hit counter attempt ledger", () => {
  it("calculates the streak only from ordered finalized attempts", () => {
    const attempts = [
      attempt(3, "finalized", "hit"),
      attempt(1, "finalized", "out"),
      attempt(4, "pending"),
      attempt(2, "finalized", "hit"),
    ]

    expect(currentHitStreak(attempts)).toBe(2)
  })

  it("upserts a pending attempt by stable ID", () => {
    const pending = attempt(1, "pending")
    const finalized = attempt(1, "finalized", "hit")

    expect(reconcileAttemptUpsert([pending], finalized)).toEqual([finalized])
  })

  it("does not replace a finalized outcome with contradictory evidence", () => {
    const hit = attempt(1, "finalized", "hit")
    const contradiction = attempt(1, "finalized", "out")

    expect(reconcileAttemptUpsert([hit], contradiction)).toEqual([hit])
  })

  it("derives every visible transition from the finalized ledger", () => {
    const messages = [
      attempt(1, "pending"),
      attempt(1, "finalized", "hit"),
      attempt(2, "pending"),
      attempt(2, "finalized", "hit"),
      attempt(3, "pending"),
      attempt(3, "finalized", "miss"),
      attempt(4, "pending"),
      attempt(4, "finalized", "hit"),
    ]
    let state = { attempts: [], streak: 0 }

    const visible = messages.map((message) => {
      state = reduceCounterState(state, message)
      return state.streak
    })

    expect(visible).toEqual([0, 1, 1, 2, 2, 0, 0, 1])
    expect(state.attempts).toHaveLength(4)
  })
})

describe("browser high score", () => {
  const memoryStorage = (initialValue = null) => {
    let value = initialValue
    return {
      getItem: jest.fn(() => value),
      setItem: jest.fn((_key, nextValue) => {
        value = nextValue
      }),
    }
  }

  it("loads a saved non-negative whole number", () => {
    const storage = memoryStorage("12")

    expect(loadHighScore(storage)).toBe(12)
    expect(storage.getItem).toHaveBeenCalledWith(HIGH_SCORE_STORAGE_KEY)
  })

  it.each([null, "", "not-a-number", "-1", "2.5"])(
    "treats %p as no saved high score",
    (savedValue) => {
      expect(loadHighScore(memoryStorage(savedValue))).toBe(0)
    },
  )

  it("saves the score under the stable browser key", () => {
    const storage = memoryStorage()

    saveHighScore(storage, 7)

    expect(storage.setItem).toHaveBeenCalledWith(HIGH_SCORE_STORAGE_KEY, "7")
  })

  it("keeps working when browser storage is unavailable", () => {
    const storage = {
      getItem: () => {
        throw new Error("storage disabled")
      },
      setItem: () => {
        throw new Error("storage disabled")
      },
    }

    expect(loadHighScore(storage)).toBe(0)
    expect(() => saveHighScore(storage, 3)).not.toThrow()
  })
})

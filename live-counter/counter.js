const orderedFinalized = (attempts) =>
  [...attempts]
    .filter((attempt) => attempt.state === "finalized")
    .sort((left, right) => left.sequence - right.sequence)

export const currentHitStreak = (attempts) => {
  let streak = 0
  for (const attempt of orderedFinalized(attempts)) {
    streak = attempt.outcome === "hit" ? streak + 1 : 0
  }
  return streak
}

const average = (values) =>
  values.length === 0
    ? null
    : values.reduce((total, value) => total + value, 0) / values.length

export const sessionStats = (attempts) => {
  const finalized = orderedFinalized(attempts)
  const lastAttempt = finalized.at(-1)
  const successfulHits = finalized.filter(
    (attempt) => attempt.outcome === "hit",
  )
  const speeds = successfulHits
    .map((attempt) => attempt.hit?.speed_mps)
    .filter((value) => Number.isFinite(value))
  const spins = successfulHits
    .map((attempt) => attempt.hit?.spin_revolutions_per_second)
    .filter((value) => Number.isFinite(value))

  return {
    hits: successfulHits.length,
    total: finalized.length,
    hitPercentage:
      finalized.length === 0
        ? null
        : (successfulHits.length / finalized.length) * 100,
    averageSpeedMps: average(speeds),
    averageSpinRevolutionsPerSecond: average(spins),
    lastBall: lastAttempt
      ? {
          outcome: lastAttempt.outcome,
          speedMps: Number.isFinite(lastAttempt.hit?.speed_mps)
            ? lastAttempt.hit.speed_mps
            : null,
          spinRevolutionsPerSecond: Number.isFinite(
            lastAttempt.hit?.spin_revolutions_per_second,
          )
            ? lastAttempt.hit.spin_revolutions_per_second
            : null,
        }
      : null,
  }
}

export const reconcileAttemptUpsert = (attempts, message) => {
  if (message.type !== "attempt_upsert") return attempts
  const index = attempts.findIndex(
    (attempt) => attempt.attempt_id === message.attempt_id,
  )
  if (index === -1) return [...attempts, message]

  const existing = attempts[index]
  if (existing.state === "finalized") {
    // Ignore delivery retries and stale evidence. A higher revision is an
    // explicit correction backed by later confirmed contact evidence.
    const existingRevision = existing.revision ?? 0
    const incomingRevision = message.revision ?? 0
    if (incomingRevision <= existingRevision) return attempts
  }
  const updated = [...attempts]
  updated[index] = message
  return updated
}

export const reduceCounterState = (state, message) => {
  const attempts = reconcileAttemptUpsert(state.attempts, message)
  return {
    attempts,
    streak: currentHitStreak(attempts),
    stats: sessionStats(attempts),
  }
}

export const HIGH_SCORE_STORAGE_KEY = "eleven-practice.high-score"

export const loadHighScore = (storage) => {
  try {
    const score = Number(storage.getItem(HIGH_SCORE_STORAGE_KEY))
    return Number.isInteger(score) && score >= 0 ? score : 0
  } catch {
    return 0
  }
}

export const saveHighScore = (storage, score) => {
  try {
    storage.setItem(HIGH_SCORE_STORAGE_KEY, String(score))
  } catch {
    // The live count should keep working when browser storage is unavailable.
  }
}

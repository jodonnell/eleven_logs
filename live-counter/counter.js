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

export const reconcileAttemptUpsert = (attempts, message) => {
  if (message.type !== "attempt_upsert") return attempts
  const index = attempts.findIndex(
    (attempt) => attempt.attempt_id === message.attempt_id,
  )
  if (index === -1) return [...attempts, message]

  const existing = attempts[index]
  if (existing.state === "finalized") {
    // Delivery retries are harmless, but a finalized visible boundary is
    // immutable even if later detector evidence disagrees.
    return attempts
  }
  const updated = [...attempts]
  updated[index] = message
  return updated
}

export const reduceCounterState = (state, message) => {
  const attempts = reconcileAttemptUpsert(state.attempts, message)
  return { attempts, streak: currentHitStreak(attempts) }
}

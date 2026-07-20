export const currentHitStreak = (shots) => {
  const ordered = [...shots].sort(
    (left, right) =>
      (left.attempt_frame_number ?? left.frame_number) -
      (right.attempt_frame_number ?? right.frame_number),
  )
  let streak = 0
  for (const shot of ordered) {
    if (shot.outcome === "hit") {
      streak += 1
    } else if (shot.outcome === "miss" || shot.outcome === "out") {
      streak = 0
    }
  }
  return streak
}

export const reconcileShotMessage = (shots, message) =>
  message.type === "snapshot"
    ? [...message.shots]
    : message.type === "reset"
      ? shots
      : [...shots, message]

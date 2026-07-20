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

export const reconcileShotMessage = (shots, message) => {
  if (message.type === "snapshot") {
    return reconcileSnapshot(shots, message.shots)
  }
  if (message.type === "reset") {
    return [
      ...shots.filter((shot) => !shot.display_only),
      {
        outcome: "miss",
        frame_number: message.after_hit_frame_number + 1,
        attempt_frame_number: message.after_hit_frame_number + 1,
        display_only: true,
        after_hit_frame_number: message.after_hit_frame_number,
      },
    ]
  }
  return [...shots, message]
}

const logicalFrame = (shot) => shot.attempt_frame_number ?? shot.frame_number

const reconcileSnapshot = (current, canonical) => {
  const reset = [...current].reverse().find((shot) => shot.display_only)
  if (
    !reset ||
    canonical.some(
      (shot) =>
        (shot.outcome === "miss" || shot.outcome === "out") &&
        logicalFrame(shot) > reset.after_hit_frame_number,
    )
  ) {
    return [...canonical]
  }
  return [...canonical, reset]
}

export const reduceCounterState = (state, message) => {
  const shots = reconcileShotMessage(state.shots, message)
  return { shots, streak: currentHitStreak(shots) }
}

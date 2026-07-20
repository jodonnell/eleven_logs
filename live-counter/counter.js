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
      {
        outcome: "miss",
        frame_number: message.after_hit_frame_number + 1,
        attempt_frame_number: message.after_hit_frame_number + 1,
        display_only: true,
        anchored: false,
        after_hit_frame_number: message.after_hit_frame_number,
      },
    ]
  }
  if (message.outcome === "hit") {
    const reset = [...shots].reverse().find((shot) => shot.display_only)
    if (reset && !reset.anchored) {
      const hitFrame = logicalFrame(message)
      return [
        {
          ...reset,
          frame_number: hitFrame - 0.5,
          attempt_frame_number: hitFrame - 0.5,
          anchored: true,
        },
        message,
      ]
    }
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
  if (message.type === "snapshot") {
    return { ...state, shots }
  }
  if (message.type === "reset") {
    return { ...state, shots, streak: 0 }
  }
  if (message.outcome === "hit") {
    return {
      shots,
      streak: state.streak + 1,
      lastHitFrame: logicalFrame(message),
    }
  }
  if (message.outcome === "miss" || message.outcome === "out") {
    const isLate =
      state.lastHitFrame !== undefined && logicalFrame(message) < state.lastHitFrame
    return { ...state, shots, streak: isLate ? state.streak : 0 }
  }
  return { ...state, shots }
}

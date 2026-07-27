import { loadHighScore, reduceCounterState, saveHighScore } from "/counter.js"

const count = document.querySelector("#count")
const bestCount = document.querySelector("#best-count")
const hitPercentage = document.querySelector("#hit-percentage")
const averageSpeed = document.querySelector("#average-speed")
const averageSpin = document.querySelector("#average-spin")
const lastOutcome = document.querySelector("#last-outcome")
const lastSpeed = document.querySelector("#last-speed")
const lastSpin = document.querySelector("#last-spin")
const status = document.querySelector("#status")
const preview = document.querySelector("#preview")
const previewLabel = document.querySelector("#preview-label")
const restartButton = document.querySelector("#restart")
const main = document.querySelector("main")
const debugMode =
  new URLSearchParams(window.location.search).get("debug") === "true"
const events = new EventSource("/events")
const RESET_HOLD_MS = 400
let counterState = { attempts: [], streak: 0 }
let counterReady = false
let counterEstablished = false
let establishTimer = null
let resetTimer = null
let healthWarning = null
let previewDescription = "Detector view · real-time"
let highScore = loadHighScore(window.localStorage)
bestCount.textContent = highScore

if (debugMode) {
  main.classList.add("debug")
  document.querySelector("#preview-shell").hidden = false
  restartButton.hidden = false

  // Starting an endless MJPEG request before window.load would keep
  // navigation and browser automation waiting forever.
  const startPreview = () => {
    if (preview.hasAttribute("src")) return
    preview.src = "/preview.mjpg"
  }
  if (document.readyState === "complete") {
    startPreview()
  } else {
    window.addEventListener("load", startPreview, { once: true })
  }
  preview.addEventListener("load", () => {
    previewLabel.textContent = previewDescription
  })
  preview.addEventListener("error", () => {
    previewLabel.textContent = "Detector video disconnected"
    preview.removeAttribute("src")
    window.setTimeout(startPreview, 1000)
  })
}

const clearCounterTimers = () => {
  window.clearTimeout(establishTimer)
  window.clearTimeout(resetTimer)
  establishTimer = null
  resetTimer = null
}

const renderCounter = () => {
  if (resetTimer !== null) return
  count.textContent = counterReady ? counterState.streak : "—"
}

const formatDecimal = (value) =>
  Number.isInteger(value) ? String(value) : value.toFixed(1)

const renderStats = () => {
  const stats = counterState.stats
  if (!stats || stats.total === 0) {
    hitPercentage.textContent = "—"
  } else {
    hitPercentage.textContent =
      `${formatDecimal(stats.hitPercentage)}% ` +
      `(${stats.hits} / ${stats.total})`
  }
  averageSpeed.textContent =
    stats?.averageSpeedMps == null
      ? "—"
      : `${stats.averageSpeedMps.toFixed(1)} m/s`
  averageSpin.textContent =
    stats?.averageSpinRevolutionsPerSecond == null
      ? "—"
      : `${stats.averageSpinRevolutionsPerSecond.toFixed(1)} rev/s`
  const lastBall = stats?.lastBall
  lastOutcome.className = "stat-value"
  if (!lastBall) {
    lastOutcome.textContent = "—"
    lastSpeed.textContent = "—"
    lastSpin.textContent = "—"
    return
  }
  const outcome = lastBall.outcome === "hit" ? "Hit" : "Miss"
  lastOutcome.textContent = outcome
  lastOutcome.classList.add(outcome.toLowerCase())
  lastSpeed.textContent =
    lastBall.speedMps == null ? "—" : `${lastBall.speedMps.toFixed(1)} m/s`
  lastSpin.textContent =
    lastBall.spinRevolutionsPerSecond == null
      ? "—"
      : `${lastBall.spinRevolutionsPerSecond.toFixed(1)} rev/s`
}

const establishCounterAfterCurrentBatch = () => {
  if (establishTimer !== null) return
  establishTimer = window.setTimeout(() => {
    establishTimer = null
    counterEstablished = true
    renderCounter()
  }, 100)
}

const showVisibleReset = () => {
  window.clearTimeout(resetTimer)
  count.textContent = "0"
  resetTimer = window.setTimeout(() => {
    resetTimer = null
    renderCounter()
  }, RESET_HOLD_MS)
}

if (debugMode) {
  restartButton.addEventListener("click", async () => {
    restartButton.disabled = true
    clearCounterTimers()
    status.className = ""
    status.textContent = "Restarting video"
    previewLabel.textContent = "Rewinding detector video"
    counterReady = false
    counterEstablished = false
    count.textContent = "—"
    try {
      const response = await fetch("/restart", { method: "POST" })
      if (!response.ok) {
        throw new Error(`restart failed: ${response.status}`)
      }
    } catch (error) {
      status.className = "error"
      status.textContent = "Could not restart detector"
      console.error(error)
    } finally {
      restartButton.disabled = false
    }
  })
}

events.onmessage = ({ data }) => {
  const shot = JSON.parse(data)
  if (shot.type === "session_reset") {
    clearCounterTimers()
    counterState = { attempts: [], streak: 0, stats: null }
    counterReady = false
    counterEstablished = false
    healthWarning = null
  } else {
    counterState = reduceCounterState(counterState, shot)
    if (shot.type === "attempt_upsert" && shot.state === "finalized") {
      if (!counterReady) establishCounterAfterCurrentBatch()
      counterReady = true
    }
  }
  if (
    shot.type === "attempt_upsert" &&
    shot.state === "finalized" &&
    shot.outcome !== "hit" &&
    counterEstablished
  ) {
    showVisibleReset()
  } else {
    renderCounter()
  }
  renderStats()
  if (counterState.streak > highScore) {
    highScore = counterState.streak
    bestCount.textContent = highScore
    saveHighScore(window.localStorage, highScore)
  }
  if (shot.type === "analyzer_exit" && shot.returncode !== 0) {
    healthWarning = shot
    status.className = "error"
    status.textContent = `Detector stopped (exit ${shot.returncode})`
    previewLabel.textContent = "Detector failed — check terminal"
  } else if (shot.type === "counter_health") {
    if (shot.status === "warning") {
      healthWarning = shot
      status.className = "error"
      status.textContent = `Warning: ${shot.message}`
    } else if (
      shot.status === "recovered" &&
      healthWarning?.code === shot.code
    ) {
      healthWarning = null
      status.className = "connected"
      status.textContent = shot.message
    }
  } else if (shot.type === "preview_only") {
    previewDescription = "Video preview · detector disabled"
    previewLabel.textContent = previewDescription
    status.className = "connected"
    status.textContent = shot.message
  } else if (shot.type === "preview_error") {
    status.className = "error"
    status.textContent = `Video preview failed: ${shot.message}`
    previewLabel.textContent = "Video preview failed"
  } else if (shot.type === "counter_status" && !healthWarning) {
    status.className = "connected"
    status.textContent = shot.message
    if (shot.status === "warming_up") {
      clearCounterTimers()
      counterReady = false
      counterEstablished = false
      count.textContent = "—"
    }
  } else if (shot.type === "attempt_upsert" && !healthWarning) {
    status.className = "connected"
    if (shot.state === "pending") {
      status.textContent = `Ball ${shot.sequence} detected`
    } else if (shot.outcome === "hit") {
      status.textContent = `Hit — streak ${counterState.streak}`
    } else {
      status.textContent = "Miss — streak reset"
    }
  }
  document.dispatchEvent(
    new CustomEvent("counter-update", {
      detail: { message: shot, streak: counterState.streak },
    }),
  )
}

events.onopen = () => {
  if (!healthWarning) {
    status.className = "connected"
    status.textContent = "Shot stream connected"
  }
}

events.onerror = () => {
  status.className = "error"
  status.textContent = "Reconnecting"
}

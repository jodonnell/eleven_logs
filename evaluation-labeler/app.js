const video = document.querySelector("#video")
const labelsElement = document.querySelector("#labels")
const clock = document.querySelector("#clock")
const speed = document.querySelector("#speed")
const counts = document.querySelector("#counts")
const saveStatus = document.querySelector("#save-status")
const auditCount = document.querySelector("#audit-count")

let labels = []
let undoStack = []
let saveChain = Promise.resolve()
let auditIndex = -1

const formatTime = (seconds) => {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return `${String(minutes).padStart(2, "0")}:${remainder
    .toFixed(3)
    .padStart(6, "0")}`
}

const auditReasons = () => {
  const reasons = new Map()
  labels.forEach((label, index) => {
    if (label.outcome === "uncertain") reasons.set(index, "unresolved")
  })
  return reasons
}

const queueSave = () => {
  const snapshot = labels.map(({ time_seconds, outcome }) => ({
    time_seconds,
    outcome,
  }))
  saveStatus.textContent = "Saving…"
  saveStatus.className = "save-status saving"
  saveChain = saveChain
    .then(async () => {
      const response = await fetch("/api/labels", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labels: snapshot }),
      })
      if (!response.ok) throw new Error(await response.text())
    })
    .then(() => {
      saveStatus.textContent = "Saved"
      saveStatus.className = "save-status saved"
    })
    .catch((error) => {
      saveStatus.textContent = `Save failed: ${error.message}`
      saveStatus.className = "save-status error"
    })
}

const preserveUndo = () => {
  undoStack.push(labels.map((label) => ({ ...label })))
  if (undoStack.length > 100) undoStack.shift()
}

const render = () => {
  labels.sort((left, right) => left.time_seconds - right.time_seconds)
  const reasons = auditReasons()
  labelsElement.replaceChildren()
  labels.forEach((label, index) => {
    const row = document.createElement("li")
    row.className = `${label.outcome}${reasons.has(index) ? " needs-audit" : ""}`
    row.innerHTML = `
      <button class="timestamp" title="Replay this attempt">${formatTime(label.time_seconds)}</button>
      <span class="outcome">${label.outcome}</span>
      <span class="reason">${reasons.get(index) || ""}</span>
      <div class="row-actions">
        <button data-outcome="hit" title="Change to hit">H</button>
        <button data-outcome="miss" title="Change to miss">M</button>
        <button data-outcome="uncertain" title="Mark uncertain">U</button>
        <button data-delete title="Delete label">×</button>
      </div>`
    row.querySelector(".timestamp").addEventListener("click", () => {
      video.currentTime = label.time_seconds
      video.pause()
    })
    row.querySelectorAll("[data-outcome]").forEach((button) => {
      button.addEventListener("click", () => {
        preserveUndo()
        label.outcome = button.dataset.outcome
        render()
        queueSave()
      })
    })
    row.querySelector("[data-delete]").addEventListener("click", () => {
      preserveUndo()
      labels.splice(index, 1)
      render()
      queueSave()
    })
    labelsElement.append(row)
  })
  const hits = labels.filter(({ outcome }) => outcome === "hit").length
  const misses = labels.filter(({ outcome }) => outcome === "miss").length
  const uncertain = labels.length - hits - misses
  counts.textContent = `${labels.length} labels · ${hits} hit · ${misses} miss${
    uncertain ? ` · ${uncertain} uncertain` : ""
  }`
  auditCount.textContent = `${reasons.size} uncertain`
}

const addLabel = (outcome) => {
  preserveUndo()
  labels.push({ time_seconds: Number(video.currentTime.toFixed(3)), outcome })
  render()
  queueSave()
}

const undo = () => {
  const previous = undoStack.pop()
  if (!previous) return
  labels = previous
  render()
  queueSave()
}

const seek = (seconds) => {
  video.currentTime = Math.max(0, Math.min(video.duration || Infinity, video.currentTime + seconds))
}

const togglePlayback = () => {
  if (video.paused) video.play()
  else video.pause()
}

const changeSpeed = (direction) => {
  const choices = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
  const current = choices.indexOf(video.playbackRate)
  const next = Math.max(0, Math.min(choices.length - 1, current + direction))
  video.playbackRate = choices[next]
  speed.textContent = `${video.playbackRate}×`
}

const auditNext = () => {
  const indexes = [...auditReasons().keys()]
  if (!indexes.length) return
  auditIndex = (auditIndex + 1) % indexes.length
  const index = indexes[auditIndex]
  video.currentTime = Math.max(0, labels[index].time_seconds - 2)
  video.pause()
  labelsElement.children[index]?.scrollIntoView({ block: "center" })
}

const exportLabels = () => {
  const content = JSON.stringify({ version: 1, labels }, null, 2)
  const link = document.createElement("a")
  link.href = URL.createObjectURL(new Blob([content], { type: "application/json" }))
  link.download = "evaluation-labels.json"
  link.click()
  URL.revokeObjectURL(link.href)
}

document.querySelector("#hit").addEventListener("click", () => addLabel("hit"))
document.querySelector("#miss").addEventListener("click", () => addLabel("miss"))
document.querySelector("#uncertain").addEventListener("click", () => addLabel("uncertain"))
document.querySelector("#back").addEventListener("click", () => seek(-1))
document.querySelector("#forward").addEventListener("click", () => seek(1))
document.querySelector("#play").addEventListener("click", togglePlayback)
document.querySelector("#undo").addEventListener("click", undo)
document.querySelector("#audit").addEventListener("click", auditNext)
document.querySelector("#export").addEventListener("click", exportLabels)

document.addEventListener("keydown", (event) => {
  if (event.repeat || ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return
  const key = event.key.toLowerCase()
  const actions = {
    h: () => addLabel("hit"),
    m: () => addLabel("miss"),
    u: () => addLabel("uncertain"),
    z: undo,
    a: auditNext,
    " ": togglePlayback,
    arrowleft: () => seek(-1),
    arrowright: () => seek(1),
    "[": () => changeSpeed(-1),
    "]": () => changeSpeed(1),
  }
  if (actions[key]) {
    event.preventDefault()
    actions[key]()
  }
})

video.addEventListener("timeupdate", () => {
  clock.textContent = formatTime(video.currentTime)
})
video.addEventListener("ratechange", () => {
  speed.textContent = `${video.playbackRate}×`
})

const response = await fetch("/api/labels")
const saved = await response.json()
labels = saved.labels || []
saveStatus.textContent = "Saved"
saveStatus.className = "save-status saved"
render()

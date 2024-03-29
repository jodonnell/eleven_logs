import { playerSessions, fileParse } from "./parser.js"
import mean from "lodash/mean.js"
import h337 from "heatmap.js"

const prettyPercentage = (float) => {
  return (float * 100).toFixed(2) + "%"
}

const prettyNumber = (float) => {
  return float.toFixed(2)
}

const statsRow = (session, date) => {
  const topspin = session.allHits.map((h) => h.topspin).filter((x) => x)
  const backspin = session.allHits.map((h) => h.backspin).filter((x) => x)
  const metersPerSecond = session.allHits
    .map((h) => h.metersPerSecond)
    .filter((x) => x)
  const revs = session.allHits.map((h) => h.revolutions).filter((x) => x)
  const consistency = session.consistency

  const tableRow = document.getElementById("table-row")
  const clone = tableRow.cloneNode(true)
  clone.querySelectorAll("div")[0].innerHTML = date
  clone.querySelectorAll("div")[1].innerHTML = prettyNumber(mean(topspin))
  clone.querySelectorAll("div")[2].innerHTML = prettyNumber(mean(backspin))
  clone.querySelectorAll("div")[3].innerHTML = prettyNumber(
    mean(metersPerSecond),
  )
  clone.querySelectorAll("div")[4].innerHTML = prettyNumber(mean(revs))
  clone.querySelectorAll("div")[5].innerHTML = prettyPercentage(consistency)

  document.getElementById("table-rows").appendChild(clone)
}

const weekly = (sessions) => {
  const byWeek = sessions.byWeek()
  const dates = Object.keys(byWeek)
  dates.sort(function (a, b) {
    return new Date(a) - new Date(b)
  })

  dates.map((d) => {
    statsRow(byWeek[d], d)
  })
  statsRow(sessions, "All")
}

let sessions
window.sessions = sessions
const logsUpload = document.getElementById("logs-upload")
logsUpload.onchange = function () {
  const files = logsUpload.files
  document.querySelector(".lds-spinner").style.display = "flex"

  const promises = Object.keys(files).map((i) => {
    return new Promise((resolve, reject) => {
      const file = files[i]
      const reader = new FileReader()
      reader.readAsText(file, "UTF-8")
      reader.onload = function (evt) {
        const contents = evt.target.result
        resolve(fileParse(contents, file.name))
      }
      reader.onerror = reject
    })
  })

  Promise.all(promises).then((values) => {
    document.querySelector("#instructions").style.display = "none"
    document.querySelector("#uploaded").style.display = "block"
    document.querySelector(".lds-spinner").style.display = "none"

    sessions = playerSessions(values)
    weekly(sessions)

    //sessions.backspinServicePercentage

    document.getElementById("serviceFaultPercentage").innerHTML =
      prettyPercentage(sessions.serviceFaultPercentage)
    document.getElementById("serviceAcePercentage").innerHTML =
      prettyPercentage(sessions.serviceAcePercentage)
    document.getElementById("serviceReturnAcePercentage").innerHTML =
      prettyPercentage(sessions.serviceReturnAcePercentage)
    document.getElementById("winServePercentage").innerHTML = prettyPercentage(
      sessions.winServePercentage,
    )

    heatmapInstance = h337.create({
      // only container is required, the rest will be defaults
      container: document.querySelector("#table"),
      radius: 10,
    })

    createHeatmap(sessions)
  })
}

const flipOnAxis = (value, axisMax) => {
  return axisMax - value
}

let heatmapInstance
const createHeatmap = (sessions) => {
  const width = 400
  const height = 350

  const positions = {}

  const forehand = document.querySelector("#forehand").checked
  const backhand = document.querySelector("#backhand").checked
  const net = document.querySelector("#net").checked
  const serve = document.querySelector("#serve").checked
  const rally = document.querySelector("#rally").checked

  sessions
    .allMyHitsToTable(forehand, backhand, net, serve, rally)
    .forEach((collision) => {
      let posx = parseInt((collision.posx + 0.8) * 250) // 400 / range
      let posz = flipOnAxis(parseInt(collision.posz * 250), 350)

      if (posx > width || posx < 0) return

      if (posz > height || posz < 0) return

      if (!(posx % 5 === 0)) {
        posx = posx - (posx % 5)
      }

      if (!(posz % 5 === 0)) {
        posz = posz - (posz % 5)
      }

      if (!positions[posx]) positions[posx] = {}

      if (!positions[posx][posz]) positions[posx][posz] = 1
      else {
        positions[posx][posz] += 1
      }
    })

  const points = []
  let max = 0
  Object.keys(positions).forEach((posx) => {
    Object.keys(positions[posx]).forEach((posz) => {
      points.push({
        x: posx,
        y: posz,
        value: positions[posx][posz],
      })
      max = Math.max(max, positions[posx][posz])
    })
  })

  const data = {
    max,
    data: points,
  }
  heatmapInstance.setData(data)
}

const checkItem = document.querySelectorAll("[type='checkbox']")

for (let i = 0; i < checkItem.length; i += 1) {
  checkItem[i].addEventListener("change", () => {
    createHeatmap(sessions)
  })
}

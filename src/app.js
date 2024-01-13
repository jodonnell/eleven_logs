import { playerSessions, fileParse } from "./parser.js"
import h337 from "heatmap.js"


const prettyPercentage = (float) => {
  return (float * 100).toFixed(2) + '%'
}

let sessions
const logsUpload = document.getElementById("logs-upload")
logsUpload.onchange = function () {
  const files = logsUpload.files

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
    sessions = playerSessions(values)

    document.getElementById("serviceFaultPercentage").innerHTML = prettyPercentage(sessions.serviceFaultPercentage)
    document.getElementById("serviceAcePercentage").innerHTML = prettyPercentage(sessions.serviceAcePercentage)
    document.getElementById("serviceReturnAcePercentage").innerHTML = prettyPercentage(sessions.serviceReturnAcePercentage)
    document.getElementById("winServePercentage").innerHTML = prettyPercentage(sessions.winServePercentage)

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

  console.log(sessions.allMyHitsToTable)
  //console.log(sessions.allTheirHitsToTable)

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

  console.log(positions)
  console.log(points)
  console.log(max)
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

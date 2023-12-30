import { playerSessions, fileParse } from './parser.js'
import h337 from "heatmap.js";


const template = (sessions) => (
`
<div>
service fault percentage: ${sessions.serviceFaultPercentage}
</div>
<div>
win on service percentage: ${sessions.serviceAcePercentage}
</div>
<div>
win on service return percentage: ${sessions.serviceReturnAcePercentage}
</div>
<div>
win point on serve percentage: ${sessions.winServePercentage}
</div>
`
)

const logsUpload = document.getElementById('logs-upload')
logsUpload.onchange = function () {
  const files = logsUpload.files

  const promises = Object.keys(files).map(i => {
    return new Promise((resolve, reject) => {
      const file = files[i]
      const reader = new FileReader();
      reader.readAsText(file, "UTF-8");
      reader.onload = function (evt) {
        const contents = evt.target.result
        resolve(fileParse(contents, file.name))
      }
      reader.onerror = reject
    })
  })

  Promise.all(promises).then((values) => {
    const sessions = playerSessions(values)
    document.getElementById("page").innerHTML = template(sessions)


    createHeatmap(sessions)
  })
}

const createHeatmap = (sessions) => {
    const heatmapInstance = h337.create({
      // only container is required, the rest will be defaults
      container: document.querySelector('#table'),
      radius: 8
    });


  const width = 400
  const height = 350

  console.log(sessions.allMyHitsToTable)
  //console.log(sessions.allTheirHitsToTable)

  const positions = {}
  sessions.allMyHitsToTable.forEach((collision) => {
    //const posx = parseInt((collision.posx + 1) * 400)
    const posx = parseInt((collision.posx + 0.5) * 400)
    const posz = parseInt((collision.posz + 1.4) * 350)

    if (posx > 400 || posx < 0)
      return

    if (posz > 350 || posz < 0)
      return

    if (!positions[posx])
      positions[posx] = {}

    if (!positions[posx][posz])
      positions[posx][posz] = 1
    else {
      positions[posx][posz] += 1
    }
  })

  const points = []
  let max = 0
  Object.keys(positions).forEach(posx => {
    Object.keys(positions[posx]).forEach(posz => {
      points.push({
        x: posx,
        y: posz,
        value: positions[posx][posz]
      })
      max = Math.max(max, positions[posx][posz]);
    })
  })

  console.log(positions)
  console.log(points)
  console.log(max)
  const data = {
    max,
    data: points
  }
  heatmapInstance.setData(data)
}

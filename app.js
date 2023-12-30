import { playerSessions, fileParse } from './parser.js'

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
  })
}

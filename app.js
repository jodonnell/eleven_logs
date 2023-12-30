import { parseFiles } from './parser.js'

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
  const file = logsUpload.files[0]

  const reader = new FileReader();
  reader.readAsText(file, "UTF-8");
  reader.onload = function (evt) {
    const contents = evt.target.result
    const sessions = parseFiles(file.name, contents)

    document.getElementById("page").innerHTML = template(sessions)
  }
  reader.onerror = function (evt) {
    document.getElementById("page").innerHTML = "error reading file";
  }
}

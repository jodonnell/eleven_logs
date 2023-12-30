import { parseFiles } from './parser.js'

const logsUpload = document.getElementById('logs-upload')
logsUpload.onchange = function () {
  const file = logsUpload.files[0]

  const reader = new FileReader();
  reader.readAsText(file, "UTF-8");
  reader.onload = function (evt) {
    const contents = evt.target.result
    const sessions = parseFiles(file.name, contents)
    console.log('service fault percentage: ', sessions.serviceFaultPercentage)
    document.getElementById("page").innerHTML = `service fault percentage: ${sessions.serviceFaultPercentage}`
  }
  reader.onerror = function (evt) {
    document.getElementById("page").innerHTML = "error reading file";
  }
}

document.addEventListener("DOMContentLoaded", function(event) {
  const page = document.getElementById('page')
  page.innerHTML = 'cow'
})

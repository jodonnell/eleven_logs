const parser = require('./parser')

const sessions = parser.allFileParser('/logs/')

console.log(sessions.serviceFaultPercentage)
console.log(sessions.serviceAcePercentage)
console.log(sessions.serviceReturnAcePercentage)
console.log(sessions.winServePercentage)

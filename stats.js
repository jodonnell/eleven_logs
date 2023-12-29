const parser = require('./parser')

const sessions = parser.allFileParser('/logs/')

console.log(sessions.serviceFaultPercentage)

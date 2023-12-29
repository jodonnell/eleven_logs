const parser = require('./parser')

const sessions = parser.allFileParser('/logs/')

console.log('service fault percentage: ', sessions.serviceFaultPercentage)
console.log('win on service percentage: ', sessions.serviceAcePercentage)
console.log('win on service return percentage: ', sessions.serviceReturnAcePercentage)
console.log('win point on serve percentage: ', sessions.winServePercentage)

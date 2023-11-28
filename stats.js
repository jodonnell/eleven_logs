const parser = require('./parser')

const average = (array) => {
  let total = 0;
  for(let i = 0; i < array.length; i++) {
    total += array[i];
  }
  return total / array.length;
}

const sessions = parser.allFileParser('/logs/')

const allHits = []
sessions.sessions.forEach(s => s.allHits().forEach(h => allHits.push(h.metersPerSecond)))
console.log(average(allHits))

const lastWeek = []
sessions.lastWeek().forEach(s => s.allHits().forEach(h => lastWeek.push(h.metersPerSecond)))
console.log(average(lastWeek))

console.log(sessions.sessions.map(s => s.matches))

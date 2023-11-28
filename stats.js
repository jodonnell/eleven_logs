const parser = require('./parser')

const sessions = parser.allFileParser()

const allHits = []

console.log(sessions.forEach(x => x))

sessions.forEach(s => s.matches.forEach(m => m.rounds.forEach(r => r.points.forEach(p => p.hits.forEach(h => allHits.push(h.metersPerSecond))))))

var total = 0;
for(var i = 0; i < allHits.length; i++) {
    total += allHits[i];
}
var avg = total / allHits.length;
console.log(avg)

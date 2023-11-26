const fs = require('fs')

class Match {
  constructor(opponent) {
    this.opponent = opponent
    this.rounds = []
  }
}

class Round {
  constructor() {
    this.points = []
  }
}

class Point {
  constructor() {
    this.hits = []
    this.didIWin = false
  }
}

class Hit {
  constructor(metersPerSecond, revolutions) {
    this.metersPerSecond = metersPerSecond
    this.revolutions = revolutions
  }
}


const magnitude = (a, b, c) => {
  return Math.sqrt(Math.abs(a) ** 2 + Math.abs(b) ** 2 + Math.abs(c) ** 2)
}

const revolutions = (line) => {
  if (!line.match)
    console.log(line)
  const match = line.match(/postCollisionState:.*rRate:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/)
  if (match) {
    const a = parseFloat(match[1]) / 360.0
    const b = parseFloat(match[2]) / 360.0
    const c = parseFloat(match[3]) / 360.0

    return magnitude(a, b, c)
  }
}

const velocity = (line) => {
  const match = line.match(/postCollisionState:.*vel:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/)
  if (match) {
    const a = parseFloat(match[1])
    const b = parseFloat(match[2])
    const c = parseFloat(match[3])

    return magnitude(a, b, c)
  }
}

const files = fs.readdirSync(__dirname + '/logs/')
files.forEach((file) => {
  console.log(file)

  const lines = fs.readFileSync(__dirname + '/logs/' + file, 'utf8').split('\n')
  lines.forEach((line, i) => {
    const metersPerSec = velocity(line)
    const revsPerSec = revolutions(line)
    if (revsPerSec) {
      new Hit(metersPerSec, revsPerSec)
    }

    if (line.match(/\[Activity\]Sending MP match prefab activity/)) {
      const nextLine = lines[i + 1]
      console.log(JSON.parse(nextLine)["PlayerNames"])
    }

  })

})

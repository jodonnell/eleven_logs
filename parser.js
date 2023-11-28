const fs = require('fs')

class PlayerSessions {
  constructor(sessions) {
    this.sessions = sessions
  }

  lastWeek() {
    const currentDate = new Date()
    const lastWeekDate = new Date(currentDate.getTime() - 7 * 24 * 60 * 60 * 1000);
    return this.sessions.filter(s => s.date > lastWeekDate)
  }
}

class PlaySession {
  constructor(date, matches) {
    this.date = date
    this.matches = matches
  }

  allHits() {
    const hits = []
    this.matches.forEach(m => m.rounds.forEach(r => r.points.forEach(p => p.hits.forEach(h => hits.push(h)))))
    return hits
  }
}

class Match {
  constructor(opponent, rounds) {
    this.opponent = opponent
    this.rounds = rounds
  }
}

class Round {
  constructor(points) {
    this.points = points
  }
}

class Point {
  constructor(hits) {
    this.hits = hits
    this.didIWin = false
  }
}

class Hit {
  constructor(vx, vy, vz, rx, ry, rz) {
    this.vx = vx
    this.vy = vy
    this.vz = vz
    this.rx = rx
    this.ry = ry
    this.rz = rz

    this.metersPerSecond = magnitude(vx, vy, vz)
    this.revolutions = magnitude(rx, ry, rz)
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

const pointParser = (point) => {
  const matches = point.match(/postCollisionState:.*vel:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\).*rRate:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/g)
  if (!matches)
    return

  return matches.map(match => {
    const vel = match.match(/vel:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/)
    const rrate = match.match(/rRate:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/)

    return new Hit(
      parseFloat(vel[1]),
      parseFloat(vel[2]),
      parseFloat(vel[3]),
      parseFloat(rrate[1]) / 360.0,
      parseFloat(rrate[2]) / 360.0,
      parseFloat(rrate[3]) / 360.0,
    )
  })
}

const roundParser = (round) => {
  const points = round.split(/ProcessGameEvent result of eleven collision:[^C]/)
  const allPoints =  points.map(point => {
    const hits = pointParser(point)
    if (hits)
      return new Point(hits)
  }).filter(x => x)
  return new Round(allPoints)
}

const gameParser = (game, username) => {
  const match = game.match(/^(\{"PlayerIds":.*)$/mg)
  const playerNames = JSON.parse(match[0])["PlayerNames"]

  let isFirst = true
  let opponent
  if (playerNames[0] === username) {
    opponent = playerNames[1]
    isFirst = false
  } else {
    opponent = playerNames[0]
  }

  let roundLines = game.split(/"RoundScores":\[\[\d+,\d+\],\[0,0\]\]/, 2)
  if (roundLines[1]) {
    const moreRounds = roundLines[1].split(/"RoundScores":\[\[\d+,\d+\],\[\d+,\d+\],\[0,0\]\]/, 2)
    roundLines = [roundLines[0], ...moreRounds]
  }

  const rounds = roundLines.map(round => {
    return roundParser(round)
  })

  return new Match(opponent, rounds)
}

const allFileParser = () => {
  const files = fs.readdirSync(__dirname + '/logs/')
  const allPlaySessions = files.map((file) => {
    console.log(file)

    const contents = fs.readFileSync(__dirname + '/logs/' + file, 'utf8')
    const userNameMatch = contents.match(/Properly authenticated (\w+)/)
    if (!userNameMatch)
      return
    const username = userNameMatch[1]

    const gamesLines = contents.split('Sending MP match prefab activity')
    gamesLines.shift()
    const games = gamesLines.map((game) => {
      return gameParser(game, username)
      // const metersPerSec = velocity(line)
      // const revsPerSec = revolutions(line)
      // if (revsPerSec) {
      //   new Hit(metersPerSec, revsPerSec)
      // }

      // if (line.match(/\[Activity\]Sending MP match prefab activity/)) {
      //   const nextLine = lines[i + 1]
      //   console.log(JSON.parse(nextLine)["PlayerNames"])
      // }
    })

    const stringDate = file.replace('ALL-', '').replace('.log', '').replace('.', '/').replace('.', '/').replace('.', ' ').replace('.', ':').replace('.', ':').replace('.', ' ')
    const date = new Date(stringDate)
    return new PlaySession(date, games)
  })
  return new PlayerSessions(allPlaySessions.filter(x => x))
}

module.exports = {
  allFileParser
}

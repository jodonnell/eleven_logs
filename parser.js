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
  constructor(hits, reason) {
    this.hits = hits
    this.isServer = null
    this.didIWin = null
    this.lostBy = reason
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

const addPointInfo = (pointInfo, allPoints, username) => {
  const playerId = pointInfo[0]["PlayerNames"][0] === "username" ? pointInfo[0]["PlayerIds"][0] : pointInfo[0]["PlayerIds"][1]

  let totalScore = 0
  pointInfo = pointInfo.filter(p => {
    const scores = p["RoundScores"].slice(-1)[0]
    const equal = totalScore === (scores[0] + scores[1])
    if (equal)
      totalScore++
    return equal
  })

  for (let i = 0; i < allPoints.length; i++) {
    if (!pointInfo[i])
      allPoints[i].isServer = null
    else {
      allPoints[i].isServer = parseInt(pointInfo[i]["CurrentServer"]) === playerId
      allPoints[i].score = pointInfo[i]["RoundScores"]
    }
  }

}

const roundParser = (round, username) => {
  const points = round.split(/writing: potential point ender:reason/)
  const allPoints =  points.map(point => {
    const hits = pointParser(point)
    const collisionMatch = point.match(/ProcessGameEvent result of eleven collision:.*/g)
    if (hits && collisionMatch) {
      const reasonMatch = collisionMatch.slice(-1)[0].match(/ProcessGameEvent result of eleven collision:(.*?) /)
      return new Point(hits, reasonMatch[1])
    }
  }).filter(x => x)

  const pointInfoMatch = round.match(/Snapshot reads:? \{"PlayerIds".*\}/g)
  // broken round in ALL-11.17.2023.7.09.47.PM.log


  //if (!pointInfoMatch)
    //return null
  const pointInfo = pointInfoMatch.map(s => JSON.parse(s.replace(/Snapshot reads:? /, '')))
  if (allPoints.length > pointInfo.length) {
    console.log('uhoh')
    return null
  }
  addPointInfo(pointInfo, allPoints, username)

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
    return roundParser(round, username)
  }).filter(x => x)

  return new Match(opponent, rounds)
}

const allFileParser = (dir) => {
  const files = fs.readdirSync(__dirname + dir)
  const allPlaySessions = files.map((file) => {
    console.log(file)

    const contents = fs.readFileSync(__dirname + dir + file, 'utf8')
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

const fs = require('fs')
const _ = require('lodash')

class PlayerSessions {
  constructor(sessions) {
    this.sessions = sessions
  }

  lastWeek() {
    const currentDate = new Date()
    const lastWeekDate = new Date(currentDate.getTime() - 7 * 24 * 60 * 60 * 1000);
    return this.sessions.filter(s => s.date > lastWeekDate)
  }

  get matchesWon() {
    return _.sum(this.sessions.map(s => s.matchesWon))
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

  get matchesWon() {
    return this.matches.filter(m => m.won).length
  }
}

class Match {
  constructor(opponent, rounds) {
    this.opponent = opponent
    this.rounds = rounds
  }

  get won() {
    return this.rounds.filter(r => r.won).length > 1
  }

}

class Round {
  constructor(points) {
    this.points = points
  }

  get myScore() {
    return this.points.filter(p => p.didIWin).length
  }

  get theirScore() {
    return this.points.filter(p => !p.didIWin).length
  }

  get won() {
    return this.myScore > this.theirScore
  }
}

class Point {
  constructor(hits, didIWin, isServer) {
    this.hits = hits
    this.isServer = isServer
    this.didIWin = didIWin
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

const didIServe = (serverId, playerId) => {
  return parseInt(serverId) === playerId
}

const didIWin = (roundScore, lastRoundScore, isFirst) => {
  const currentRound = _.last(roundScore)
  if (!lastRoundScore) {
    if (currentRound[0] === 1 && isFirst)
      return true

    if (currentRound[0] === 1 && !isFirst)
      return false

    return currentRound[1] === 1 && !isFirst
  }

  const lastRound = _.last(lastRoundScore)

  if (!currentRound) {// someone one
    if (lastRound[0] > lastRound[1])
      return isFirst
    return !isFirst
  }
  if (currentRound[0] > lastRound[0]) {
    return isFirst
  }
  return !isFirst
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

const roundParser = (round, username, isFirst) => {
  const points = round.split(/"PongGameState":"PrePoint"/)
  points.shift()

  let lastPointInfo = null
  const allPoints =  points.map(point => {

    const pointInfoMatch = point.match(/Snapshot reads:? \{"PlayerIds".*/)
    // broken round in ALL-11.17.2023.7.09.47.PM.log
    const fullString = pointInfoMatch[0].indexOf("PongGameState") === -1 ? pointInfoMatch[0] + '"PongGameState":"PrePoint"}' : pointInfoMatch[0]
    const json = '{' + fullString.split('{')[1]

    //try { JSON.parse(json) } catch (e) {console.log('JSON BREAKS', json)}
    const pointInfo = JSON.parse(json)

    const myPlayerIdString = pointInfo["PlayerNames"][0] === username ? pointInfo["PlayerIds"][0] : pointInfo["PlayerIds"][1]
    const myPlayerId = parseInt(myPlayerIdString)

    const hits = pointParser(point) || []
    const served = didIServe(lastPointInfo?.CurrentServer || pointInfo?.CurrentServer, myPlayerId)

    const won = didIWin(pointInfo.RoundScores, lastPointInfo?.RoundScores, isFirst)
    lastPointInfo = pointInfo

    return new Point(hits, won, served)
  }).filter(x => x)

  //addPointInfo(pointInfo, allPoints, username)

  return new Round(allPoints, false)
}

const gameParser = (game, username) => {
  const match = game.match(/^(\{"PlayerIds":.*)$/mg)
  const playerNames = JSON.parse(match[0])["PlayerNames"]

  let isFirst = true
  let opponent
  if (playerNames[0] === username) {
    opponent = playerNames[1]
  } else {
    opponent = playerNames[0]
    isFirst = false
  }

  let roundLines = game.split(/"RoundScores":\[\[\d+,\d+\],\[0,0\]\]/, 2)
  if (roundLines[1]) {
    const moreRounds = roundLines[1].split(/"RoundScores":\[\[\d+,\d+\],\[\d+,\d+\],\[0,0\]\]/, 2)
    roundLines = [roundLines[0], ...moreRounds]
  }

  const rounds = roundLines.map(round => {
    return roundParser(round, username, isFirst)
  }).filter(x => x)

  return new Match(opponent, rounds)
}

const getDateFromFilename = (filename) => {
  const stringDate = filename.replace('ALL-', '').replace('.log', '').replace('.', '/').replace('.', '/').replace('.', ' ').replace('.', ':').replace('.', ':').replace('.', ' ')
  const splitDate = stringDate.split(' ')
  const dateParts = splitDate[0].split('/')
  const timeParts = splitDate[1].split(':')
  const amOrPm = splitDate[2]
  let hours = timeParts[0]
  if (hours === "12")
    hours = "00"

  if (amOrPm === "PM")
    hours = parseInt(hours, 10) + 12

  return new Date(dateParts[2], parseInt(dateParts[0]) - 1, dateParts[1], hours, timeParts[1], timeParts[2])
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
    })

    return new PlaySession(getDateFromFilename(file), games)
  })
  return new PlayerSessions(allPlaySessions.filter(x => x))
}

module.exports = {
  allFileParser
}

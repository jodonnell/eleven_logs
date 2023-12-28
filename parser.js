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
  constructor(vx, vy, vz, rx, ry, rz, posx, posy, posz) {
    this.vx = vx
    this.vy = vy
    this.vz = vz
    this.rx = rx
    this.ry = ry
    this.rz = rz
    this.posx = posx
    this.posy = posy
    this.posz = posz

    this.metersPerSecond = magnitude(vx, vy, vz)
    this.revolutions = magnitude(rx, ry, rz)
  }

  get isForehand() {
    return this.posx > 0
  }

}

const magnitude = (a, b, c) => {
  return Math.sqrt(Math.abs(a) ** 2 + Math.abs(b) ** 2 + Math.abs(c) ** 2)
}


const pos = (line) => {
  const match = line.match(/postCollisionState:.*pos:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/)
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
    const pos = match.match(/pos:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/)

    return new Hit(
      parseFloat(vel[1]),
      parseFloat(vel[2]),
      parseFloat(vel[3]),
      parseFloat(rrate[1]) / 360.0,
      parseFloat(rrate[2]) / 360.0,
      parseFloat(rrate[3]) / 360.0,
      parseFloat(pos[1]),
      parseFloat(pos[2]),
      parseFloat(pos[3]),
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

const getPointInfo = (point) => {
  const pointInfoMatch = point.match(/Snapshot reads:? \{"PlayerIds".*/)
  const fullString = pointInfoMatch[0].indexOf("PongGameState") === -1 ? pointInfoMatch[0] + '"PongGameState":"PrePoint"}' : pointInfoMatch[0]
  const json = '{' + fullString.split('{')[1]

  return JSON.parse(json)
}

const getMyPlayerId = (pointInfo, username) => {
  const myPlayerIdString = pointInfo["PlayerNames"][0] === username ? pointInfo["PlayerIds"][0] : pointInfo["PlayerIds"][1]
  return parseInt(myPlayerIdString)
}

const roundParser = (round, username, isFirst) => {
  const points = round.split(/"PongGameState":"PrePoint"/)
  points.shift()

  let lastPointInfo = null
  const allPoints =  points.map(point => {
    const pointInfo = getPointInfo(point)
    const myPlayerId = getMyPlayerId(pointInfo, username)

    const hits = pointParser(point) || []
    const served = didIServe(lastPointInfo?.CurrentServer || pointInfo?.CurrentServer, myPlayerId)

    const won = didIWin(pointInfo.RoundScores, lastPointInfo?.RoundScores, isFirst)
    lastPointInfo = pointInfo

    return new Point(hits, won, served)
  }).filter(x => x)


  return new Round(allPoints, false)
}

const getOppenentAndIsFirst = (game, username) => {
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

  return { opponent, isFirst }
}

const getRoundChunks = (game) => {
  let roundLines = game.split(/"RoundScores":\[\[\d+,\d+\],\[0,0\]\]/, 2)
  if (roundLines[1]) {
    const moreRounds = roundLines[1].split(/"RoundScores":\[\[\d+,\d+\],\[\d+,\d+\],\[0,0\]\]/, 2)
    roundLines = [roundLines[0], ...moreRounds]
  }
  return roundLines
}

const gameParser = (game, username) => {
  const { opponent, isFirst } = getOppenentAndIsFirst(game, username)

  const rounds = getRoundChunks(game).map(round => {
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

const getUsername = (fileContents) => {
  const userNameMatch = fileContents.match(/Properly authenticated (\w+)/)
  return userNameMatch[1]
}

const fileParse = (dir, file) => {
  console.log(file)

  const contents = fs.readFileSync(__dirname + dir + file, 'utf8')

  const gamesLines = contents.split('Sending MP match prefab activity')
  gamesLines.shift()
  const games = gamesLines.map((game) => {
    return gameParser(game, getUsername(contents))
  })

  return new PlaySession(getDateFromFilename(file), games)
}

const allFileParser = (dir) => {
  const files = fs.readdirSync(__dirname + dir)
  const allPlaySessions = files.map((file) => fileParse(dir, file))
  return new PlayerSessions(allPlaySessions.filter(x => x))
}

module.exports = {
  allFileParser
}

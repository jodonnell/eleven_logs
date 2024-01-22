import last from "lodash/last.js"

import Point from "./point.js"
import Collision from "./collision.js"
import Round from "./round.js"
import Match from "./match.js"
import Hit from "./hit.js"
import PlaySession from "./play_session.js"
import PlaySessions from "./play_sessions.js"

const xyzParser = (anchor) => {
  const number = "(-?\\d+(?:.\\d+(?:E-\\d+)?)?)"
  const regexString = `\\(${number},${number},${number}\\)`
  const re = new RegExp(`${anchor}${regexString}`)
  return re
}

const getCollidedWith = (collision, isFirst) => {
  const collidedWithMatch = collision.match(/pongGameCollisionType:(.*)/)

  if (!collidedWithMatch) {
    return "TheirHit"
  }

  const collidedWithPart = collidedWithMatch[1]
  const collidedWithSideRemoved = collidedWithPart.slice(0, -1)
  if (isFirst) {
    if (collidedWithPart.endsWith("A")) {
      return "My" + collidedWithSideRemoved
    }
    if (collidedWithPart.endsWith("B")) {
      return "Their" + collidedWithSideRemoved
    }
  } else {
    if (collidedWithPart.endsWith("A")) {
      return "Their" + collidedWithSideRemoved
    }
    if (collidedWithPart.endsWith("B")) {
      return "My" + collidedWithSideRemoved
    }
  }
  return collidedWithPart
}

const cleanUpCollisions = (collisions) => {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const theirServe = collisions?.[0]?.with === "TheirHit"
    const myServe = collisions?.[0]?.with === "MyToss"

    if (
      collisions?.[0]?.with === "TheirHit" &&
      collisions?.[1]?.with === "TheirHit"
    )
      collisions.shift()
    else if (!(theirServe || myServe)) {
      if (collisions.length === 0) {
        break
      }
      collisions.shift()
    } else break
  }
}

const addUnforcedError = (collisions) => {
  let lastHit
  let hitIndex = 0
  let tableIndex = 0
  collisions.forEach((c, i) => {
    if (c.with === "MyHit" || c.with === "TheirHit") {
      hitIndex = i
      lastHit = c
    }

    if (c.with === "MyTable" && lastHit?.with === "TheirHit") {
      tableIndex = i
    }

    if (c.with === "TheirTable" && lastHit?.with === "MyHit") {
      tableIndex = i
    }
  })

  if (hitIndex > tableIndex) return (lastHit.unforcedError = true)
}

const collisionParser = (point, isFirst) => {
  const collisionChunks = point.split(
    /(?:MyCollision:)|(?:Received ball hit from opponent:)/,
  )
  collisionChunks.shift()
  //console.log('POOP NEW POINT')

  let lastHit = null
  let lastCollision = null
  let firstHitHappened = false
  const collisions = collisionChunks.map((collision) => {
    collision = collision.split("YourCollision:")[0]
    collision = collision.replace("pos:", "position:")
    collision = collision.replace("vel:", "velocity:")
    collision = collision.replace("rrate:", "rotationRate:")

    const vel = collision.match(xyzParser("velocity:"))
    const rrate = collision.match(xyzParser("rotationRate:"))
    const pos = collision.match(xyzParser("position:"))

    if (!isFirst && pos) {
      pos[1] = parseFloat(pos[1]) * -1
      pos[3] = parseFloat(pos[3]) * -1
    }

    if (!isFirst && rrate) {
      rrate[1] = parseFloat(rrate[1]) * -1
      rrate[2] = parseFloat(rrate[2]) * -1
      rrate[3] = parseFloat(rrate[3]) * -1
    }

    const collidedWith = getCollidedWith(collision, isFirst)

    const isServe =
      (collidedWith === "MyHit" || collidedWith === "TheirHit") &&
      !firstHitHappened
    if (isServe) {
      firstHitHappened = true
    }

    const offNet = lastCollision?.with === "Net"

    //console.log('POOP', collidedWith)
    const collisionObj = new Collision(
      collidedWith,
      parseFloat(vel?.[1] || 0),
      parseFloat(vel?.[2] || 0),
      parseFloat(vel?.[3] || 0),
      parseFloat(rrate?.[1]) / 360.0,
      parseFloat(rrate?.[2]) / 360.0,
      parseFloat(rrate?.[3]) / 360.0,
      parseFloat(pos?.[1] || 0),
      parseFloat(pos?.[2] || 0),
      parseFloat(pos?.[3] || 0),
      lastHit,
      isServe,
      offNet,
    )

    if (collidedWith === "MyHit" || collidedWith === "TheirHit")
      lastHit = collisionObj

    lastCollision = collisionObj
    return collisionObj
  })

  cleanUpCollisions(collisions)
  addUnforcedError(collisions)

  return collisions
}

const pointParser = (point, isFirst) => {
  const matches = point.match(
    /postCollisionState:.*vel:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\).*rRate:\((-?\d+.\d+),(-?\d+.\d+),(-?\d+.\d+)\)/g,
  )
  if (!matches) return

  return matches.map((match) => {
    const vel = match.match(xyzParser("vel:"))
    const rrate = match.match(xyzParser("rRate:"))
    const pos = match.match(xyzParser("pos:"))

    if (!isFirst && rrate) {
      rrate[1] = parseFloat(rrate[1]) * -1
      rrate[2] = parseFloat(rrate[2]) * -1
      rrate[3] = parseFloat(rrate[3]) * -1
    }

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
  const currentRound = last(roundScore)

  if (!lastRoundScore) {
    if (currentRound?.[0] === 1 && isFirst) return true

    if (currentRound?.[0] === 1 && !isFirst) return false

    return currentRound?.[1] === 1 && !isFirst
  }

  const lastRound = last(lastRoundScore)

  if (!currentRound) {
    // someone one
    if (lastRound[0] > lastRound[1]) return isFirst
    return !isFirst
  }
  if (currentRound[0] > lastRound[0]) {
    return isFirst
  }
  return !isFirst
}

const getPointInfo = (point) => {
  const pointInfoMatch = point.match(/Snapshot reads:? \{"PlayerIds".*/)
  if (!pointInfoMatch) return null
  const fullString =
    pointInfoMatch[0].indexOf("PongGameState") === -1
      ? `${pointInfoMatch[0]}"PongGameState":"PrePoint"}`
      : pointInfoMatch[0]
  const json = "{" + fullString.split("{")[1]

  return JSON.parse(json)
}

const getMyPlayerId = (pointInfo, username) => {
  const myPlayerIdString =
    pointInfo["PlayerNames"][0] === username
      ? pointInfo["PlayerIds"][0]
      : pointInfo["PlayerIds"][1]
  return parseInt(myPlayerIdString)
}

const addHitsToCollisions = (collisions, hits) => {
  let hitIndex = 0
  collisions.forEach((collision) => {
    if (collision.with === "MyHit") {
      collision.hit = hits[hitIndex]
      hitIndex++
    }
  })
}

const roundParser = (round, username, isFirst) => {
  const points = round.split(/"PongGameState":"PrePoint"/)
  points.shift()

  let lastPointInfo = null
  const allPoints = points
    .map((point) => {
      const pointInfo = getPointInfo(point)
      if (!pointInfo) return null
      const myPlayerId = getMyPlayerId(pointInfo, username)

      const hits = pointParser(point, isFirst) || []
      const collisions = collisionParser(point, isFirst) || []
      addHitsToCollisions(collisions, hits)
      if (collisions.length === 0) return null
      const served = didIServe(
        lastPointInfo?.CurrentServer || pointInfo?.CurrentServer,
        myPlayerId,
      )

      const won = didIWin(
        pointInfo.RoundScores,
        lastPointInfo?.RoundScores,
        isFirst,
      )
      lastPointInfo = pointInfo
      const pointObj = new Point(hits, collisions, won, served)
      pointObj.collisions.forEach((c) => (c.point = pointObj))
      return pointObj
    })
    .filter((x) => x)

  const roundObj = new Round(allPoints, false)
  allPoints.forEach((p) => (p.round = roundObj))
  return roundObj
}

const getOppenentAndIsFirst = (game, username) => {
  const match = game.match(/^(\{"PlayerIds":.*)$/gm)
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
    const moreRounds = roundLines[1].split(
      /"RoundScores":\[\[\d+,\d+\],\[\d+,\d+\],\[0,0\]\]/,
      2,
    )
    roundLines = [roundLines[0], ...moreRounds]
  }
  return roundLines
}

const gameParser = (game, username) => {
  const { opponent, isFirst } = getOppenentAndIsFirst(game, username)

  const rounds = getRoundChunks(game)
    .map((round) => {
      return roundParser(round, username, isFirst)
    })
    .filter((x) => x)

  const match = new Match(opponent, rounds)
  rounds.forEach((r) => (r.match = match))
  return match
}

const getDateFromFilename = (filename) => {
  const stringDate = filename
    .replace("ALL-", "")
    .replace(".log", "")
    .replace(".", "/")
    .replace(".", "/")
    .replace(".", " ")
    .replace(".", ":")
    .replace(".", ":")
    .replace(".", " ")
  const splitDate = stringDate.split(" ")
  const dateParts = splitDate[0].split("/")
  const timeParts = splitDate[1].split(":")
  const amOrPm = splitDate[2]
  let hours = timeParts[0]
  if (hours === "12") hours = "00"

  if (amOrPm === "PM") hours = parseInt(hours, 10) + 12

  return new Date(
    dateParts[2],
    parseInt(dateParts[0]) - 1,
    dateParts[1],
    hours,
    timeParts[1],
    timeParts[2],
  )
}

const getUsername = (fileContents) => {
  const userNameMatch = fileContents.match(/Properly authenticated (\w+)/)
  return userNameMatch?.[1] || "wagonman"
}

export const fileParse = (contents, filename) => {
  const gamesLines = contents.split("Sending MP match prefab activity")
  gamesLines.shift()
  const games = gamesLines.map((game) => {
    return gameParser(game, getUsername(contents))
  })

  const playSession = new PlaySession(getDateFromFilename(filename), games)
  games.forEach((g) => (g.session = playSession))
  return playSession
}

export const parseFiles = (filename, contents) => {
  const allPlaySessions = [fileParse(contents, filename)]
  return new PlaySessions(allPlaySessions.filter((x) => x))
}

export const playerSessions = (playSessions) => {
  return new PlaySessions(playSessions.filter((x) => x))
}

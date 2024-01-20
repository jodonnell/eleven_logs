import sum from "lodash/sum.js"
import groupBy from "lodash/groupBy.js"

class PlaySessions {
  constructor(sessions) {
    this.sessions = sessions
  }

  byWeek() {
    const grouped = groupBy(this.sessions, "weekStartDateString")
    const dates = Object.keys(grouped)
    dates.map((date) => {
      grouped[date] = new PlaySessions(grouped[date])
    })
    return grouped
  }

  lastWeek() {
    const currentDate = new Date()
    const lastWeekDate = new Date(
      currentDate.getTime() - 7 * 24 * 60 * 60 * 1000,
    )
    return this.sessions.filter((s) => s.date > lastWeekDate)
  }

  get matchesWon() {
    return sum(this.sessions.map((s) => s.matchesWon))
  }

  get serviceFaultPercentage() {
    let faultCount = 0
    this.allPoints.forEach((p) => {
      if (p.isServer && p.isServiceFault) faultCount += 1
    })

    return faultCount / this.allServingPoints.length
  }

  get allServingPoints() {
    return this.allPoints.filter((p) => {
      return p.isServer
    })
  }

  get allReturningPoints() {
    return this.allPoints.filter((p) => {
      return !p.isServer
    })
  }

  allMyHitsToTable(
    forehand = true,
    backhand = true,
    net = true,
    serve = true,
    rally = true,
  ) {
    return this.allPoints
      .map((p) => {
        return p.collisions
          .map((c, i) => {
            if (c.with === "TheirTable") {
              const lastHitWasServe = p.collisions?.[i - 1]?.with === "TheirHit"

              let exclude = lastHitWasServe
              if (!forehand) exclude = exclude || c.lastHit?.isForehand
              if (!backhand) exclude = exclude || c.lastHit?.isBackhand
              if (!serve) exclude = exclude || c.lastHit?.isServe
              if (!rally) exclude = exclude || !c.lastHit?.isServe
              if (!net) exclude = exclude || c.offNet

              if (!exclude) {
                return c
              }
            }
          })
          .filter((x) => x)
      })
      .flat()
  }

  get allTheirHitsToTable() {
    return this.allPoints
      .map((p) => {
        return p.collisions.filter((c) => c.with === "MyTable")
      })
      .flat()
  }

  get winServePercentage() {
    const wins = this.allServingPoints.filter((p) => {
      return p.didIWin
    })

    return wins.length / this.allServingPoints.length
  }

  get serviceAcePercentage() {
    const aceServeCount = this.allServingPoints.filter((point) => {
      if (point.isServiceFault) return false

      return point.collisions.filter((c) => c.with === "MyTable").length < 2
    })
    return aceServeCount.length / this.allServingPoints.length
  }

  get serviceReturnAcePercentage() {
    const aceServeCount = this.allReturningPoints.filter((point) => {
      if (point.isServiceFault) return false

      return point.collisions.filter((c) => c.with === "TheirTable").length < 2
    })
    return aceServeCount.length / this.allReturningPoints.length
  }

  get allPoints() {
    const points = []
    this.sessions.forEach((s) =>
      s.matches.forEach((m) =>
        m.rounds.forEach((r) => r.points.forEach((p) => points.push(p))),
      ),
    )
    return points
  }

  get allHits() {
    return this.sessions.map((s) => s.hits).flat()
  }
}

export default PlaySessions

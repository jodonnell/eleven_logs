import { parseDirectory } from "../src/directory_parser.js"

const sessions = parseDirectory("/../test_logs/")

describe("log parse", () => {
  beforeEach(() => {
    jest
      .spyOn(sessions, "urlParams")
      .mockImplementation(() => ({ get: jest.fn() }))
  })

  it("has two play sessions", () => {
    expect(sessions.sessions.length).toBe(2)
  })

  it("has all my hits to table", () => {
    expect(sessions.allMyHitsToTable().length).toBe(574)
    expect(sessions.allMyHitsToTable(false).length).toBe(259)
    expect(sessions.allMyHitsToTable(false, false).length).toBe(0)
    expect(sessions.allMyHitsToTable(true, true, false).length).toBe(524)
    expect(sessions.allMyHitsToTable(true, true, true, false).length).toBe(425)
    expect(
      sessions.allMyHitsToTable(true, true, true, true, false).length,
    ).toBe(149)
  })

  it("has correct wins", () => {
    expect(sessions.sessions[0].matchesWon).toBe(1)
    expect(sessions.sessions[1].matchesWon).toBe(5)
    expect(sessions.sessions[1].matches.length).toBe(6)
    expect(sessions.matchesWon).toBe(6)
  })

  it("can get consistency", () => {
    expect(sessions.consistency).toBe(0.8101851851851851)
  })

  it("can get service fault percentage", () => {
    expect(sessions.serviceFaultPercentage).toBe(0.0967741935483871)
  })

  it("shows serve ace percentage", () => {
    expect(sessions.serviceAcePercentage).toBe(0.21935483870967742)
  })

  it("shows serve return ace percentage", () => {
    expect(sessions.serviceReturnAcePercentage).toBe(0.20987654320987653)
  })

  it("shows the percent you win on your serve", () => {
    expect(sessions.winServePercentage).toBe(0.5419354838709678)
  })

  describe("first session", () => {
    const session = sessions.sessions[0]
    const match = session.matches[0]

    const round1 = match.rounds[0]
    const round2 = match.rounds[1]

    const round1Points = match.rounds[0].points
    const round2Points = match.rounds[1].points

    const hits = round1Points[0].hits
    const collisions = round1Points[0].collisions

    it("has the right date", () => {
      expect(session.date.getMonth()).toBe(10)
      expect(session.date.getDate()).toBe(18)
      expect(session.date.getFullYear()).toBe(2023)
      expect(session.date.getHours()).toBe(10)
      expect(session.date.getMinutes()).toBe(13)
      expect(session.date.getSeconds()).toBe(47)
    })

    it("has the right amount of matches", () => {
      expect(session.matches.length).toBe(1)
    })

    it("has the right match winner", () => {
      expect(match.won).toBe(true)
    })

    it("has the right amount of rounds", () => {
      expect(match.rounds.length).toBe(2)
    })

    it("has points winner", () => {
      expect(round1Points[0].didIWin).toBe(false)
      expect(round1Points[1].didIWin).toBe(true)
      expect(round1Points[2].didIWin).toBe(false)
      expect(round1Points[3].didIWin).toBe(true)
      expect(round1Points[4].didIWin).toBe(false)
      expect(round1Points[5].didIWin).toBe(false)
      expect(round1Points[6].didIWin).toBe(true)
      expect(round1Points[7].didIWin).toBe(true)
      expect(round1Points[8].didIWin).toBe(false)
      expect(round1Points[9].didIWin).toBe(true)
      expect(round1Points[10].didIWin).toBe(true)
      expect(round1Points[11].didIWin).toBe(false)
      expect(round1Points[12].didIWin).toBe(true)
      expect(round1Points[13].didIWin).toBe(true)
      expect(round1Points[14].didIWin).toBe(true)
      expect(round1Points[15].didIWin).toBe(true)
      expect(round1Points[16].didIWin).toBe(false)
      expect(round1Points[17].didIWin).toBe(false)
      expect(round1Points[18].didIWin).toBe(false)
      expect(round1Points[19].didIWin).toBe(true)

      expect(round2Points[0].didIWin).toBe(true)
    })

    it("has the correct round score", () => {
      expect(round1.myScore).toEqual(11)
      expect(round1.theirScore).toEqual(9)
      expect(round2.myScore).toEqual(12)
      expect(round2.theirScore).toEqual(10)
    })

    it("has the correct round winner", () => {
      expect(round1.won).toEqual(true)
      expect(round2.won).toEqual(true)
    })

    it("has the correct amount of points", () => {
      expect(round1Points.length).toBe(20)
      expect(round2Points.length).toBe(22)
    })

    it("has hits", () => {
      expect(hits.length).toBe(1)
      expect(hits[0].posx).toBe(0.611844658851624)
      expect(hits[0].isForehand).toBe(true)
    })

    it("has collisions", () => {
      expect(collisions.length).toBe(5)
    })

    it("does flip coords when you are side B", () => {
      expect(collisions[0].posx).toBe(-0.71672260761261)
      expect(collisions[0].posz).toBe(1.82300412654877)
      expect(collisions[0].rx).toBe(-25.802948676215276)
    })

    it("tracks unforced error", () => {
      expect(collisions[0].unforcedError).toBe(false)
      expect(collisions[3].unforcedError).toBe(true)

      expect(round1Points[1].collisions[0].unforcedError).toBe(false)
      expect(round1Points[1].collisions[1].unforcedError).toBe(false)
      expect(round1Points[1].collisions[2].unforcedError).toBe(false)
      expect(round1Points[1].collisions[3].unforcedError).toBe(false)
    })

    it("collisions have what they collided with", () => {
      expect(collisions[0].with).toBe("TheirHit")
      expect(collisions[1].with).toBe("TheirTable")
      expect(collisions[2].with).toBe("MyTable")
      expect(collisions[3].with).toBe("MyHit")
      expect(collisions[4].with).toBe("MyTable")

      expect(
        sessions.sessions[1].matches[1].rounds[1].points[13].collisions[4].with,
      ).toBe("Other")
    })

    it("collisions with my hit has hits", () => {
      expect(collisions[3].hit.metersPerSecond).toBe(5.036091482129569)
    })

    it("collisions with my hit has hits", () => {
      expect(collisions[3].hit.metersPerSecond).toBe(5.036091482129569)
    })

    it("reverses rotation rate on side b", () => {
      expect(collisions[3].hit.rx).toBe(12.630810546875)
    })

    it("collisions have points", () => {
      expect(collisions[0].point.collisions.length).toBe(5)
    })

    it("points have rounds", () => {
      expect(round1Points[0].round).toBe(round1)
    })

    it("rounds have matches", () => {
      expect(round1.match).toBe(match)
    })

    it("matches to have session", () => {
      expect(match.session).toBe(session)
    })

    it("collision is forehand", () => {
      expect(collisions[0].isForehand).toBe(false)
      expect(collisions[0].isBackhand).toBe(true)

      expect(round1Points[2].collisions[1].isForehand).toBe(true)
    })

    it("collision has backspin", () => {
      expect(collisions[0].topspin).toBe(null)
      expect(collisions[0].backspin).toBe(2.78053232828775)
    })

    it("collision has topspin", () => {
      expect(round1Points[5].collisions[4].topspin).toBe(9.911038547092028)
    })

    it("collisions have last hits", () => {
      expect(collisions[4].lastHit).toBe(collisions[3])
    })

    it("has service faults", () => {
      expect(round1Points[0].isServiceFault).toBe(false)
      expect(round1Points[1].isServiceFault).toBe(true)
      expect(round1Points[2].isServiceFault).toBe(false)
      expect(round1Points[3].isServiceFault).toBe(false)
      expect(round1Points[4].isServiceFault).toBe(false)
      expect(round1Points[5].isServiceFault).toBe(false)
      expect(round1Points[6].isServiceFault).toBe(false)
      expect(round1Points[7].isServiceFault).toBe(false)
      expect(round1Points[8].isServiceFault).toBe(false)
      expect(round1Points[9].isServiceFault).toBe(true)
      expect(round1Points[10].isServiceFault).toBe(false)
      expect(round1Points[11].isServiceFault).toBe(false)
      expect(round1Points[12].isServiceFault).toBe(true)
      expect(round1Points[13].isServiceFault).toBe(false)
      expect(round1Points[14].isServiceFault).toBe(false)
      expect(round1Points[15].isServiceFault).toBe(false)
    })

    it("has serves correct for round 1", () => {
      let isServer = false
      for (let i = 0; i < 20; i += 2) {
        expect(round1Points[i].isServer).toBe(isServer)
        expect(round1Points[i + 1].isServer).toBe(isServer)

        isServer = !isServer
      }
    })

    it("has serves correct for round 2", () => {
      let isServer = true
      for (let i = 0; i < 20; i += 2) {
        expect(round2Points[i].isServer).toBe(isServer)
        expect(round2Points[i + 1].isServer).toBe(isServer)

        isServer = !isServer
      }
      expect(round2Points[20].isServer).toBe(true)
      expect(round2Points[21].isServer).toBe(false)
    })
  })

  describe("second session", () => {
    const session = sessions.sessions[1]
    const match = session.matches[0]
    const thirdMatch = session.matches[2]

    const round1 = match.rounds[0]

    const round1Points = round1.points

    const collisions = round1Points[0].collisions

    it("has the right date", () => {
      expect(session.date.getMonth()).toBe(10)
      expect(session.date.getDate()).toBe(19)
      expect(session.date.getFullYear()).toBe(2023)
      expect(session.date.getHours()).toBe(18)
      expect(session.date.getMinutes()).toBe(50)
      expect(session.date.getSeconds()).toBe(38)
    })

    it("has the right amount of matches", () => {
      expect(session.matches.length).toBe(6)
    })

    it("has three rounds", () => {
      expect(thirdMatch.rounds.length).toBe(3)
    })

    it("has the right matches won", () => {
      expect(session.matches[0].won).toBe(false)
      expect(session.matches[1].won).toBe(true)
      expect(session.matches[2].won).toBe(true)
      expect(session.matches[3].won).toBe(true)
      expect(session.matches[4].won).toBe(true)
      expect(session.matches[5].won).toBe(true)
    })

    it("does not flip coords when you are side A", () => {
      expect(collisions[0].posx).toBe(0.288470834493637)
      expect(collisions[0].posz).toBe(1.73829102516174)
      expect(collisions[0].rx).toBe(-37.32234971788194)
    })
  })
})

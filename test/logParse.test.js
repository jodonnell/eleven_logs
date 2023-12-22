const parser = require('../parser')

const sessions = parser.allFileParser('/test_logs/')

const session = sessions.sessions[0]
const match = session.matches[0]

const round1 = match.rounds[0]
const round2 = match.rounds[1]

const round1Points = match.rounds[0].points
const round2Points = match.rounds[1].points

describe("log parse", () => {
  it("has one session", () => {
    expect(sessions.sessions.length).toBe(1)
  })

  it("has the right date", () => {
    expect(session.date.getMonth()).toBe(10)
    expect(session.date.getDay()).toBe(6)
    expect(session.date.getFullYear()).toBe(2023)
  })

  it("has the right amount of matches", () => {
    expect(session.matches.length).toBe(1)
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

  it.skip("has the correct round score", () => {
    expect(round1.myScore).toEqual(11)
    expect(round1.opponentScore).toEqual(9)
    expect(round2.myScore).toEqual(12)
    expect(round2.opponentScore).toEqual(10)
  })

  it.skip("has the correct round winner", () => {
    expect(round1.won).toEqual(true)
    expect(round2.won).toEqual(true)
  })

  it("has the correct amount of points", () => {
    expect(round1Points.length).toBe(20)
    expect(round2Points.length).toBe(22)
  })

  it("has serves correct for round 1", () => {
    let isServer = false
    for (let i = 0; i < 20; i+= 2) {
      expect(round1Points[i].isServer).toBe(isServer)
      expect(round1Points[i + 1].isServer).toBe(isServer)

      isServer = !isServer
    }
  })

  it("has serves correct for round 2", () => {
    let isServer = true
    for (let i = 0; i < 20; i+= 2) {
      expect(round2Points[i].isServer).toBe(isServer)
      expect(round2Points[i + 1].isServer).toBe(isServer)

      isServer = !isServer
    }
    expect(round2Points[20].isServer).toBe(true)
    expect(round2Points[21].isServer).toBe(false)
  })
})

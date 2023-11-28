const parser = require('../parser')

const sessions = parser.allFileParser('/test_logs/')

describe("log parse", () => {
  it("has one session", () => {
    expect(sessions.sessions.length).toBe(1)
  })

  it("has the right date", () => {
    expect(sessions.sessions[0].date.getMonth()).toBe(10)
    expect(sessions.sessions[0].date.getDay()).toBe(6)
    expect(sessions.sessions[0].date.getFullYear()).toBe(2023)
  })

  it("has the right amount of matches", () => {
    expect(sessions.sessions[0].matches.length).toBe(1)
  })

  it("has the right amount of rounds", () => {
    expect(sessions.sessions[0].matches[0].rounds.length).toBe(2)
  })

  it("has points correct", () => {
    const round1 = sessions.sessions[0].matches[0].rounds[0].points
    const round2 = sessions.sessions[0].matches[0].rounds[1].points

    expect(round1[0].didIWin).toBe(false)
    expect(round1[0].lostBy).toBe("SenderLoss_DidNotHitFarSide")

    expect(round1[1].didIWin).toBe(true)
    expect(round1[1].lostBy).toBe("SenderLoss_DidNotHitFarSide")

    expect(round1[2].didIWin).toBe(false)

    expect(round1[3].didIWin).toBe(true)
  })

  it("has serves correct", () => {
    const round1 = sessions.sessions[0].matches[0].rounds[0].points
    const round2 = sessions.sessions[0].matches[0].rounds[1].points

    let isServer = false
    for (let i = 0; i < 30; i+= 2) {
      expect(round1[i].isServer).toBe(isServer)
      expect(round1[i + 1].isServer).toBe(isServer)

      isServer = !isServer
    }
  })

})

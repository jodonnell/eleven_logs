class PlaySession {
  constructor(date, matches) {
    this.date = date
    this.matches = matches
  }

  allHits() {
    const hits = []
    this.matches.forEach((m) =>
      m.rounds.forEach((r) =>
        r.points.forEach((p) => p.hits.forEach((h) => hits.push(h))),
      ),
    )
    return hits
  }

  get matchesWon() {
    return this.matches.filter((m) => m.won).length
  }
}

export default PlaySession

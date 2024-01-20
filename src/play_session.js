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

  get weekStartDateString() {
    const day = this.date.getDay()
    const diff = this.date.getDate() - day
    return new Date(this.date.setDate(diff)).toLocaleDateString()
  }

  get matchesWon() {
    return this.matches.filter((m) => m.won).length
  }

  get hits() {
    return this.matches.map((m) => m.hits).flat()
  }
}

export default PlaySession

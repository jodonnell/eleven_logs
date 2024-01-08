class Match {
  constructor(opponent, rounds) {
    this.opponent = opponent
    this.rounds = rounds
  }

  get won() {
    return this.rounds.filter((r) => r.won).length > 1
  }
}

export default Match

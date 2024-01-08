class Round {
  constructor(points) {
    this.points = points
  }

  get myScore() {
    return this.points.filter((p) => p.didIWin).length
  }

  get theirScore() {
    return this.points.filter((p) => !p.didIWin).length
  }

  get won() {
    return this.myScore > this.theirScore
  }
}

export default Round

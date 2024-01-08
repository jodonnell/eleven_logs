class Point {
  constructor(hits, collisions, didIWin, isServer) {
    this.hits = hits
    this.collisions = collisions
    this.isServer = isServer
    this.didIWin = didIWin
  }

  get isServiceFault() {
    if (this.collisions.length < 2) return false

    if (this.isServer) {
      if (this.collisions[0].with !== "MyToss") return true

      if (this.collisions[1].with !== "MyHit") return true

      if (this.collisions[2].with !== "MyTable") return true

      if (!this.collisions[3]) return true

      if (this.collisions[3].with === "Net") {
        if (!this.collisions[4]) return true
        return this.collisions[4].with !== "TheirTable"
      }

      if (this.collisions[3].with !== "TheirTable") return true
    } else {
      if (this.collisions[0].with !== "TheirHit") return true

      if (this.collisions[1].with !== "TheirTable") return true

      if (!this.collisions[2]) return true

      if (this.collisions[2].with !== "MyTable") return true
    }
    return false
  }
}

export default Point

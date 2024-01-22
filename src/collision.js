const magnitude = (a, b, c) => {
  return Math.sqrt(Math.abs(a) ** 2 + Math.abs(b) ** 2 + Math.abs(c) ** 2)
}

class Collision {
  constructor(
    with_,
    vx,
    vy,
    vz,
    rx,
    ry,
    rz,
    posx,
    posy,
    posz,
    lastHit,
    isServe,
    offNet,
  ) {
    this.with = with_
    this.vx = vx
    this.vy = vy
    this.vz = vz
    this.rx = rx
    this.ry = ry
    this.rz = rz
    this.posx = posx
    this.posy = posy
    this.posz = posz
    this.lastHit = lastHit
    this.isServe = isServe
    this.offNet = offNet
    this.unforcedError = false

    this.metersPerSecond = magnitude(vx, vy, vz)
    this.revolutions = magnitude(rx, ry, rz)
  }

  get isForehand() {
    return this.posx >= 0
  }

  get isBackhand() {
    return this.posx < 0
  }

  get topspin() {
    if (this.rz < 0) return null
    return this.rz
  }

  get backspin() {
    if (this.rz > 0) return null
    return Math.abs(this.rz)
  }
}

export default Collision

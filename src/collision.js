const magnitude = (a, b, c) => {
  return Math.sqrt(Math.abs(a) ** 2 + Math.abs(b) ** 2 + Math.abs(c) ** 2)
}

class Collision {
  constructor(with_, vx, vy, vz, rx, ry, rz, posx, posy, posz, lastHit) {
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

    this.metersPerSecond = magnitude(vx, vy, vz)
    this.revolutions = magnitude(rx, ry, rz)
  }

  get isForehand() {
    return this.posx >= 0
  }

  get isBackhand() {
    return this.posx < 0
  }
}

export default Collision

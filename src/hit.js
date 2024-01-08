const magnitude = (a, b, c) => {
  return Math.sqrt(Math.abs(a) ** 2 + Math.abs(b) ** 2 + Math.abs(c) ** 2)
}

class Hit {
  constructor(vx, vy, vz, rx, ry, rz, posx, posy, posz) {
    this.vx = vx
    this.vy = vy
    this.vz = vz
    this.rx = rx
    this.ry = ry
    this.rz = rz
    this.posx = posx
    this.posy = posy
    this.posz = posz

    this.metersPerSecond = magnitude(vx, vy, vz)
    this.revolutions = magnitude(rx, ry, rz)
  }

  get isForehand() {
    return this.posx > 0
  }
}

export default Hit

import { fileParse, playerSessions } from "./parser.js"
import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"

let dirname
if (import.meta.url) {
  const filename = fileURLToPath(import.meta.url) // get the resolved path to the file
  dirname = path.dirname(filename) // get the name of the directory
} else {
  dirname = __dirname
}

export const parseDirectory = (dir) => {
  const files = fs.readdirSync(dirname + dir)
  const allPlaySessions = files.map((file) => {
    console.log(file)
    const contents = fs.readFileSync(dirname + dir + file, "utf8")
    return fileParse(contents, file)
  })
  return playerSessions(allPlaySessions.filter((x) => x))
}

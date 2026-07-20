import { expect, test } from "@playwright/test"
import { existsSync, readFileSync } from "node:fs"

const output = "/tmp/eleven-playwright-sample2.jsonl"
const fixture = JSON.parse(
  readFileSync("test/fixtures/sample2-live-counter.json", "utf8"),
)

test("leaves zero after a reset while processing all of sample2", async ({
  page,
  request,
}) => {
  await page.addInitScript(() => {
    window.visibleCounts = []
    window.counterUpdates = []
    document.addEventListener("counter-update", ({ detail }) => {
      window.counterUpdates.push(detail)
    })
    window.addEventListener("DOMContentLoaded", () => {
      const count = document.querySelector("#count")
      new MutationObserver(() => {
        window.visibleCounts.push(Number(count.textContent))
      }).observe(count, { childList: true, characterData: true, subtree: true })
    })
  })
  await page.goto("/")

  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const wanted = [1, 2, 0, 1]
          let cursor = 0
          for (const count of window.visibleCounts) {
            if (count === wanted[cursor]) cursor += 1
          }
          return cursor === wanted.length
        }),
      { timeout: 85_000 },
    )
    .toBe(true)

  await expect
    .poll(
      async () => (await (await request.get("/status")).json()).done,
      { timeout: 85_000 },
    )
    .toBe(true)
  expect(existsSync(output)).toBe(true)
  const canonical = readFileSync(output, "utf8").trim().split("\n").map(JSON.parse)
  expect(canonical.map((shot) => shot.outcome === "hit")).toEqual(
    fixture.outcomes.map((outcome) => outcome === "hit"),
  )

  const updates = await page.evaluate(() => window.counterUpdates)
  const hitUpdates = updates.filter(({ message }) => message.outcome === "hit")
  expect(hitUpdates).toHaveLength(39)
  expect(hitUpdates.every(({ streak }) => streak > 0)).toBe(true)
  expect(
    updates
      .filter(({ message }) => message.type === "reset")
      .every(({ streak }) => streak === 0),
  ).toBe(true)
  await expect(page.locator("#count")).toHaveText("0")
})

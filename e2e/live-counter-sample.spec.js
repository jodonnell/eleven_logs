import { expect, test } from "@playwright/test"
import { existsSync, readFileSync } from "node:fs"

const output = "/tmp/eleven-playwright-sample2.jsonl"

test("leaves zero after a reset while processing all of sample2", async ({
  page,
  request,
}) => {
  await page.addInitScript(() => {
    window.visibleCounts = []
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
  expect(readFileSync(output, "utf8").trim().split("\n")).toHaveLength(48)
  await expect(page.locator("#count")).toHaveText("0")
})

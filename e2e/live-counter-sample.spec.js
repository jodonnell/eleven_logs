import { expect, test } from "@playwright/test"

test("recovers from zero while analyzing the checked-in sample video", async ({
  page,
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
      { timeout: 40_000 },
    )
    .toBe(true)

  await page.waitForTimeout(5_000)
  await expect(page.locator("#count")).toHaveText("0")
})

import { expect, test } from "@playwright/test"

test("resets after a miss and counts the following hits", async ({ page }) => {
  await page.goto("/")
  const count = page.locator("#count")
  const best = page.locator("#best-count")
  await expect(page.locator("#preview-shell")).toBeHidden()
  await expect(page.locator("#restart")).toBeHidden()

  for (const [expected, expectedBest] of [
    [1, 1],
    [2, 2],
    [3, 3],
    [0, 3],
    [1, 3],
    [2, 3],
    [3, 3],
  ]) {
    await expect(count).toHaveText(String(expected), { timeout: 2_000 })
    await expect(best).toHaveText(String(expectedBest))
  }

  await page.reload()
  await expect(page.locator("#best-count")).toHaveText("3")
  await expect(page.locator("#hit-percentage")).toHaveText("85.7% (6 / 7)")
  await expect(page.locator("#average-speed")).toHaveText("—")
  await expect(page.locator("#average-spin")).toHaveText("—")

  await page.goto("/?debug=true")
  await expect(page.locator("#preview-shell")).toBeVisible()
  await expect(page.locator("#restart")).toBeVisible()
})

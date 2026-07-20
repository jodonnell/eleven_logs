import { expect, test } from "@playwright/test"

test("resets after a miss and counts the following hits", async ({ page }) => {
  await page.goto("/")
  const count = page.locator("#count")

  for (const expected of [1, 2, 3, 0, 1, 2, 3]) {
    await expect(count).toHaveText(String(expected), { timeout: 2_000 })
  }
})

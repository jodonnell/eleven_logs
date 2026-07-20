import { expect, test } from "@playwright/test"
import { existsSync, readFileSync } from "node:fs"

const output = "/tmp/eleven-playwright-sample2.jsonl"
const fixture = JSON.parse(
  readFileSync("test/fixtures/sample2-live-counter.json", "utf8"),
)

const expectedStreaks = (outcomes) => {
  let streak = 0
  return outcomes.map((outcome) => {
    streak = outcome === "hit" ? streak + 1 : 0
    return streak
  })
}

test("displays every finalized sample2 attempt exactly once", async ({
  page,
  request,
}) => {
  await page.addInitScript(() => {
    window.counterUpdates = []
    document.addEventListener("counter-update", ({ detail }) => {
      window.counterUpdates.push(detail)
    })
  })
  await page.goto("/")

  await expect
    .poll(async () => (await (await request.get("/status")).json()).done, {
      timeout: 85_000,
    })
    .toBe(true)

  expect(existsSync(output)).toBe(true)
  const canonical = readFileSync(output, "utf8")
    .trim()
    .split("\n")
    .map(JSON.parse)
  expect(canonical.map(({ outcome }) => outcome)).toEqual(fixture.outcomes)

  const updates = await page.evaluate(() => window.counterUpdates)
  const attempts = updates
    .map(({ message }) => message)
    .filter(({ type }) => type === "attempt_upsert")
  const finalizedUpdates = updates.filter(
    ({ message }) =>
      message.type === "attempt_upsert" && message.state === "finalized",
  )
  const finalized = finalizedUpdates.map(({ message }) => message)

  expect(finalized.map(({ outcome }) => outcome)).toEqual(fixture.outcomes)
  expect(finalizedUpdates.map(({ streak }) => streak)).toEqual(
    expectedStreaks(fixture.outcomes),
  )
  expect(finalized.map(({ attempt_id }) => attempt_id)).toEqual(
    fixture.outcomes.map(
      (_outcome, index) => `attempt-${String(index + 1).padStart(4, "0")}`,
    ),
  )
  expect(new Set(finalized.map(({ attempt_id }) => attempt_id)).size).toBe(48)

  for (const finalizedAttempt of finalized) {
    const versions = attempts.filter(
      ({ attempt_id }) => attempt_id === finalizedAttempt.attempt_id,
    )
    expect(versions.map(({ state }) => state)).toEqual(["pending", "finalized"])
    expect(
      new Set(
        versions
          .filter(({ state }) => state === "finalized")
          .map(({ outcome }) => outcome),
      ).size,
    ).toBe(1)
  }

  for (const attempt of finalized) {
    const limit =
      fixture[`max_${attempt.outcome}_publication_delay_seconds`]
    expect(attempt.attempt_publication_delay_seconds).toBeLessThanOrEqual(limit)
  }
  await expect(page.locator("#count")).toHaveText("0")
})

import { expect, test } from "@playwright/test"
import { existsSync, readFileSync } from "node:fs"

const output = "/tmp/eleven-playwright-side-view.jsonl"
const fixture = JSON.parse(
  readFileSync("test/fixtures/side-view-live-counter.json", "utf8"),
)

const expectedStreaks = (outcomes) => {
  let streak = 0
  return outcomes.map((outcome) => {
    streak = outcome === "hit" ? streak + 1 : 0
    return streak
  })
}

const reconcileFinalized = (attempts) => {
  const latest = new Map()
  for (const attempt of attempts) {
    if (attempt.state !== "finalized") continue
    const current = latest.get(attempt.attempt_id)
    if (
      current === undefined ||
      (attempt.revision ?? 0) > (current.revision ?? 0)
    ) {
      latest.set(attempt.attempt_id, attempt)
    }
  }
  return [...latest.values()].sort(
    (left, right) => left.sequence - right.sequence,
  )
}

const missingPlayerTelemetry = (attempts, source) =>
  attempts
    .filter(({ outcome }) => outcome === "hit")
    .flatMap((hit, index) => {
      const identity =
        hit.attempt_id ?? hit.video_timestamp ?? `hit ${index + 1}`
      return Number.isFinite(hit.hit?.speed_mps) &&
        Number.isFinite(hit.hit?.spin_revolutions_per_second)
        ? []
        : [`${source}: ${identity}`]
    })

test("reconciles every labeled side-view attempt exactly once", async ({
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
  const finalized = reconcileFinalized(attempts)

  expect(finalized.map(({ outcome }) => outcome)).toEqual(fixture.outcomes)
  expect(
    [
      ...missingPlayerTelemetry(canonical, "canonical"),
      ...missingPlayerTelemetry(finalized, "SSE"),
    ],
    "every successful hit must include complete player speed/spin telemetry",
  ).toEqual([])
  expect(finalized.map(({ attempt_id }) => attempt_id)).toEqual(
    fixture.outcomes.map(
      (_outcome, index) => `attempt-${String(index + 1).padStart(4, "0")}`,
    ),
  )
  expect(new Set(finalized.map(({ attempt_id }) => attempt_id)).size).toBe(
    fixture.outcomes.length,
  )

  for (const finalizedAttempt of finalized) {
    const versions = attempts.filter(
      ({ attempt_id }) => attempt_id === finalizedAttempt.attempt_id,
    )
    expect(versions.at(0).state).toBe("pending")
    expect(versions.at(-1).state).toBe("finalized")
    expect(
      new Set(
        versions
          .filter(({ state }) => state === "finalized")
          .map(({ outcome }) => outcome),
      ).size,
    ).toBe(1)
  }

  for (const attempt of finalized) {
    if (attempt.sequence < fixture.latency_assertions_start_sequence) continue
    const limit =
      attempt.outcome === "hit"
        ? fixture.max_hit_contact_publication_delay_seconds
        : fixture.max_miss_decision_publication_delay_seconds
    expect(attempt.feedback_delay_seconds).toBeLessThanOrEqual(limit)
  }
  expect(finalizedUpdates.at(-1).streak).toBe(
    expectedStreaks(fixture.outcomes).at(-1),
  )
  await expect(page.locator("#count")).toHaveText(
    String(expectedStreaks(fixture.outcomes).at(-1)),
  )
})

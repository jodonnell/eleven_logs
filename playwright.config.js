import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./e2e",
  timeout: 20_000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:8899",
    browserName: "chromium",
    launchOptions: { channel: "chrome" },
  },
  webServer: {
    command:
      "python3 scripts/live_counter_server.py --replay-events test/fixtures/live-counter-browser-events.jsonl --replay-interval-ms 350 --port 8899",
    url: "http://127.0.0.1:8899",
    reuseExistingServer: false,
    timeout: 10_000,
  },
})

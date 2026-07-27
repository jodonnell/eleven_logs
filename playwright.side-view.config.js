import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:8898",
    browserName: "chromium",
    launchOptions: { channel: "chrome" },
  },
  webServer: {
    command:
      "python3 scripts/live_counter_server.py side-view-regression.mkv --calibration artifacts/live-2026-07-24-side-calibration.json --wait-for-subscriber --port 8898 --output /tmp/eleven-playwright-side-view.jsonl",
    url: "http://127.0.0.1:8898",
    reuseExistingServer: false,
    timeout: 10_000,
  },
})

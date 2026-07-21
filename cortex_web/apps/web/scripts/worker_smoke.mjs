/** Real-Chromium qualification for the coordinator + adaptive n-way pool. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createRequire } from "node:module";
import { chromium } from "playwright";

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const viteBin = path.resolve(path.dirname(require.resolve("vite/package.json")), "bin/vite.js");
const port = 41973;
const server = spawn(process.execPath, [
  viteBin, "--host", "127.0.0.1", "--port", String(port), "--strictPort",
], { cwd: appDir, stdio: ["ignore", "pipe", "pipe"] });
let serverOutput = "";
server.stdout.on("data", (chunk) => { serverOutput += chunk; });
server.stderr.on("data", (chunk) => { serverOutput += chunk; });

async function waitForServer() {
  const url = `http://127.0.0.1:${port}/test/browser/worker_harness.html`;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (server.exitCode !== null) {
      throw new Error(`Vite exited before readiness:\n${serverOutput}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) return url;
    } catch { /* server is still starting */ }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`Vite did not become ready:\n${serverOutput}`);
}

let browser;
try {
  const url = await waitForServer();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  // Exercise the nested pool under the production-relevant restriction,
  // not Vite's header-free default.
  await page.route("**/test/browser/worker_harness.html", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      headers: {
        ...response.headers(),
        "content-security-policy": "default-src 'self'; script-src 'self'; worker-src 'self'",
      },
    });
  });
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(String(error.stack || error)));
  await page.goto(url);
  await page.waitForFunction(() => window.workerHarnessReady === true);

  const options = { maxQuestions: 1, answerPattern: "yes" };
  const serial = await page.evaluate(
    (runOptions) => window.runWorkerHarness("serial", runOptions), options);
  const adaptive = await page.evaluate(
    (runOptions) => window.runWorkerHarness("dual_branch_auto", runOptions), options);
  assert.deepStrictEqual(adaptive.result, serial.result,
    "adaptive-pool execution changed the authoritative session result");
  assert.equal(pageErrors.length, 0, pageErrors.join("\n"));

  const serialProfile = serial.events.find((event) => event.kind === "execution_profile");
  const adaptiveProfile = adaptive.events.find(
    (event) => event.kind === "execution_profile");
  assert.equal(serialProfile?.executionMode, "serial");
  assert.equal(adaptiveProfile?.executionMode, "adaptive_pool");
  assert.ok(Number.isInteger(adaptiveProfile?.selectedWorkerCount));
  assert.ok(adaptiveProfile.selectedWorkerCount >= 2);
  const calibration = JSON.parse(adaptiveProfile.calibrationResult);
  assert.equal(calibration.result, "measured_v1");
  assert.equal(calibration.selectedWorkerCount, adaptiveProfile.selectedWorkerCount);
  assert.ok(adaptive.events.some((event) =>
    event.kind === "engine_step" && event.executionMode === "adaptive_pool"));
  process.stdout.write(
    `worker-smoke: exact native-IIIC serial/adaptive result parity (${serial.result.nQuestions} questions); `
    + `browser reported ${adaptiveProfile?.hardwareConcurrency} logical cores; `
    + `${adaptiveProfile.selectedWorkerCount} calibrated pool workers used under worker-src 'self'\n`,
  );
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}

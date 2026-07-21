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

  const options = { maxQuestions: 2, answerPattern: "no" };
  const serial = await page.evaluate(
    (runOptions) => window.runWorkerHarness("serial", runOptions), options);
  const transientSpikes = await page.evaluate(
    (runOptions) => window.runWorkerHarness("dual_branch_auto", runOptions), {
      ...options,
      qualificationRuntimeLoads: [
        { trialIndex: 0, sampleCount: 8, meanDelayMs: 11.46, maxDelayMs: 85.4 },
        { trialIndex: 0, sampleCount: 8, meanDelayMs: 0.5, maxDelayMs: 4 },
        { trialIndex: 0, sampleCount: 8, meanDelayMs: 9.55, maxDelayMs: 59.9 },
        { trialIndex: 0, sampleCount: 8, meanDelayMs: 0.4, maxDelayMs: 3 },
        { trialIndex: 0, sampleCount: 8, meanDelayMs: 7, maxDelayMs: 55.9 },
      ],
    });
  const adaptive = await page.evaluate(
    (runOptions) => window.runWorkerHarness("dual_branch_auto", runOptions), {
      ...options,
      qualificationRuntimeLoad: {
        trialIndex: 0, sampleCount: 8, meanDelayMs: 25, maxDelayMs: 75,
      },
    });
  assert.deepStrictEqual(adaptive.result, serial.result,
    "adaptive-pool execution changed the authoritative session result");
  assert.deepStrictEqual(transientSpikes.result, serial.result,
    "transient-spike execution changed the authoritative session result");
  assert.equal(pageErrors.length, 0, pageErrors.join("\n"));

  const serialProfile = serial.events.find((event) => event.kind === "execution_profile");
  const adaptiveProfile = adaptive.events.find(
    (event) => event.kind === "execution_profile");
  const transientProfile = transientSpikes.events.find(
    (event) => event.kind === "execution_profile");
  assert.equal(serialProfile?.executionMode, "serial");
  assert.equal(adaptiveProfile?.executionMode, "adaptive_pool");
  assert.equal(transientProfile?.executionMode, "adaptive_pool");
  assert.equal(transientSpikes.events.some(
    (event) => event.kind === "runtime_pool_adjustment"), false,
  "isolated max-only heartbeat spikes reduced the worker pool");
  assert.ok(Number.isInteger(adaptiveProfile?.selectedWorkerCount));
  assert.ok(adaptiveProfile.selectedWorkerCount >= 2);
  const calibration = JSON.parse(adaptiveProfile.calibrationResult);
  assert.equal(calibration.version, 2);
  assert.equal(calibration.result, "measured_v2");
  assert.ok(calibration.sampleCandidates > 24);
  assert.ok(calibration.sampleScreenCandidates > 0);
  assert.ok(calibration.sampleHistoryObservations > 0);
  assert.equal(calibration.selectedWorkerCount, adaptiveProfile.selectedWorkerCount);
  assert.ok(adaptive.events.some((event) =>
    event.kind === "engine_step" && event.executionMode === "adaptive_pool"));
  const cancelledSpeculation = adaptive.events.some((event) => event.kind === "engine_step"
    && event.phaseV2?.branches.some((branch) => branch.cancelled));
  const observedRanks = adaptive.events.flatMap((event) => event.kind === "engine_step"
    ? [event.phaseV2?.observedOutcomeRank] : []);
  assert.ok(cancelledSpeculation,
    `immediate non-rank-one answer did not cancel speculative work; ranks=${observedRanks}`);
  const runtimeAdjustment = adaptive.events.find(
    (event) => event.kind === "runtime_pool_adjustment");
  assert.equal(runtimeAdjustment?.trialIndex, 1);
  assert.equal(runtimeAdjustment?.reason, "main_realm_load");
  assert.equal(runtimeAdjustment?.trigger, "sustained_mean");
  assert.equal(runtimeAdjustment?.evidenceWindowCount, 1);
  assert.equal(runtimeAdjustment?.previousWorkerCount, adaptiveProfile.selectedWorkerCount);
  assert.ok(runtimeAdjustment.selectedWorkerCount < runtimeAdjustment.previousWorkerCount);
  process.stdout.write(
    `worker-smoke: exact native-IIIC serial/adaptive result parity (${serial.result.nQuestions} questions); `
    + `browser reported ${adaptiveProfile?.hardwareConcurrency} logical cores; `
    + `${adaptiveProfile.selectedWorkerCount} calibrated pool workers used under worker-src 'self'\n`,
  );
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}

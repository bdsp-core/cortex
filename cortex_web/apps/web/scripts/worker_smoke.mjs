/** Real-Chromium qualification for the coordinator + nested branch workers. */
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
  // Exercise the nested worker under the production-relevant restriction,
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

  const serial = await page.evaluate(() => window.runWorkerHarness("serial"));
  const dual = await page.evaluate(() => window.runWorkerHarness("dual_branch_auto"));
  assert.deepStrictEqual(dual.result, serial.result,
    "nested-worker execution changed the authoritative session result");
  assert.equal(pageErrors.length, 0, pageErrors.join("\n"));

  const serialProfile = serial.events.find((event) => event.kind === "execution_profile");
  const dualProfile = dual.events.find((event) => event.kind === "execution_profile");
  assert.equal(serialProfile?.executionMode, "serial");
  assert.equal(dualProfile?.executionMode, "dual_branch");
  assert.ok(dual.events.some((event) =>
    event.kind === "engine_step" && event.executionMode === "dual_branch"));
  process.stdout.write(
    `worker-smoke: exact serial/dual result parity (${serial.result.nQuestions} questions); `
    + `browser reported ${dualProfile?.hardwareConcurrency} logical cores; `
    + `2 compute threads used under worker-src 'self'\n`,
  );
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}

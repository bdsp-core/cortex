/** Production-bank browser benchmark for serial vs dual response branches. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createRequire } from "node:module";
import { chromium } from "playwright";

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const defaultManifest = path.resolve(
  appDir, "../../../cortex_web/apps/web/public/bundle/v1.6-k7-35k/manifest.json",
);
const manifestPath = path.resolve(
  process.env.CORTEX_FULL_BANK_MANIFEST || defaultManifest,
);
if (!fs.existsSync(manifestPath)) {
  throw new Error(
    `full-bank manifest not found: ${manifestPath}; set CORTEX_FULL_BANK_MANIFEST`,
  );
}
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const inputs = {
  taskCodes: manifest.taskCodes,
  taskLabels: manifest.taskLabels,
  taskPatternWords: manifest.taskPatternWords,
  taskClasses: manifest.taskClasses,
  certBlock: manifest.certBlock,
  corrL: manifest.corrL,
  corrT: manifest.corrT,
  nParticles: manifest.nParticles,
  perDomainCap: manifest.perDomainCap,
  terminationPolicy: "precision_v1",
  precisionBandEdges: manifest.precisionBandEdges,
  ellStar: manifest.ellStar,
  segments: manifest.segments.map((segment) => ({
    segId: segment.segId,
    applicableTaskIdx: segment.applicableTaskIdx,
    sMean: segment.sMean,
    sSd: segment.sSd,
  })),
};

const appPort = 41974;
const require = createRequire(import.meta.url);
const viteBin = path.resolve(path.dirname(require.resolve("vite/package.json")), "bin/vite.js");
const server = spawn(process.execPath, [
  viteBin, "--host", "127.0.0.1", "--port", String(appPort), "--strictPort",
], { cwd: appDir, stdio: ["ignore", "pipe", "pipe"] });
let serverOutput = "";
server.stdout.on("data", (chunk) => { serverOutput += chunk; });
server.stderr.on("data", (chunk) => { serverOutput += chunk; });

async function waitForServer() {
  const url = `http://127.0.0.1:${appPort}/test/browser/worker_harness.html`;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (server.exitCode !== null) throw new Error(serverOutput);
    try {
      const response = await fetch(url);
      if (response.ok) return url;
    } catch { /* starting */ }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`Vite did not become ready:\n${serverOutput}`);
}

function percentile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(q * sorted.length) - 1] ?? null;
}

function metrics(run) {
  const answer = run.events.filter((event) => event.kind === "answer_to_item")
    .map((event) => event.durationMs);
  const steps = run.events.filter((event) => event.kind === "engine_step");
  return {
    nQuestions: run.result.nQuestions,
    answerToItemMs: {
      p50: percentile(answer, 0.5), p95: percentile(answer, 0.95),
      max: answer.length ? Math.max(...answer) : null,
    },
    engineTotalMs: {
      p50: percentile(steps.map((step) => step.totalMs), 0.5),
      p95: percentile(steps.map((step) => step.totalMs), 0.95),
    },
    selectionMs: {
      p50: percentile(steps.map((step) => step.selectionMs), 0.5),
      p95: percentile(steps.map((step) => step.selectionMs), 0.95),
    },
    requiredBranchReady: steps.filter((step) => step.requiredBranchReadyAtAnswer).length,
    serialFallbacks: steps.filter((step) => step.executionMode === "serial_fallback").length,
  };
}

let browser;
try {
  const url = await waitForServer();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url);
  await page.waitForFunction(() => window.workerHarnessReady === true);
  const options = {
    inputs,
    maxQuestions: Number(process.env.CORTEX_BENCH_QUESTIONS || 28),
    answerDelayMs: Number(process.env.CORTEX_BENCH_ANSWER_DELAY_MS || 750),
    answerPattern: process.env.CORTEX_BENCH_ANSWER_PATTERN || "yes",
  };
  const serial = await page.evaluate(
    ({ options: runOptions }) => window.runWorkerHarness("serial", runOptions),
    { options },
  );
  const dual = await page.evaluate(
    ({ options: runOptions }) => window.runWorkerHarness("dual_branch_auto", runOptions),
    { options },
  );
  assert.deepStrictEqual(dual.result, serial.result,
    "full-bank dual execution changed the partial-session result");
  const K = inputs.taskCodes.length;
  const packedBytes = inputs.segments.length * (8 + 4 + 16 * K);
  process.stdout.write(`${JSON.stringify({
    manifestPath,
    manifestBytes: fs.statSync(manifestPath).size,
    segments: inputs.segments.length,
    packedBytes,
    answerDelayMs: options.answerDelayMs,
    answerPattern: options.answerPattern,
    serial: metrics(serial),
    dualBranch: metrics(dual),
  }, null, 2)}\n`);
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}

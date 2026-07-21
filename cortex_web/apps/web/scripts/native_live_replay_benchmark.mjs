/** Privacy-safe deterministic Chromium replay of the 2026-07-20 native sitting. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromium, firefox, webkit } from "playwright";

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const browserName = process.env.CORTEX_REPLAY_BROWSER || "chromium";
const browserTypes = { chromium, firefox, webkit };
const browserType = browserTypes[browserName];
if (!browserType) throw new Error(`unsupported CORTEX_REPLAY_BROWSER: ${browserName}`);
const manifestPath = path.resolve(process.env.CORTEX_FULL_BANK_MANIFEST || "");
if (!manifestPath || !fs.existsSync(manifestPath)) {
  throw new Error("set CORTEX_FULL_BANK_MANIFEST to the served v1.6 manifest");
}
const fixturePath = path.resolve(
  appDir, "engine/__testdata__/native_live_replay_20260720.json",
);
const manifestBytes = fs.readFileSync(manifestPath);
const manifestSha256 = createHash("sha256").update(manifestBytes).digest("hex");
const manifest = JSON.parse(manifestBytes.toString());
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
const replayQuestions = Math.min(
  fixture.trials.length,
  Math.max(1, Number(process.env.CORTEX_REPLAY_QUESTIONS || fixture.trials.length)),
);
const requestedComputeMode = process.env.CORTEX_REPLAY_COMPUTE_MODE || "serial";
if (requestedComputeMode !== "serial" && requestedComputeMode !== "dual_branch_auto") {
  throw new Error("CORTEX_REPLAY_COMPUTE_MODE must be serial or dual_branch_auto");
}
const replayTrials = fixture.trials.slice(0, replayQuestions);
assert.equal(fixture.schemaVersion, 1);
assert.equal(fixture.fixtureKind, "privacy_safe_native_live_replay");
assert.equal(manifestSha256, fixture.candidateBankSha256);
assert.equal(fixture.nwayProfile.candidateBankSha256, manifestSha256);
assert.equal(fixture.nwayProfile.responseArtifactSha256,
  "0654fc210e67ece152dcaef6b40641fc9c94cb259d100ed3829d90224bc5ef8b");

function quantile(values, q) {
  const ordered = [...values].sort((a, b) => a - b);
  const position = (ordered.length - 1) * q;
  const low = Math.floor(position), high = Math.min(low + 1, ordered.length - 1);
  return ordered[low] + (position - low) * (ordered[high] - ordered[low]);
}

const precisionBandEdges = manifest.precisionBandEdges
  || manifest.taskCodes.map((_code, k) => {
    const values = manifest.segments
      .filter((segment) => !segment.applicableTaskIdx
        || segment.applicableTaskIdx.includes(k))
      .map((segment) => segment.sMean[k]);
    return [quantile(values, 1 / 3), quantile(values, 2 / 3)];
  });
const excluded = new Set(fixture.candidateExclusion);
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
  terminationPolicy: fixture.terminationPolicy,
  precisionBandEdges,
  ellStar: manifest.ellStar,
  nwayProfile: fixture.nwayProfile,
  segments: manifest.segments
    .filter((segment) => !excluded.has(segment.segId))
    .map((segment) => ({
      segId: segment.segId,
      applicableTaskIdx: segment.applicableTaskIdx,
      sMean: segment.sMean,
      sSd: segment.sSd,
    })),
};

const appPort = Number(process.env.CORTEX_REPLAY_PORT || 41975);
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

function distribution(values) {
  return {
    count: values.length,
    p50: percentile(values, 0.5),
    p90: percentile(values, 0.9),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
    max: values.length ? Math.max(...values) : null,
  };
}

let browser;
try {
  const url = await waitForServer();
  browser = await browserType.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url);
  await page.waitForFunction(() => window.workerHarnessReady === true);
  const options = {
    inputs,
    maxQuestions: replayTrials.length,
    answerDelayMs: Number(process.env.CORTEX_REPLAY_ANSWER_DELAY_MS || 0),
    answerSequence: replayTrials.map((trial) => trial.rawPick),
    expectedItems: replayTrials.map(({ trialIndex, taskK, segId }) => (
      { trialIndex, taskK, segId }
    )),
    sessionId: `web-${fixture.sampleSeed}`,
    deriveSeedFromSessionId: true,
    ...(process.env.CORTEX_REPLAY_HARDWARE_CONCURRENCY ? {
      qualificationHardwareConcurrency: Number(
        process.env.CORTEX_REPLAY_HARDWARE_CONCURRENCY),
    } : {}),
  };
  const run = await page.evaluate(async ({ runOptions, computeMode }) => {
    const replay = await window.runWorkerHarness(computeMode, runOptions);
    const hash = async (value) => {
      const bytes = new TextEncoder().encode(JSON.stringify(value));
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0"))
        .join("");
    };
    const { traj, ...withoutTrajectory } = replay.result;
    return {
      nQuestions: replay.result.nQuestions,
      stopReason: replay.result.stopReason,
      hashes: {
        statisticalResult: await hash(replay.result),
        resultWithoutTrajectory: await hash(withoutTrajectory),
        trajectory: await hash(traj),
        servedSequence: await hash(replay.result.servedSegIds),
        trialDiagnostics: await hash(replay.result.trials),
      },
      events: replay.events,
    };
  }, { runOptions: options, computeMode: requestedComputeMode });
  assert.equal(run.nQuestions, replayTrials.length);
  const steps = run.events.filter((event) => event.kind === "engine_step");
  assert.equal(steps.length, replayTrials.length);
  const answer = run.events.filter((event) => event.kind === "answer_to_item")
    .map((event) => event.durationMs);
  const output = {
    schemaVersion: 1,
    benchmarkKind: "privacy_safe_native_live_replay",
    browserName,
    requestedComputeMode,
    manifestSha256,
    fixtureSha256: createHash("sha256")
      .update(fs.readFileSync(fixturePath)).digest("hex"),
    segments: inputs.segments.length,
    nParticles: inputs.nParticles,
    nMhSteps: 30,
    essThreshold: 0.5,
    trials: run.nQuestions,
    stopReason: run.stopReason,
    exactRecordedItemSequence: true,
    hashes: run.hashes,
    timing: {
      answerToItemMs: distribution(answer),
      engineTotalMs: distribution(steps.map((step) => step.totalMs)),
      selectionMs: distribution(steps.map((step) => step.selectionMs)),
      updateMs: distribution(steps.map((step) => step.updateMs)),
      rejuvenationMs: distribution(steps.map((step) => step.rejuvenationMs)),
      rejuvenationCount: steps.filter((step) => step.rejuvenated).length,
      serialFallbackCount: steps.filter(
        (step) => step.executionMode === "serial_fallback",
      ).length,
    },
    events: run.events,
  };
  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}

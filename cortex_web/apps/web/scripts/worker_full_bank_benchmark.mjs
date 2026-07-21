/** Production-bank browser benchmark for serial vs dual response branches. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createRequire } from "node:module";
import { chromium, firefox, webkit } from "playwright";

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const browserName = process.env.CORTEX_BENCH_BROWSER || "chromium";
const browserTypes = { chromium, firefox, webkit };
const browserType = browserTypes[browserName];
if (!browserType) throw new Error(`unsupported CORTEX_BENCH_BROWSER: ${browserName}`);
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
function quantile(values, q) {
  const ordered = [...values].sort((a, b) => a - b);
  const position = (ordered.length - 1) * q;
  const low = Math.floor(position), high = Math.min(low + 1, ordered.length - 1);
  return ordered[low] + (position - low) * (ordered[high] - ordered[low]);
}
const precisionBandEdges = manifest.precisionBandEdges
  || manifest.taskCodes.map((_code, k) => {
    const values = manifest.segments
      .filter((segment) => !segment.applicableTaskIdx || segment.applicableTaskIdx.includes(k))
      .map((segment) => segment.sMean[k]);
    return [quantile(values, 1 / 3), quantile(values, 2 / 3)];
  });
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
  precisionBandEdges,
  ellStar: manifest.ellStar,
  nwayProfile: {
    engineProfileId: "precision_nway_f1_ensemble9_fisher_v1",
    responseModel: "iiic_conditional_f1_v1",
    responseArtifactId: "iiic-f1-crossfit-ensemble9-rd-20260720",
    responseArtifactSha256: "0654fc210e67ece152dcaef6b40641fc9c94cb259d100ed3829d90224bc5ef8b",
    selectorVersion: "categorical_fisher_totalvar_v1",
    particleProfileVersion: "production_1200p_ess050_30mh_rd",
    engineAlgorithmVersion: "nway_protocol_0.2.0-rd",
    candidateBankSha256: createHash("sha256").update(fs.readFileSync(manifestPath)).digest("hex"),
  },
  segments: manifest.segments
    .filter((segment) => process.env.CORTEX_BENCH_IIIC_ONLY !== "1"
      || segment.testClass !== "spike")
    .map((segment) => ({
    segId: segment.segId,
    applicableTaskIdx: segment.applicableTaskIdx,
    sMean: segment.sMean,
    sSd: segment.sSd,
  })),
};

const appPort = Number(process.env.CORTEX_BENCH_PORT || 41974);
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
  const phases = steps.flatMap((step) => step.phaseV2 ? [step.phaseV2] : []);
  const executionProfile = run.events.find((event) => event.kind === "execution_profile")
    || null;
  const heartbeat = run.events.findLast((event) => event.kind === "event_loop_heartbeat")
    || null;
  const phaseDistribution = (read) => {
    const values = phases.map(read);
    return {
      p50: percentile(values, 0.5),
      p95: percentile(values, 0.95),
      max: values.length ? Math.max(...values) : null,
    };
  };
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
    executionProfile,
    heartbeat,
    phaseV2: phases.length ? {
      candidateBankPreparationMs: {
        p50: percentile(steps.map((step) => step.bankPreparationMs), 0.5),
        p95: percentile(steps.map((step) => step.bankPreparationMs), 0.95),
      },
      candidatePreparationMs: phaseDistribution(
        (phase) => phase.selection.candidatePreparationMs),
      posteriorMomentsMs: phaseDistribution(
        (phase) => phase.selection.posteriorMomentsMs),
      coarseMinSdScanMs: phaseDistribution(
        (phase) => phase.selection.coarseMinSdScanMs),
      entropyScanMs: phaseDistribution((phase) => phase.selection.entropyScanMs),
      fisherScanMs: phaseDistribution((phase) => phase.selection.fisherScanMs),
      exactRefinementMs: phaseDistribution(
        (phase) => phase.selection.exactRefinementMs),
      categoricalUpdateMs: phaseDistribution(
        (phase) => phase.particle.categoricalUpdateMs),
      essMs: phaseDistribution((phase) => phase.particle.essMs),
      resamplingMs: phaseDistribution((phase) => phase.particle.resamplingMs),
      mhProposalGenerationMs: phaseDistribution(
        (phase) => phase.particle.mhProposalGenerationMs),
      mhPriorMs: phaseDistribution((phase) => phase.particle.mhPriorMs),
      mhHistoryLikelihoodMs: phaseDistribution(
        (phase) => phase.particle.mhHistoryLikelihoodMs),
      mhAcceptanceMs: phaseDistribution((phase) => phase.particle.mhAcceptanceMs),
      mhCopyingMs: phaseDistribution((phase) => phase.particle.mhCopyingMs),
    } : null,
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
    maxQuestions: Number(process.env.CORTEX_BENCH_QUESTIONS || 4),
    answerDelayMs: Number(process.env.CORTEX_BENCH_ANSWER_DELAY_MS || 750),
    answerPattern: process.env.CORTEX_BENCH_ANSWER_PATTERN || "yes",
    ...(process.env.CORTEX_BENCH_HARDWARE_CONCURRENCY ? {
      qualificationHardwareConcurrency: Number(
        process.env.CORTEX_BENCH_HARDWARE_CONCURRENCY),
    } : {}),
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
    browserName,
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

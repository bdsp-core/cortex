// Drive ONE complete certification sitting headlessly through the real SPA
// against a running server, as an EXISTING signed-in account, and report the
// session's stamps and diagnostics. Built for the staged n-way activation
// (promotion checklist: offline shadow → internal test → allowlist): the
// internal-test stage signs in with the allowlisted internal account, sits a
// full session end-to-end (spike phase + IIIC phase until the completion
// screen), and prints a JSON report of what the server stamped and what the
// engine reported — sessionId, nwayProfile, computeMode, trial count, stop
// reason, and the result diagnostics.
//
//   CORTEX_DRIVE_EMAIL=... CORTEX_DRIVE_PASSWORD=... \
//     node scripts/nway_session_drive.mjs <baseUrl> [maxTrials]
//
// Exits 0 when the completion screen is reached and the result was ingested;
// 1 on any failure. Picks are deterministic (spike: "yes"; IIIC: cycle 1..6)
// — this is an operational activation probe, not a competence measurement.

import { chromium } from "playwright";

const [baseUrl, maxTrialsArg] = process.argv.slice(2);
const email = process.env.CORTEX_DRIVE_EMAIL;
const password = process.env.CORTEX_DRIVE_PASSWORD;
if (!baseUrl || !email || !password) {
  console.error("usage: CORTEX_DRIVE_EMAIL=.. CORTEX_DRIVE_PASSWORD=.. "
    + "node scripts/nway_session_drive.mjs <baseUrl> [maxTrials]");
  process.exit(2);
}
const maxTrials = Number(maxTrialsArg ?? 420);

const log = (m) => console.log(`  ${m}`);
const report = {
  baseUrl, startedUtc: new Date().toISOString(),
  sessionId: null, nwayProfile: null, terminationPolicy: null,
  computeMode: null, resumed: false, resumedTrials: 0, trialsPosted: 0,
  resultPosted: null, resultStatus: null, stopReason: null, clientErrors: [],
};
let browser;

try {
  browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  page.on("pageerror", (e) => {
    report.clientErrors.push(String(e.message));
    console.error("  [pageerror]", e.message);
  });
  page.on("response", async (res) => {
    const url = res.url();
    const method = res.request().method();
    try {
      if (method === "POST" && url.endsWith("/api/session")) {
        const body = await res.json();
        report.sessionId = body.sessionId;
        report.terminationPolicy = body.terminationPolicy;
        report.computeMode = body.computeMode;
        report.nwayProfile = body.bank?.nwayProfile ?? null;
      } else if (method === "GET" && url.endsWith("/api/session/active")) {
        const active = (await res.json())?.active;
        if (active) {
          report.resumed = true;
          report.sessionId = active.sessionId;
          report.terminationPolicy = active.bank?.terminationPolicy ?? null;
          report.computeMode = active.computeMode;
          report.nwayProfile = active.bank?.nwayProfile ?? null;
          report.resumedTrials = active.trials?.length ?? 0;
        }
      } else if (method === "POST" && url.endsWith("/api/progress")) {
        if (res.ok()) report.trialsPosted += 1;
      } else if (method === "POST" && url.endsWith("/api/progress/batch")) {
        if (res.ok()) {
          const posted = JSON.parse(res.request().postData() ?? "{}");
          report.trialsPosted += posted.trials?.length ?? 0;
        }
      } else if (method === "POST" && url.endsWith("/api/results")) {
        report.resultStatus = res.status();
        report.resultPosted = JSON.parse(
          res.request().postData() ?? "null");
      }
    } catch { /* diagnostics only */ }
  });

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /Sign in/ }).waitFor({ timeout: 15000 });
  await page.locator('input[type="email"]').fill(email);
  await page.locator('input[type="password"]').first().fill(password);
  await page.getByRole("button", { name: /Sign in/ }).click();
  log("sign in ✓");

  const milestoneDialog = page.getByRole("dialog", { name: "Milestone reached" });
  if (await milestoneDialog.waitFor({ state: "visible", timeout: 8000 })
    .then(() => true).catch(() => false)) {
    await milestoneDialog.getByRole("button", { name: "Noted" }).click();
    await milestoneDialog.waitFor({ state: "hidden", timeout: 8000 });
  }
  const testName = /^(?:Start|Resume|(?:Re-)?[Tt]ake) (?:the )?certification test$/i;
  const welcomeDialog = page.getByRole("dialog")
    .filter({ hasText: "Welcome to the CORTEX protocol" });
  const testCta = await welcomeDialog.isVisible().catch(() => false)
    ? welcomeDialog.getByRole("button", { name: testName })
    : page.getByRole("button", { name: testName }).first();
  await testCta.waitFor({ timeout: 15000 });
  await testCta.click();
  log("dashboard → certification test ✓");

  // A stored unfinished sitting offers "Resume your test?" first.
  if (await page.getByRole("button", { name: /^Resume test$/ })
    .waitFor({ timeout: 6000 }).then(() => true).catch(() => false)) {
    await page.getByRole("button", { name: /^Resume test$/ }).click();
    log("resume-confirmation → Resume test ✓");
  }

  // Consent appears for new sittings only; a resumed sitting skips it.
  if (await page.getByRole("button", { name: "I Accept" })
    .waitFor({ timeout: 8000 }).then(() => true).catch(() => false)) {
    await page.getByRole("button", { name: "I Accept" }).click();
    log("consent ✓");
  } else {
    log("consent skipped (resumed sitting)");
  }

  // Tutorial appears for first-time accounts only.
  if (await page.getByText(/STEP 1 OF 5/).waitFor({ timeout: 20000 })
    .then(() => true).catch(() => false)) {
    for (let s = 0; s < 4; s++) {
      await page.getByRole("button", { name: "NEXT" }).click();
      await page.waitForTimeout(120);
    }
    await page.getByRole("button", { name: "BEGIN" }).click();
    log("tutorial ✓");
  }

  // Answer until the completion screen: spike phase takes "1" (yes); the
  // IIIC phase cycles the six classes. The loop is EVENT-driven, not
  // time-driven: after each press it waits for the trial to actually post
  // (serial compute recomputes selection between questions, which can take
  // seconds under load), re-pressing only when nothing landed.
  let complete = false;
  let iiicPick = 0;
  let stalls = 0;
  const wallStart = Date.now();
  const isComplete = () => page.getByText("Assessment Complete").isVisible()
    .catch(() => false);
  while (!complete && Date.now() - wallStart < 60 * 60 * 1000
         && report.trialsPosted <= maxTrials) {
    complete = await isComplete();
    if (complete) break;
    if (await page.getByText("Something went wrong").isVisible()
      .catch(() => false)) {
      throw new Error("assessment entered the error screen");
    }
    const spike = await page.getByRole("button", { name: /^(?:1 · )?Spike$/i })
      .isVisible().catch(() => false);
    const iiic = await page.getByRole("button", { name: /Seizure/ })
      .isVisible().catch(() => false);
    if (spike && !iiic) {
      await page.keyboard.press("1");
    } else if (iiic) {
      await page.keyboard.press(String((iiicPick % 6) + 1));
      iiicPick += 1;
    } else {
      await page.waitForTimeout(400);
      continue;
    }
    // Wait up to 45 s for the press to land (a /api/progress post, the
    // completion screen, or a phase transition), then decide.
    const before = report.trialsPosted;
    for (let w = 0; w < 112 && report.trialsPosted === before; w++) {
      if (await isComplete()) break;
      await page.waitForTimeout(400);
    }
    if (report.trialsPosted === before && !(await isComplete())) {
      stalls += 1;
      if (stalls >= 8) throw new Error(
        `no trial posted after ${stalls} consecutive presses `
        + `(${report.trialsPosted} posted so far)`);
    } else {
      stalls = 0;
    }
    if (report.trialsPosted % 25 === 0 && report.trialsPosted !== before) {
      log(`… ${report.trialsPosted} trials posted`);
    }
  }
  if (!complete) throw new Error(
    `no completion screen (${report.trialsPosted} trials posted)`);
  report.stopReason = report.resultPosted?.stopReason ?? null;
  log(`assessment complete ✓ (${report.trialsPosted} trials posted, `
    + `result status ${report.resultStatus})`);
  report.finishedUtc = new Date().toISOString();
  console.log("=== SESSION REPORT ===");
  console.log(JSON.stringify({
    ...report,
    resultPosted: report.resultPosted && {
      stopReason: report.resultPosted.stopReason,
      nQuestions: report.resultPosted.nQuestions,
      terminationPolicy: report.resultPosted.result?.terminationPolicy,
      nwayProfileId: report.resultPosted.result?.nwayProfile?.engineProfileId,
      verdicts: report.resultPosted.result?.verdicts,
      determinations: report.resultPosted.result?.determinations,
      diagnostics: report.resultPosted.result?.diagnostics ?? null,
    },
  }, null, 2));
  await browser.close();
  process.exit(report.resultStatus === 200 && report.clientErrors.length === 0 ? 0 : 1);
} catch (error) {
  console.error("DRIVE FAILED:", error?.message ?? error);
  console.log(JSON.stringify(report, null, 2));
  try { await browser?.close(); } catch { /* already gone */ }
  process.exit(1);
}

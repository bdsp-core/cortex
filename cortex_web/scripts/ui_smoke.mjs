// Headless UI smoke — drives the whole participant flow in a real browser
// (system Chrome via Playwright) against a running server: public signup →
// consent → registration → tutorial → spike phase (SpikeViewer) → IIIC phase.
// Validates the integration the unit/API tests can't: the screen chain,
// engine-worker boot, EEG/spectrogram render, spike-first sectioning, and
// pick-and-advance answering.
//
//   node scripts/ui_smoke.mjs <baseUrl>
// Exits 0 on success, 1 on any failed milestone. Invoked by ui_smoke.sh.

import { chromium } from "playwright";

const [baseUrl] = process.argv.slice(2);
if (!baseUrl) { console.error("usage: ui_smoke.mjs <baseUrl>"); process.exit(2); }

const log = (m) => console.log(`  ${m}`);
let browser;

try {
  browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  page.on("pageerror", (e) => console.error("  [pageerror]", e.message));
  await page.goto(baseUrl, { waitUntil: "networkidle" });

  // Landing → BEGIN
  await page.getByText("BEGIN ASSESSMENT").click();
  log("landing → begin ✓");

  // Auth: public email/password signup (required-field labels carry a " *",
  // so select by input type / placeholder instead of by label text).
  const email = `qa+${Date.now()}@example.org`;
  await page.locator('input[type="email"]').fill(email);
  const pw = page.locator('input[type="password"]');
  await pw.nth(0).fill("smoke-pass-123");   // Password
  await pw.nth(1).fill("smoke-pass-123");   // Confirm password
  await page.getByPlaceholder("e.g. J. Doe, MD").fill("QA Bot");
  await page.getByRole("button", { name: "Create account" }).click();
  log("signup (email/password) ✓");

  // Consent
  await page.getByRole("button", { name: "I Accept" }).waitFor({ timeout: 10000 });
  await page.getByRole("button", { name: "I Accept" }).click();
  log("consent ✓");

  // Registration (3-page wizard; page 0 requires name + expertise + institution)
  await page.getByPlaceholder("e.g. J.D.").fill("QA Bot");
  await page.locator("select").first().selectOption({ index: 1 });   // expertise
  await page.locator('input[type="text"]:not([placeholder])').first().fill("Test Institution");
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByRole("button", { name: "Continue to tutorial" }).click();
  log("registration ✓");

  // In-context tutorial overlay (over an example IIIC segment): walk the 5
  // coach-marks (NEXT ×4 → BEGIN), which then starts the real test.
  await page.getByText(/STEP 1 OF 5/).waitFor({ timeout: 30000 });
  log("tutorial overlay shown (STEP 1 OF 5) ✓");
  for (let s = 0; s < 4; s++) {
    await page.getByRole("button", { name: "NEXT" }).click();
    await page.waitForTimeout(120);
  }
  await page.getByRole("button", { name: "BEGIN" }).click();
  log("tutorial walkthrough → begin ✓");

  // ── SPIKE phase (served first): SpikeViewer Yes/No, EEG only (no spectrogram)
  await page.getByRole("button", { name: /YES — spike/ }).waitFor({ timeout: 30000 });
  log("spike phase: Yes/No panel ✓");
  await page.waitForTimeout(700);
  if ((await page.locator("canvas").count()) !== 1) {
    throw new Error("spike phase should show ONLY the EEG canvas (no spectrogram)");
  }
  log("spike phase: no spectrogram (1 canvas) ✓");

  // Answer spike (press "1" = Yes) until the IIIC phase begins (Seizure button).
  let reachedIIIC = false;
  for (let i = 0; i < 24 && !reachedIIIC; i++) {
    await page.keyboard.press("1");
    await page.waitForTimeout(450);
    reachedIIIC = (await page.getByRole("button", { name: /Seizure/ }).count()) > 0;
  }
  if (!reachedIIIC) throw new Error("never transitioned spike → IIIC");
  log("transitioned spike → IIIC ✓");

  // ── IIIC phase: spectrogram + EEG, 6-button viewer
  await page.getByText(/Question\s+\d+/).first().waitFor({ timeout: 15000 });
  await page.waitForTimeout(700);
  const canvases = await page.locator("canvas").count();
  if (canvases < 2) throw new Error(`IIIC expected ≥2 canvases (eeg+spec), got ${canvases}`);
  log(`IIIC phase: ${canvases} canvases (eeg + spectrogram) ✓`);

  // cosmetics: inline red scoring banner + bare "Question N" (no "of up to"),
  // confidence readout gone.
  if ((await page.getByText(/Classify the pattern found within the red box/).count()) === 0)
    throw new Error("red scoring banner missing");
  if ((await page.getByText(/of up to/).count()) > 0) throw new Error('counter still shows "of up to"');
  if ((await page.getByText(/Confidence of reaching/).count()) > 0) throw new Error("confidence readout still present");
  log("cosmetics: red banner + bare 'Question N' + no confidence readout ✓");

  // responsive: EEG/spectrogram never overrun the controls at several widths.
  for (const vw of [1500, 1180, 960]) {
    await page.setViewportSize({ width: vw, height: 820 });
    await page.waitForTimeout(250);
    const lay = await page.evaluate(() => {
      const c = [...document.querySelectorAll("canvas")].sort((a, b) => b.width - a.width)[0];
      const ctrl = [...document.querySelectorAll("button, select")]
        .find((e) => /Pan|Montage/.test(e.textContent || "") || e.tagName === "SELECT");
      return { cb: c.getBoundingClientRect().bottom, ct: ctrl ? ctrl.getBoundingClientRect().top : Infinity };
    });
    if (lay.ct < lay.cb - 4) throw new Error(`EEG spills into controls at ${vw}px`);
  }
  await page.setViewportSize({ width: 1500, height: 950 });
  log("responsive: EEG/spectrogram stay above controls at 1500/1180/960px ✓");

  const counterNum = async () =>
    Number((await page.getByText(/Question\s+\d+/).first().innerText()).match(/\d+/)[0]);
  const before = await counterNum();
  for (let i = 0; i < 4; i++) { await page.keyboard.press(String((i % 6) + 1)); await page.waitForTimeout(450); }
  const after = await counterNum();
  if (after <= before) throw new Error(`counter did not advance (${before} → ${after})`);
  log(`answered IIIC via keyboard: question ${before} → ${after} ✓`);

  console.log("UI SMOKE: PASS");
  await browser.close();
  process.exit(0);
} catch (e) {
  console.error("UI SMOKE: FAIL —", e.message);
  if (browser) await browser.close();
  process.exit(1);
}

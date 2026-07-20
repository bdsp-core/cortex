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
let page;

try {
  browser = await chromium.launch({ channel: "chrome", headless: true });
  page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  page.on("pageerror", (e) => console.error("  [pageerror]", e.message));
  await page.goto(baseUrl, { waitUntil: "networkidle" });

  // App opens directly on the auth screen (no landing page).
  // Auth: public email/password signup → verify email → sign in. The signup
  // screen starts on "Sign in"; switch to "Create an account" first.
  const email = `qa+${Date.now()}@example.org`;
  const password = "smoke-pass-123";

  // Capture the dev verification code echoed by the server in dev/CI
  // (CORTEX_EMAIL_EXPOSE_CODE=1 — see ui_smoke.sh).
  let devCode = null;
  page.on("response", async (res) => {
    if (res.url().endsWith("/api/register") && res.request().method() === "POST") {
      try { devCode = (await res.json())?.devCode ?? null; } catch { /* ignore */ }
    }
  });

  await page.getByRole("button", { name: /Create (?:an |new )?account/i }).click();
  await page.getByRole("button", { name: "Continue" }).waitFor({ timeout: 10000 });
  // required-field labels carry a " *", so select by input type / placeholder.
  await page.locator('input[type="email"]').fill(email);
  const pw = page.locator('input[type="password"]');
  await pw.nth(0).fill(password);   // Password
  await pw.nth(1).fill(password);   // Confirm password
  await page.getByPlaceholder("e.g. J. Doe, MD").fill("QA Bot");
  await page.locator("select").first().selectOption({ index: 1 });
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Create account" }).waitFor({ timeout: 10000 });
  await page.getByRole("button", { name: "Create account" }).click();
  log("signup (email/password) ✓");

  // Verify email: the current UI auto-submits when the sixth digit lands.
  await page.getByRole("button", { name: "Verify & continue" }).waitFor({ timeout: 10000 });
  for (let i = 0; i < 40 && devCode == null; i++) await page.waitForTimeout(100);
  if (!devCode || !/^\d{6}$/.test(devCode)) throw new Error(`no 6-digit devCode from /api/register (got ${devCode})`);
  const codeInputs = page.locator('input[inputmode="numeric"][maxlength="1"]');
  for (let i = 0; i < 6; i++) await codeInputs.nth(i).fill(devCode[i]);
  log("verify email (6-digit code) ✓");

  // Success → sign in with the same credentials (no auto-login after verify).
  await page.getByRole("button", { name: "Continue to sign in" }).waitFor({ timeout: 10000 });
  await page.getByRole("button", { name: "Continue to sign in" }).click();
  await page.getByRole("button", { name: /Sign in/ }).waitFor({ timeout: 10000 });
  await page.locator('input[type="email"]').fill(email);
  await page.locator('input[type="password"]').first().fill(password);
  await page.getByRole("button", { name: /Sign in/ }).click();
  log("sign in (post-verify) ✓");

  // Post-verify sign-in now lands on the DASHBOARD shell, not consent. Launch
  // the certification test from the rail CTA to reach the consent screen.
  // The account-created milestone is rendered asynchronously after the shell.
  // Wait briefly for the optional dialog instead of racing the CTA beneath it.
  const milestoneDialog = page.getByRole("dialog", { name: "Milestone reached" });
  if (await milestoneDialog.waitFor({ state: "visible", timeout: 10000 }).then(() => true).catch(() => false)) {
    await milestoneDialog.getByRole("button", { name: "Noted" }).click();
    await milestoneDialog.waitFor({ state: "hidden", timeout: 10000 });
  }
  const testName = /^(?:Re-)?[Tt]ake (?:the )?certification test$/;
  const welcomeDialog = page.getByRole("dialog").filter({ hasText: "Welcome to the CORTEX protocol" });
  const testCta = await welcomeDialog.isVisible().catch(() => false)
    ? welcomeDialog.getByRole("button", { name: testName })
    : page.getByRole("button", { name: testName }).first();
  await testCta.waitFor({ timeout: 10000 });
  await testCta.click();
  log("dashboard → certification test ✓");

  // Consent
  await page.getByRole("button", { name: "I Accept" }).waitFor({ timeout: 10000 });
  await page.getByRole("button", { name: "I Accept" }).click();
  log("consent ✓");

  // Older accounts may still enter the three-page registration wizard here;
  // current signup captures the required profile and proceeds to the tutorial.
  const registrationName = page.getByPlaceholder("e.g. J.D.");
  const hasRegistration = await registrationName.waitFor({ state: "visible", timeout: 3000 })
    .then(() => true)
    .catch(() => false);
  if (hasRegistration) {
    await registrationName.fill("QA Bot");
    await page.locator("select").first().selectOption({ index: 1 });   // expertise
    await page.locator('input[type="text"]:not([placeholder])').first().fill("Test Institution");
    await page.getByRole("button", { name: "Next" }).click();
    await page.getByRole("button", { name: "Next" }).click();
    await page.getByRole("button", { name: "Continue to tutorial" }).click();
    log("registration ✓");
  } else {
    log("registration already captured at signup ✓");
  }

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

  // ── SPIKE phase (served first): Spike/No spike, EEG only (no spectrogram)
  await page.getByRole("button", { name: /^(?:1 · )?Spike$/i }).waitFor({ timeout: 30000 });
  log("spike phase: Spike/No spike panel ✓");
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
  if (page) {
    const text = await page.locator("body").innerText().catch(() => "<body unavailable>");
    console.error("UI SMOKE: page text at failure:\n", text.slice(0, 4000));
  }
  if (browser) await browser.close();
  process.exit(1);
}

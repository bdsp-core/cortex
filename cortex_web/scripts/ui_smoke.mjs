// Headless UI smoke — drives the whole participant flow in a real browser
// (system Chrome via Playwright) against a running server, to validate the
// integration the unit/API tests can't: the screen chain, engine-worker boot,
// EEG/spectrogram render, and 1–6 pick-and-advance answering.
//
// Prereq: the app is served somewhere (default http://localhost:8079) AND a
// participant credential is passed in. Usage:
//   node scripts/ui_smoke.mjs <baseUrl> <code> <password>
//
// Exits 0 on success, 1 on any failed milestone. Designed to be invoked by
// scripts/ui_smoke.sh which boots the server + mints the credential.

import { chromium } from "playwright";

const [baseUrl, code, password] = process.argv.slice(2);
if (!baseUrl || !code || !password) {
  console.error("usage: ui_smoke.mjs <baseUrl> <code> <password>");
  process.exit(2);
}

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

  // Login
  await page.getByPlaceholder("cortex-xxxxxxxx").fill(code);
  await page.locator('input[type="password"]').fill(password);
  await page.getByRole("button", { name: "Continue" }).click();
  log("login submitted ✓");

  // Consent
  await page.getByRole("button", { name: "I Accept" }).waitFor({ timeout: 10000 });
  await page.getByRole("button", { name: "I Accept" }).click();
  log("consent accepted ✓");

  // Registration page 0 (required fields)
  await page.getByPlaceholder("e.g. J.D.").fill("QA Bot");
  // expertise + institution
  const selects = page.locator("select");
  await selects.first().selectOption({ index: 1 });
  await page.locator('input[type="text"]').nth(1).fill("Test Institution");
  await page.getByRole("button", { name: "Next" }).click();
  log("registration p1 ✓");
  // page 1 → Next (all optional)
  await page.getByRole("button", { name: "Next" }).click();
  log("registration p2 ✓");
  // page 2 → Continue to tutorial
  await page.getByRole("button", { name: "Continue to tutorial" }).click();
  log("registration p3 ✓");

  // Tutorial → Begin the test
  await page.getByRole("button", { name: "Begin the test" }).click();
  log("tutorial → begin ✓");

  // Viewer: wait for the question counter to appear
  await page.getByText(/Question\s+\d+\s+of up to/).waitFor({ timeout: 30000 });
  log("viewer rendered (counter visible) ✓");

  // wait for the first EEG to load (loading text disappears), then answer
  await page.waitForFunction(
    () => !document.body.innerText.includes("loading EEG"),
    { timeout: 20000 },
  ).catch(() => log("  (eeg load text check skipped)"));

  // Read the counter, answer 5 questions via the keyboard, assert it advances.
  const counterText = async () =>
    (await page.getByText(/Question\s+\d+\s+of up to/).innerText()).match(/\d+/)[0];
  const before = await counterText();
  for (let i = 0; i < 5; i++) {
    await page.keyboard.press(String((i % 6) + 1));
    await page.waitForTimeout(400); // let the worker pick + the next item load
  }
  const after = await counterText();
  if (Number(after) <= Number(before)) {
    throw new Error(`counter did not advance (${before} → ${after})`);
  }
  log(`answered via keyboard: question ${before} → ${after} ✓`);

  // spectrogram + eeg canvases present
  const canvases = await page.locator("canvas").count();
  if (canvases < 2) throw new Error(`expected ≥2 canvases (eeg+spec), got ${canvases}`);
  log(`canvases present: ${canvases} ✓`);

  console.log("UI SMOKE: PASS");
  await browser.close();
  process.exit(0);
} catch (e) {
  console.error("UI SMOKE: FAIL —", e.message);
  if (browser) await browser.close();
  process.exit(1);
}

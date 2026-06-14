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

// code/password are accepted but unused — auth was removed from the flow.
const [baseUrl] = process.argv.slice(2);
if (!baseUrl) {
  console.error("usage: ui_smoke.mjs <baseUrl>");
  process.exit(2);
}

const log = (m) => console.log(`  ${m}`);
let browser;

try {
  browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  page.on("pageerror", (e) => console.error("  [pageerror]", e.message));

  await page.goto(baseUrl, { waitUntil: "networkidle" });

  // Landing → BEGIN (auth removed → straight to consent)
  await page.getByText("BEGIN ASSESSMENT").click();
  log("landing → begin ✓");

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

  // In-context tutorial overlay (over an example segment): walk the 5
  // coach-marks (NEXT ×4 → BEGIN), which then starts the real test.
  await page.getByText(/STEP 1 OF 5/).waitFor({ timeout: 30000 });
  log("tutorial overlay shown (STEP 1 OF 5) ✓");
  for (let s = 0; s < 4; s++) {
    await page.getByRole("button", { name: "NEXT" }).click();
    await page.waitForTimeout(120);
  }
  await page.getByRole("button", { name: "BEGIN" }).click();
  log("tutorial walkthrough → begin ✓");

  // Viewer: wait for the question counter to appear
  await page.getByText(/Question\s+\d+/).first().waitFor({ timeout: 30000 });
  log("viewer rendered (counter visible) ✓");

  // ── helpers ──────────────────────────────────────────────────────────────
  // EEG traces NON-FLAT: a flat-baseline bug draws one horizontal line/channel,
  // so two x-columns share an identical dark-pixel row set; real traces differ.
  // Polled inside ONE evaluate so the wide-EEG-canvas read is atomic.
  // Non-flat detector: a flat-baseline bug draws horizontal lines at fixed y, so
  // EVERY column shares an identical dark-pixel-row set. Real traces move, so the
  // sets differ between columns. Sample several columns (robust to sparse ones).
  const traceNonFlat = () => page.evaluate(async () => {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const darkRows = (ctx, x, H) => {
      const col = ctx.getImageData(x, 0, 1, H).data, ys = [];
      for (let y = 0; y < H; y++) if (col[y * 4] + col[y * 4 + 1] + col[y * 4 + 2] < 360) ys.push(y);
      return ys.join(",");
    };
    for (let attempt = 0; attempt < 60; attempt++) {
      const c = [...document.querySelectorAll("canvas")].sort((a, b) => b.width - a.width)[0];
      if (c && c.width >= 800) {
        const ctx = c.getContext("2d");
        const cols = [0.30, 0.42, 0.54, 0.66, 0.78].map((f) => darkRows(ctx, Math.floor(c.width * f), c.height));
        const total = cols.reduce((s, r) => s + (r ? r.split(",").length : 0), 0);
        const distinct = new Set(cols.filter((r) => r.length)).size;
        if (total >= 20 && distinct >= 2) return { ok: true, total, distinct };
        if (total >= 20 && distinct === 1) return { ok: false, total, distinct }; // flat baselines
      }
      await sleep(200);
    }
    return { ok: false, total: 0, distinct: 0 };
  });
  const canvasCount = () => page.locator("canvas").count();
  const redPx = () => page.evaluate(() => {
    const c = [...document.querySelectorAll("canvas")].sort((a, b) => b.width - a.width)[0];
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] > 180 && d[i + 1] < 120 && d[i + 2] < 120) n++;
    return n;
  });
  const counterNum = async () =>
    Number((await page.getByText(/Question\s+\d+/).first().innerText()).match(/\d+/)[0]);
  const has = async (re) => (await page.getByText(re).count()) > 0;

  // ── SPIKE PHASE (served first) ─────────────────────────────────────────────
  await page.getByRole("button", { name: /Yes \(spike present\)/ }).waitFor({ timeout: 30000 });
  log("spike phase first: Yes/No answer panel ✓");
  const ts = await traceNonFlat();
  if (!ts.ok) throw new Error("spike EEG traces not rendering");
  log(`spike EEG traces non-flat (${ts.total}px, ${ts.distinct} distinct cols) ✓`);
  if ((await canvasCount()) !== 1) throw new Error("spike must show ONLY the EEG (no spectrogram)");
  log("spike phase: no spectrogram (1 canvas) ✓");
  if (await has(/Classify the pattern found within the red box/)) throw new Error("IIIC banner shown during spike");
  log("spike phase: no IIIC red banner ✓");

  // Answer spike questions (press "1" = Yes) until the IIIC phase begins. The
  // smoke bundle has ~8 spike segs → the spike bank exhausts → IIIC starts.
  let reachedIIIC = false;
  for (let i = 0; i < 24 && !reachedIIIC; i++) {
    await page.keyboard.press("1");
    await page.waitForTimeout(450);
    reachedIIIC = (await page.getByRole("button", { name: /Seizure/ }).count()) > 0;
  }
  if (!reachedIIIC) throw new Error("never transitioned from spike to IIIC phase");
  log("transitioned spike → IIIC ✓");

  // ── IIIC PHASE ─────────────────────────────────────────────────────────────
  await page.getByText(/Classify the pattern found within the red box/).waitFor({ timeout: 15000 });
  const ti = await traceNonFlat();
  if (!ti.ok) throw new Error("IIIC EEG traces not rendering");
  log(`IIIC EEG traces non-flat (${ti.total}px, ${ti.distinct} distinct cols) ✓`);
  if ((await canvasCount()) < 2) throw new Error("IIIC must show EEG + spectrogram");
  log("IIIC phase: spectrogram present (2 canvases) ✓");
  const rb = await redPx();
  if (rb < 50) throw new Error(`IIIC red scoring box not found (${rb} red px)`);
  log(`IIIC red scoring box present (${rb} red px) ✓`);
  if (await has(/Confidence of reaching/)) throw new Error("confidence readout still present");
  if (await has(/of up to/)) throw new Error('counter still shows "of up to …"');
  log("confidence removed + bare 'Question N' counter ✓");

  // Layout guard (responsive): controls never overrun by the EEG/spectrogram.
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
  log("EEG/spectrogram stay above controls at 1500/1180/960px ✓");

  // Answer a few IIIC questions, assert the counter advances.
  const before = await counterNum();
  for (let i = 0; i < 3; i++) { await page.keyboard.press(String((i % 6) + 1)); await page.waitForTimeout(450); }
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

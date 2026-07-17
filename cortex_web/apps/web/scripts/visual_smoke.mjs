// Visual regression smoke — screenshots of the key stable screens, pixel-
// diffed against committed baselines. This is the "silent rewrite" net: a
// change that accidentally restyles or breaks a screen nobody was looking at
// fails here even when every unit test still passes.
//
// Screens (public, deterministic — no per-user data, so baselines are stable):
//   signin-desktop    /            1280×800   the auth landing
//   create-account    /            1280×800   signup form (via the toggle)
//   forgot-password   /            1280×800   reset-request form
//   report            /report      1280×800   public support form
//   signin-mobile     /?mobile=1    390×844   phone auth surface
// The canvas-heavy test viewer is deliberately NOT pixel-diffed (EEG render
// depends on the local bundle); ui_smoke.mjs guards it structurally.
//
//   node scripts/visual_smoke.mjs <baseUrl>            compare against baselines
//   node scripts/visual_smoke.mjs <baseUrl> --update   (re)write baselines
//
// Baselines live in scripts/visual_baselines/ (committed). They are captured
// on the dev box that runs the smoke — regenerate with --update after an
// INTENTIONAL visual change and review the diff in the git client. On
// mismatch a *-diff.png / *-got.png pair lands in /tmp/cortex_visual_smoke
// for inspection. Exits 0 on pass, 1 on any mismatch. Invoked by
// visual_smoke.sh, which boots the same throwaway server as ui_smoke.sh.

import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import pixelmatch from "pixelmatch";
import { PNG } from "pngjs";

const args = process.argv.slice(2);
const update = args.includes("--update");
const baseUrl = args.find((a) => !a.startsWith("--"));
if (!baseUrl) { console.error("usage: visual_smoke.mjs <baseUrl> [--update]"); process.exit(2); }

const BASE_DIR = join(dirname(fileURLToPath(import.meta.url)), "visual_baselines");
const OUT_DIR = "/tmp/cortex_visual_smoke";
// Per-pixel color threshold (pixelmatch) and page-level failure bar: tiny
// antialiasing jitter passes; a moved/restyled element does not.
const PIXEL_THRESHOLD = 0.1;
const MAX_DIFF_RATIO = 0.005;

// Kill animation/caret nondeterminism before every capture.
const FREEZE_CSS = `*, *::before, *::after {
  animation: none !important; transition: none !important;
  caret-color: transparent !important; }`;

const log = (m) => console.log(`  ${m}`);
let browser;
let failures = 0;

async function launch() {
  try { return await chromium.launch({ channel: "chrome", headless: true }); }
  catch { return await chromium.launch({ headless: true }); }   // bundled fallback
}

async function capture(page, name) {
  await page.addStyleTag({ content: FREEZE_CSS });
  await page.waitForTimeout(250);   // let layout + fonts settle
  const shot = await page.screenshot({ fullPage: false });
  const basePath = join(BASE_DIR, `${name}.png`);
  if (update || !existsSync(basePath)) {
    mkdirSync(BASE_DIR, { recursive: true });
    writeFileSync(basePath, shot);
    log(`${name}: baseline ${update ? "updated" : "created"} ✓`);
    return;
  }
  const want = PNG.sync.read(readFileSync(basePath));
  const got = PNG.sync.read(shot);
  if (want.width !== got.width || want.height !== got.height) {
    failures++;
    console.error(`  ✗ ${name}: size ${got.width}×${got.height} vs baseline ${want.width}×${want.height}`);
    return;
  }
  const diff = new PNG({ width: want.width, height: want.height });
  const n = pixelmatch(want.data, got.data, diff.data, want.width, want.height,
                       { threshold: PIXEL_THRESHOLD });
  const ratio = n / (want.width * want.height);
  if (ratio > MAX_DIFF_RATIO) {
    failures++;
    mkdirSync(OUT_DIR, { recursive: true });
    writeFileSync(join(OUT_DIR, `${name}-got.png`), shot);
    writeFileSync(join(OUT_DIR, `${name}-diff.png`), PNG.sync.write(diff));
    console.error(`  ✗ ${name}: ${(ratio * 100).toFixed(2)}% of pixels differ `
      + `(bar ${(MAX_DIFF_RATIO * 100).toFixed(1)}%) — see ${OUT_DIR}/${name}-{got,diff}.png`);
  } else {
    log(`${name}: matches baseline (${(ratio * 100).toFixed(3)}% diff) ✓`);
  }
}

try {
  browser = await launch();

  // ── desktop auth screens ──
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on("pageerror", (e) => console.error("  [pageerror]", e.message));
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator('input[type="email"]').first().waitFor({ timeout: 20000 });
  await capture(page, "signin-desktop");

  await page.getByText("Create new account").click();
  await page.getByRole("button", { name: "Continue" }).waitFor({ timeout: 10000 });
  await capture(page, "create-account");

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByText("Forgot password?").click();
  await page.getByRole("button", { name: "Send reset code" }).waitFor({ timeout: 10000 });
  await capture(page, "forgot-password");

  // In prod Caddy serves index.html for every SPA path; the throwaway
  // uvicorn StaticFiles mount only does that for "/". Shim the document
  // request so /report loads the SPA while the page keeps its URL.
  await page.route("**/report", (route) =>
    route.request().resourceType() === "document"
      ? route.continue({ url: `${baseUrl}/index.html` })
      : route.continue());
  await page.goto(`${baseUrl}/report`, { waitUntil: "networkidle" });
  await page.locator("textarea").first().waitFor({ timeout: 20000 });
  await capture(page, "report");
  await page.close();

  // ── phone auth surface ──
  const mob = await browser.newPage({
    viewport: { width: 390, height: 844 },
    isMobile: true, hasTouch: true, deviceScaleFactor: 1,
  });
  mob.on("pageerror", (e) => console.error("  [pageerror]", e.message));
  await mob.goto(`${baseUrl}/?mobile=1`, { waitUntil: "networkidle" });
  await mob.locator('input[type="email"]').first().waitFor({ timeout: 20000 });
  await capture(mob, "signin-mobile");
  await mob.close();

  if (failures) {
    console.error(`VISUAL SMOKE: FAIL (${failures} screen(s) diverged)`);
    process.exitCode = 1;
  } else {
    console.log(update ? "VISUAL SMOKE: BASELINES WRITTEN" : "VISUAL SMOKE: PASS");
  }
} catch (e) {
  console.error("VISUAL SMOKE: FAIL —", e.message);
  process.exitCode = 1;
} finally {
  await browser?.close().catch(() => {});
}
process.exit(process.exitCode ?? 0);

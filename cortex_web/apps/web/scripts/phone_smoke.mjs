// Phone-viewport smoke — the post-deploy gate for the phone surface. Loads
// the LIVE site under iPhone-class emulation (390×844, coarse pointer) and
// fails on: any page JS error, horizontal overflow, or a missing sign-in
// form. Non-destructive: it only renders public screens (unlike
// ui_smoke.mjs, which drives the full signup flow against a throwaway local
// server), so it is safe to run against production after every deploy.
// The 2026-07-10 sign-in overflow (hero min-width on a 375px screen) is the
// class of regression this exists to catch.
//
//   node scripts/phone_smoke.mjs <baseUrl>
// Exits 0 on success, 1 on any failure. Invoked by deploy/scripts/deploy_app.sh.

import { chromium } from "playwright";

const [baseUrl] = process.argv.slice(2);
if (!baseUrl) { console.error("usage: phone_smoke.mjs <baseUrl>"); process.exit(2); }

const log = (m) => console.log(`  ${m}`);
const fail = (m) => { console.error(`  ✗ ${m}`); process.exitCode = 1; };
let browser;

async function launch() {
  try { return await chromium.launch({ channel: "chrome", headless: true }); }
  catch { return await chromium.launch({ headless: true }); }   // bundled fallback
}

try {
  browser = await launch();
  const ctx = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true, hasTouch: true, deviceScaleFactor: 3,
    userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
      + "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
  });
  const errors = [];
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push(e.message));

  // "/" exercises the real device-detection branch; "/?mobile=1" pins the
  // mobile surface explicitly (the escape-hatch path).
  for (const path of ["/", "/?mobile=1"]) {
    await page.goto(baseUrl + path, { waitUntil: "load" });
    await page.locator('input[type="email"]').first().waitFor({ timeout: 20000 });
    const { scrollW, innerW } = await page.evaluate(() => ({
      scrollW: document.documentElement.scrollWidth,
      innerW: window.innerWidth,
    }));
    if (scrollW > innerW + 1) {
      fail(`${path}: horizontal overflow (${scrollW}px content in ${innerW}px viewport)`);
    } else {
      log(`${path}: sign-in renders, no horizontal overflow (${scrollW}/${innerW}px) ✓`);
    }
  }
  if (errors.length) fail(`page JS errors: ${errors.join(" ;; ")}`);
  else log("no page JS errors ✓");
} catch (e) {
  fail(e instanceof Error ? e.message : String(e));
} finally {
  await browser?.close().catch(() => {});
}
process.exit(process.exitCode ?? 0);

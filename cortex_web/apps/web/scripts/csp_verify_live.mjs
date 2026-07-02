// Live CSP verifier: loads the DEPLOYED site (real prod CSP + real Google
// Identity Services) in Chromium and asserts the strict CSP did not break the
// app. Google blocks automated OAuth *completion*, but LOADING the GIS client
// and RENDERING the button is not OAuth — so a CSP that wrongly blocks
// accounts.google.com/gsi shows up here as (a) a CSP violation and/or (b) the
// button iframe never appearing.
//
//   node scripts/csp_verify_live.mjs [url]   (default https://app.cortexeeg.org)

import { chromium } from "playwright";

const URL = process.argv[2] || "https://app.cortexeeg.org";
const fail = (m) => { console.error("FAIL:", m); process.exitCode = 1; };

const browser = await chromium.launch();
const page = await browser.newPage();
const violations = [];
page.on("console", (msg) => {
  const t = msg.text();
  if (/Content Security Policy|Refused to/i.test(t)) violations.push(t);
});
page.on("pageerror", (e) => violations.push("pageerror: " + e.message));

await page.goto(URL, { waitUntil: "networkidle", timeout: 45000 });
await page.waitForTimeout(2500);   // give GIS time to inject + render

// (1) React mounted under the strict CSP.
const rootChildren = await page.evaluate(() => document.getElementById("root")?.childElementCount ?? 0);
rootChildren > 0 ? console.log(`OK: app rendered (#root has ${rootChildren} children)`)
                 : fail("#root empty — the SPA did not run under the live CSP");

// (2) The GIS client script actually loaded (window.google.accounts) — proves
//     script-src https://accounts.google.com/gsi/client is honoured.
const gis = await page.evaluate(() => !!(window.google && window.google.accounts && window.google.accounts.id));
gis ? console.log("OK: Google Identity Services loaded (window.google.accounts.id present)")
    : fail("GIS did not load — CSP may be blocking accounts.google.com/gsi/client");

// (3) The rendered Google button iframe appeared — proves frame-src for
//     accounts.google.com/gsi is honoured.
const frames = page.frames().map((f) => f.url()).filter((u) => /accounts\.google\.com/.test(u));
frames.length ? console.log(`OK: Google button iframe present (${frames.length})`)
              : fail("no accounts.google.com iframe — frame-src may be blocking the GIS button");

// (4) No CSP violations / page errors at all.
if (violations.length) fail("CSP violations / page errors:\n  " + violations.join("\n  "));
else console.log("OK: zero CSP violations / page errors");

await browser.close();
console.log(process.exitCode ? "LIVE CSP VERIFY: FAILED" : "LIVE CSP VERIFY: PASSED");

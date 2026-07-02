// Headless CSP verifier. Serves the built dist/ under the EXACT production
// Content-Security-Policy and loads it in Chromium, asserting:
//   1. No CSP violations fire for the app's own resources (module bundle,
//      externalised theme-init.js, styles) — i.e. script-src 'self' actually
//      runs the SPA with no 'unsafe-inline'.
//   2. React mounts (#root gets children) — the module script executed.
//   3. theme-init.js ran (data-theme reflects a pre-seeded localStorage value).
//
// The Google Identity Services allowlist can't be exercised offline (needs
// network to accounts.google.com); it's verified live against prod after
// deploy. Google-origin CSP reports here are therefore ignored.
//
//   node scripts/csp_verify.mjs
// Exits 0 on success, 1 on any failed assertion.

import { chromium } from "playwright";
import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const DIST = new URL("../dist/", import.meta.url).pathname;
// Must byte-match the Caddyfile.template CSP.
const CSP = "default-src 'self'; script-src 'self' https://accounts.google.com/gsi/client; " +
  "worker-src 'self'; style-src 'self' 'unsafe-inline' https://accounts.google.com/gsi/style; " +
  "img-src 'self' data:; font-src 'self'; connect-src 'self' https://accounts.google.com/gsi/; " +
  "frame-src https://accounts.google.com/gsi/; frame-ancestors 'none'; base-uri 'self'; " +
  "form-action 'self'; object-src 'none'";

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json",
  ".map": "application/json", ".woff2": "font/woff2" };

const server = http.createServer(async (req, res) => {
  try {
    let p = normalize(decodeURIComponent(req.url.split("?")[0]));
    if (p === "/" || p.endsWith("/")) p = "/index.html";
    let file = join(DIST, p);
    try { await stat(file); } catch { file = join(DIST, "index.html"); }  // SPA fallback
    const body = await readFile(file);
    res.setHeader("Content-Security-Policy", CSP);
    res.setHeader("Content-Type", MIME[extname(file)] || "application/octet-stream");
    res.end(body);
  } catch (e) {
    res.statusCode = 500; res.end(String(e));
  }
});

function isAppOrigin(uri) {
  return !uri || uri.startsWith("http://127.0.0.1") || uri.startsWith("http://localhost")
    || uri === "inline" || uri === "eval";
}

const fail = (m) => { console.error("FAIL:", m); process.exitCode = 1; };

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const base = `http://127.0.0.1:${port}`;

const browser = await chromium.launch();
const page = await browser.newPage();
const violations = [];
page.on("console", (msg) => {
  const t = msg.text();
  if (/Content Security Policy|Refused to/i.test(t)) violations.push(t);
});
// Seed a theme so theme-init.js has something to apply, then load.
await page.addInitScript(() => { try { localStorage.setItem("cortex-theme", "dark"); } catch {} });
await page.goto(base, { waitUntil: "networkidle" });
await page.waitForTimeout(500);

// (1) No CSP violation for the app's own origins.
const appViolations = violations.filter((v) => !/accounts\.google\.com/.test(v));
if (appViolations.length) fail("app-origin CSP violations:\n  " + appViolations.join("\n  "));
else console.log("OK: no app-origin CSP violations");

// (2) React mounted → the module bundle ran under script-src 'self'.
const rootChildren = await page.evaluate(() => document.getElementById("root")?.childElementCount ?? 0);
if (rootChildren > 0) console.log(`OK: React mounted (#root has ${rootChildren} children)`);
else fail("#root is empty — the module bundle did not execute under the CSP");

// (3) theme-init.js executed under script-src 'self'.
const theme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
if (theme === "dark") console.log("OK: theme-init.js ran (data-theme=dark)");
else fail(`theme-init.js did not run (data-theme=${theme})`);

await browser.close();
server.close();
console.log(process.exitCode ? "CSP VERIFY: FAILED" : "CSP VERIFY: PASSED");

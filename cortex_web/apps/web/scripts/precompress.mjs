// Precompress the built SPA (dist/) into .br/.gz siblings so Caddy's
// file_server `precompressed` directive serves them straight off disk —
// max-quality compression paid once at build time instead of per-request
// CPU on the 2-vCPU box. Runs as the last step of `npm run build`; stdlib
// zlib only (no new dependency). Text-ish assets only; images and the EEG
// bundle (not under dist/) are never touched. A sibling is written only
// when it is actually smaller than the original.
import { brotliCompressSync, gzipSync, constants } from "node:zlib";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join, extname } from "node:path";

const ROOT = fileURLToPath(new URL("../dist", import.meta.url));
const COMPRESSIBLE = new Set([".js", ".css", ".html", ".svg", ".json", ".map", ".txt"]);
const MIN_BYTES = 1024; // below this the headers outweigh the savings

let files = 0, rawTotal = 0, brTotal = 0;

function walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) { walk(p); continue; }
    if (!COMPRESSIBLE.has(extname(name)) || st.size < MIN_BYTES) continue;
    const buf = readFileSync(p);
    const br = brotliCompressSync(buf, {
      params: { [constants.BROTLI_PARAM_QUALITY]: 11,
                [constants.BROTLI_PARAM_SIZE_HINT]: buf.length },
    });
    const gz = gzipSync(buf, { level: 9 });
    if (br.length < buf.length) writeFileSync(`${p}.br`, br);
    if (gz.length < buf.length) writeFileSync(`${p}.gz`, gz);
    files += 1;
    rawTotal += buf.length;
    brTotal += Math.min(br.length, buf.length);
  }
}

walk(ROOT);
console.log(`precompress: ${files} files, ${(rawTotal / 1024).toFixed(0)} KB → ` +
            `${(brTotal / 1024).toFixed(0)} KB brotli`);

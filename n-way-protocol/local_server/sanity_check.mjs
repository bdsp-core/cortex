// End-to-end sanity check for the local draw-latent session server.
//
// Builds the bundle, boots the server on a scratch port, runs one simulated
// sharp-expert session to completion through the HTTP API at the production
// engine shape (1200 particles, 30 MH, 420-segment bank draw), and asserts
// the session stops via the unchanged Precision policy with sane counts.
// Then asserts the production-layout capture: the SQLite DB exists in the
// scratch data dir, holds exactly this session with prod-shaped trials +
// diag payloads, and analyze.py renders its report from it.
//
//   node local_server/sanity_check.mjs [--fast]
//
// --fast keeps the same policy but shrinks particles/bank for a quick smoke.

import { spawn, spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 8735;
const BASE = `http://127.0.0.1:${PORT}`;
const FAST = process.argv.includes("--fast");
// Scratch persistence dir so the assertions below see exactly one session
// (the real server default local_server/data/ is left untouched).
const DATA_DIR = path.join(ROOT, ".local-server-dist", "sanity-data");

const failures = [];
function check(label, ok, detail) {
  const status = ok ? "ok " : "FAIL";
  console.log(`  [${status}] ${label}${detail !== undefined ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(label);
}

async function api(pathname, body) {
  const response = await fetch(`${BASE}${pathname}`, body === undefined
    ? {}
    : {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`${pathname} -> ${response.status}: ${payload.error}`);
  }
  return payload;
}

console.log("building bundle...");
const build = spawnSync("bash", ["scripts/local_server.sh", "--build-only"], {
  cwd: ROOT, stdio: "inherit",
});
if (build.status !== 0) {
  console.error("build failed");
  process.exit(1);
}

rmSync(DATA_DIR, { recursive: true, force: true });
const server = spawn("node", [".local-server-dist/local_server.mjs"], {
  cwd: ROOT,
  env: { ...process.env, PORT: String(PORT), LOCAL_SERVER_DATA_DIR: DATA_DIR },
  stdio: "ignore",
});
process.on("exit", () => server.kill());

let health = null;
for (let attempt = 0; attempt < 50 && health === null; attempt += 1) {
  await new Promise((resolve) => setTimeout(resolve, 200));
  health = await api("/healthz").catch(() => null);
}
if (health === null) {
  console.error("server did not become healthy");
  process.exit(1);
}
console.log(`server healthy: bank ${health.bankRows} rows, artifact ${health.artifactId}`);
check("healthz reports 17 atoms", health.atomCount === 17, String(health.atomCount));
check(
  "healthz reports draw_latent aggregation",
  health.responseAggregation === "draw_latent",
);

const startedAt = Date.now();
let status = await api("/session", {
  mode: "simulated",
  reader: { preset: "sharp_expert" },
  seed: 20260731,
  ...(FAST ? { particles: 300, bankSegments: 200, mhSteps: 10 } : {}),
});
const sessionId = status.sessionId;
console.log(`session ${sessionId} started (seed ${status.config.seed}, `
  + `${status.config.particles} particles, ${status.config.bankSegments}-segment draw)`);
check("first item is an IIIC class",
  status.currentItem !== null && status.currentItem.askedK >= 1
  && status.currentItem.askedK <= 6,
  status.currentItem?.askedClass);

let batches = 0;
while (!status.stopped && batches < 16) {
  status = await api(`/session/${sessionId}/autostep`, { n: 50 });
  batches += 1;
  console.log(`  batch ${batches}: ${status.questionsAsked} questions, `
    + `states ${status.policy.selectionStates.join(",")} `
    + `(${((Date.now() - startedAt) / 1000).toFixed(0)}s elapsed)`);
}

const elapsedS = (Date.now() - startedAt) / 1000;
console.log(`session finished in ${elapsedS.toFixed(0)}s wall, `
  + `${(status.timing.totalMs / 1000).toFixed(0)}s engine`);

check("session stopped", status.stopped === true);
check(
  "stop came from the Precision policy",
  status.stopReason === "all_estimated_or_undeterminable",
  String(status.stopReason),
);
check(
  "question count is sane (6*nMin=120 .. 6*cap=360)",
  status.questionsAsked >= 120 && status.questionsAsked <= 360,
  String(status.questionsAsked),
);
const spike = status.domains[0];
check(
  "spike domain terminalized as UNDETERMINABLE_BANK (no spike candidates)",
  spike.status === "UNDETERMINABLE_BANK",
  spike.status,
);
for (const domain of status.domains.slice(1)) {
  const terminal = domain.status === "ESTIMATE_COMPLETE"
    || domain.status === "UNDETERMINABLE_CAP"
    || domain.status === "UNDETERMINABLE_BANK";
  check(
    `${domain.label}: terminal status, n in [20, 60]`,
    terminal && domain.n <= 60
    && (domain.status !== "ESTIMATE_COMPLETE" || domain.n >= 20),
    `${domain.status} n=${domain.n} skill=${domain.skill.mean.toFixed(2)} `
    + `[${domain.skill.ci95.map((x) => x.toFixed(2)).join(", ")}]`,
  );
}
const massSum = status.atomPosterior.mass.reduce((a, b) => a + b, 0);
check("atom posterior sums to 1", Math.abs(massSum - 1) < 1e-9, massSum.toFixed(12));

const { betas, mass, mapBeta, meanBeta } = status.atomPosterior;
const upperMass = mass.filter((_, i) => betas[i] >= 1.0).reduce((a, b) => a + b, 0);
console.log("atom posterior (beta: mass%):");
console.log("  " + betas.map((beta, i) => `${beta.toFixed(2)}:${(mass[i] * 100).toFixed(1)}`).join(" "));
console.log(`  MAP beta ${mapBeta.toFixed(3)}, posterior-mean beta ${meanBeta.toFixed(3)}, `
  + `mass on beta>=1.0: ${(upperMass * 100).toFixed(1)}%`);
// Informational only: on a single session the atom posterior is a noisy,
// lineage-coupled quantity (the R&D campaign's MAP atom lands within
// |log(map/world)| < 0.2 only ~49% of the time at 17 atoms), so the sharp
// expert's shift toward high beta is reported, not gated.
console.log(meanBeta > 1.0
  ? "  note: posterior-mean beta sits above the population prior mean (~0.98), as expected for world beta 1.6"
  : "  note: posterior-mean beta did NOT exceed the population prior mean on this seed; single-session atom identification is noisy by design");

console.log(`nPerTask: ${status.nPerTask.join(",")}`);

// ── production-layout persistence ──────────────────────────────────────
console.log("checking production-layout persistence...");
const dbPath = path.join(DATA_DIR, "local_test.db");
check("sqlite db exists after session stop", existsSync(dbPath), dbPath);

const listing = await api("/sessions");
check("GET /sessions lists exactly the smoke session",
  listing.sessions.length === 1
  && listing.sessions[0].session_id === sessionId
  && listing.sessions[0].status === "complete"
  && listing.sessions[0].n_trials === status.questionsAsked,
  JSON.stringify(listing.sessions[0] ?? null));

const PROBE = `
import json, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.row_factory = sqlite3.Row
sessions = conn.execute("SELECT * FROM sessions").fetchall()
trials = conn.execute("SELECT * FROM trials ORDER BY trial_index").fetchall()
atoms = [len(json.loads(t["diag"]).get("atom_posterior", [])) for t in trials]
row = sessions[0] if sessions else {}
print(json.dumps({
    "sessions": len(sessions),
    "status": row["status"] if sessions else None,
    "stop_reason": row["stop_reason"] if sessions else None,
    "n_questions": row["n_questions"] if sessions else None,
    "participant": row["participant"] if sessions else None,
    "drawn": len(json.loads(row["drawn_seg_ids"])) if sessions else 0,
    "trials": len(trials),
    "atom_min": min(atoms, default=0),
    "atom_max": max(atoms, default=0),
}))
`;
const probeRun = spawnSync("python3", ["-c", PROBE, dbPath], { encoding: "utf8" });
if (probeRun.status !== 0) {
  check("sqlite probe ran", false, probeRun.stderr.trim());
} else {
  const probe = JSON.parse(probeRun.stdout);
  check("sessions row count is 1", probe.sessions === 1, String(probe.sessions));
  check("session row is complete with stop_reason populated",
    probe.status === "complete" && probe.stop_reason === status.stopReason,
    `${probe.status} / ${probe.stop_reason}`);
  check("trials rows == n_questions",
    probe.trials === status.questionsAsked
    && probe.n_questions === status.questionsAsked,
    `trials=${probe.trials} n_questions=${probe.n_questions} api=${status.questionsAsked}`);
  check("every trials.diag parses with 17 atom masses",
    probe.atom_min === 17 && probe.atom_max === 17,
    `min=${probe.atom_min} max=${probe.atom_max}`);
  check("participant recorded as sim:<preset>",
    probe.participant === "sim:sharp_expert", String(probe.participant));
  check("drawn_seg_ids carries the session draw",
    probe.drawn === status.config.bankSegments, String(probe.drawn));
}

console.log("running analyze.py on the smoke DB...");
const analyze = spawnSync(
  "python3", [path.join(ROOT, "local_server", "analyze.py"), dbPath],
  { stdio: ["ignore", "inherit", "inherit"] },
);
check("analyze.py rendered the report", analyze.status === 0,
  `exit ${analyze.status}`);

console.log(failures.length === 0
  ? "SANITY: all checks passed"
  : `SANITY: ${failures.length} check(s) FAILED`);
server.kill();
process.exit(failures.length === 0 ? 0 : 1);

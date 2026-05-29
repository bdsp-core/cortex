// Local headless demonstration: run full adaptive sessions on the real bank
// bundle for raters of different skill and print n_questions + per-task
// verdicts. Run from cortex_web/ with:
//
//   npx tsx engine/demo_session.ts
//
// (tsx handles the project's extensionless imports; plain `node` does not).
// Requires the local bundle to have been generated:
//   python scripts/prepare_web_bundle.py --version v1.1-local
//
// This is the no-UI, no-AWS proof that the browser engine runs a real
// session locally; the same code runs unchanged inside the Web Worker.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { WebCortexSession } from "./session";
import { EngineInputs } from "./types";
import { pResponseYes, signalZ } from "./likelihood";
import { Rng } from "./rng";

const m = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../public/bundle/v1.1-local/manifest.json", import.meta.url)),
    "utf8",
  ),
);
const inputs: EngineInputs = {
  taskCodes: m.taskCodes,
  taskLabels: m.taskLabels,
  taskPatternWords: m.taskPatternWords,
  corrL: m.corrL,
  ellStar: m.ellStar,
  segments: m.segments,
};
const K = inputs.taskCodes.length;
const segById = new Map(inputs.segments.map((s) => [s.segId, s]));

async function run(name: string, trueL: number, seed: number) {
  const simRng = new Rng(seed ^ 0x5eed);
  const t0 = Date.now();
  const session = new WebCortexSession(inputs, `demo-${seed}`, seed, {
    onItem: ({ taskK, segId }) => {
      const s = segById.get(segId)!.sMean[taskK];
      const p = pResponseYes(signalZ(trueL, 0, s));
      const y = simRng.random() < p ? 1 : 0;
      const pick = y === 1 ? taskK : (taskK + 1) % K;
      queueMicrotask(() => session.submitAnswer(pick));
    },
  });
  const r = await session.run();
  const ms = Date.now() - t0;
  const byTask = inputs.taskLabels
    .map((lab, k) => `${lab}:${r.verdicts[k]}`)
    .join("  ");
  console.log(`\n${name}  (true ℓ=${trueL})`);
  console.log(`  n_questions=${r.nQuestions}  stop=${r.stopReason}  (${ms} ms)`);
  console.log(`  ${byTask}`);
}

(async () => {
  console.log(`CORTEX engine — local headless session demo`);
  console.log(`bank: ${inputs.segments.length} IIIC segments, K=${K} tasks\n`);
  await run("Strong examinee ", 1.0, 11);
  await run("Borderline      ", 0.45, 12);
  await run("At-chance        ", 0.0, 13);
})();

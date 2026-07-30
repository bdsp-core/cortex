// Golden parity fixture for the Precision-stopping sidecar.
//
// The committed fixture (tests/fixtures/precision_join_golden.json) carries
// verbatim wire requests plus the responses the sidecar dispatch produced for
// them. This test regenerates the deterministic requests, replays them
// through the exact dispatch the CLI uses, and asserts both sides still
// match the committed bytes; python/tests/test_precision_stop.py replays the
// same requests through the real subprocess + Python client and asserts the
// identical responses. Regenerate with CORTEX_NWAY_REGEN_GOLDEN=1.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { PrecisionSidecar } from "../src/precision_cli";
import type {
  SidecarRequest, SidecarResponse, WireBank, WireState,
} from "../src/precision_cli";

const K = 7;
const N = 48;
const SESSION = "golden:categorical_f1";
const fixturePath = fileURLToPath(
  new URL("./fixtures/precision_join_golden.json", import.meta.url),
);

function identity(): number[][] {
  return Array.from({ length: K }, (_, i) => (
    Array.from({ length: K }, (_, j) => (i === j ? 1 : 0))
  ));
}

function wireState(rng: Rng, rejuvenated: boolean): WireState {
  const t: number[] = [];
  const l: number[] = [];
  for (let i = 0; i < N * K; i += 1) {
    t.push(rng.gaussian());
    l.push(rng.gaussian());
  }
  const raw = Array.from({ length: N }, () => Math.exp(0.4 * rng.gaussian()));
  const total = raw.reduce((a, b) => a + b, 0);
  return {
    N,
    K,
    t,
    l,
    w: raw.map((x) => x / total),
    lastRejuvenation: rejuvenated
      ? {
        qIndex: 2,
        acceptanceRate: 0.41,
        distinctAncestors: 31,
        distinctAncestorFraction: 31 / N,
      }
      : null,
  };
}

// 30 shared segments, one candidate per IIIC domain each; domain 0 has no
// candidates (never served by the harness) and domain 6's signals sit
// entirely in the top band so its lower-band supply starves the content
// floor — both UNDETERMINABLE_BANK paths are exercised.
function wireBank(rng: Rng, removed: number[]): WireBank {
  const bank: WireBank = { askedK: [], segId: [], sMean: [], sSd: [] };
  for (let k = 1; k < K; k += 1) {
    for (let index = 0; index < 30; index += 1) {
      const segId = 100 + index;
      if (removed.includes(segId)) continue;
      bank.askedK.push(k);
      bank.segId.push(segId);
      bank.sMean.push(k === 6 ? 0.8 + 0.4 * rng.random() : -1.4 + 0.1 * index);
      bank.sSd.push(0.05 + 0.1 * rng.random());
    }
  }
  return bank;
}

export function buildRequests(): SidecarRequest[] {
  const rng = new Rng(970431);
  const edges = Array.from({ length: K }, (_, k) => [-0.5 - 0.01 * k, 0.5 + 0.02 * k]);
  const init: SidecarRequest = {
    op: "init",
    sessionId: SESSION,
    corrL: identity(),
    corrT: identity(),
    precisionBandEdges: edges,
  };
  const steps: SidecarRequest[] = [
    {
      op: "evaluate",
      sessionId: SESSION,
      administered: { k: 2, signal: -0.8 },
      state: wireState(rng, false),
      nPerTask: [0, 0, 1, 0, 0, 0, 0],
      bank: wireBank(rng, [100]),
    },
    {
      op: "evaluate",
      sessionId: SESSION,
      administered: { k: 3, signal: 0.1 },
      state: wireState(rng, true),
      nPerTask: [0, 0, 1, 1, 0, 0, 0],
      bank: wireBank(rng, [100, 101]),
    },
    {
      op: "evaluate",
      sessionId: SESSION,
      administered: { k: 2, signal: 0.9 },
      state: wireState(rng, true),
      nPerTask: [0, 0, 2, 1, 0, 0, 0],
      bank: wireBank(rng, [100, 101, 102]),
    },
  ];
  return [{ op: "ping" }, init, ...steps];
}

function roundTrip<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function replay(requests: SidecarRequest[]): SidecarResponse[] {
  const sidecar = new PrecisionSidecar();
  return requests.map((request) => roundTrip(sidecar.handle(request)));
}

describe("precision sidecar golden parity", () => {
  it("committed fixture matches a direct frozen-policy evaluation", () => {
    const built = roundTrip(buildRequests());
    if (process.env.CORTEX_NWAY_REGEN_GOLDEN === "1") {
      writeFileSync(fixturePath, `${JSON.stringify(
        { requests: built, responses: replay(built) }, null, 2,
      )}\n`);
    }
    const committed = JSON.parse(readFileSync(fixturePath, "utf8")) as {
      requests: SidecarRequest[];
      responses: SidecarResponse[];
    };
    expect(committed.requests).toEqual(built);
    expect(replay(committed.requests)).toEqual(committed.responses);
    // The init echo pins the frozen production configuration
    // (session.ts:284-287 defaults).
    expect(committed.responses[1]).toMatchObject({ perDomainCap: 60, nMin: 20 });
    const last = committed.responses[4].result as {
      stop: boolean; selectionStates: string[];
    };
    expect(last.stop).toBe(false);
    expect(last.selectionStates[0]).toBe("UNDETERMINABLE_BANK");
    expect(last.selectionStates[6]).toBe("UNDETERMINABLE_BANK");
    expect(last.selectionStates[1]).toBe("ACTIVE");
  });
});

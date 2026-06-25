// Per-trial engine step, factored out of WebCortexSession.run so it can run
// either INLINE (committed straight onto the live core) or SPECULATIVELY (on a
// clone of the core during the participant's think-time, then adopted when the
// real answer arrives). The answer is binary (Y = 1 iff the rater's 6-way pick
// == the asked task), so precomputing BOTH branches covers every outcome.
//
// Both paths call the SAME advanceCore on an exact clone, so a speculative
// session is BIT-IDENTICAL to the inline one — proven in speculative.test.ts.
// Speculation only changes WHEN the work runs (idle think-time vs after the
// answer), never WHAT it produces.

import { EngineInputs, ParticleState, TrialDiag } from "./types";
import { update, ess, resampleAndRejuvenate, posteriorMeans, cloneState } from "./particles";
import { aurocSummary } from "./auroc";
import { BankArrays, chooseItem, chooseFirstItem, Chosen } from "./choose_item";
import { AD6Policy, VERDICT } from "./policy";
import { Rng } from "./rng";

// The full MUTABLE per-trial session state — everything the next item's
// selection depends on. Append-only OUTPUTS (trials/traj/served) live on the
// session and are committed after a branch is adopted, so they are NOT part of
// the branchable core.
export interface SessionCore {
  state: ParticleState;
  rng: Rng;
  policy: AD6Policy;
  remaining: Set<number>;
  nPerTask: number[];
  cappedTasks: Set<number>;
  lastVerdicts: string[];
  lastTaskK: number;
  streakCount: number;
}

// Engine constants + per-session derived values, passed in so this module never
// imports session.ts (avoids an import cycle).
export interface AdvanceParams {
  K: number;
  nParticles: number;
  perDomainCap: number;
  nMhSteps: number;
  essThresholdFrac: number;
  proposalScale: number;
  firstItemTopN: number;
  maxConsecutiveSameDomain: number;
  k7Spike: boolean;
  spikeIdx: number;
}

export interface AdvanceResult {
  core: SessionCore;          // the (mutated) core — for spec, the clone to adopt
  diag: TrialDiag;
  trajSnapshot: { t: Float64Array; l: Float64Array; w: Float64Array };
  servedSegId: number;
  rejuv: boolean;
  nextChosen: Chosen;         // segId === -1 when the active bank is exhausted
  done: boolean;              // every task resolved or capped → stop
  stopReason: string;         // set when done; "" otherwise
}

const NO_ITEM: Chosen = { k: 0, s: 0, sSd: 0, segId: -1, loss: Infinity };

// Deep clone of the branchable core. state/rng/policy cloned; the immutable
// prior is shared; Sets/arrays copied.
export function cloneCore(c: SessionCore): SessionCore {
  return {
    state: cloneState(c.state),
    rng: c.rng.clone(),
    policy: c.policy.clone(),
    remaining: new Set(c.remaining),
    nPerTask: c.nPerTask.slice(),
    cappedTasks: new Set(c.cappedTasks),
    lastVerdicts: c.lastVerdicts.slice(),
    lastTaskK: c.lastTaskK,
    streakCount: c.streakCount,
  };
}

// Per-task candidate arrays over the remaining (unserved) bank. A segment only
// contributes to its applicableTaskIdx (IIIC → tasks 1..6, spike → task 0);
// pre-K=7 bundles omit it and fall back to "all K tasks".
function bankArrays(inputs: EngineInputs, remaining: Set<number>): BankArrays {
  const K = inputs.taskCodes.length;
  const sMean: number[][] = Array.from({ length: K }, () => []);
  const sSd: number[][] = Array.from({ length: K }, () => []);
  const segId: number[][] = Array.from({ length: K }, () => []);
  for (const seg of inputs.segments) {
    if (!remaining.has(seg.segId)) continue;
    const applicable = seg.applicableTaskIdx;
    if (applicable) {
      for (const k of applicable) {
        sMean[k].push(seg.sMean[k]);
        sSd[k].push(seg.sSd[k]);
        segId[k].push(seg.segId);
      }
    } else {
      for (let k = 0; k < K; k++) {
        sMean[k].push(seg.sMean[k]);
        sSd[k].push(seg.sSd[k]);
        segId[k].push(seg.segId);
      }
    }
  }
  return { sMean, sSd, segId };
}

// Select the item for `trialIndex` from the current core. trialIndex 0 uses the
// top-N uniform opener (consumes core.rng via chooseFirstItem); i≥1 uses the
// deterministic A-optimal chooseItem. Identical hard/phase/variety exclusions +
// defensive fallbacks as the original inline loop.
export function chooseNext(
  core: SessionCore, inputs: EngineInputs, params: AdvanceParams, trialIndex: number,
): Chosen {
  const { K, k7Spike, spikeIdx, maxConsecutiveSameDomain, firstItemTopN } = params;
  const bank = bankArrays(inputs, core.remaining);

  // Hard exclusions — survive ALL fallbacks. RESOLVED tasks (verdict locked) +
  // CAPPED tasks (PENDING but spent their budget → REFER) are never reselected.
  const hardExcluded = new Set<number>();
  for (let k = 0; k < K; k++) {
    if (core.lastVerdicts[k] !== VERDICT.PENDING || core.cappedTasks.has(k)) hardExcluded.add(k);
  }
  // Phase exclusion (spike-first sectioning), layered on the hard floor.
  const phaseExcluded = new Set<number>(hardExcluded);
  if (k7Spike) {
    const spikePending = core.lastVerdicts[spikeIdx] === VERDICT.PENDING;
    const spikeBankNonEmpty = bank.sMean[spikeIdx]?.length > 0;
    if (spikePending && spikeBankNonEmpty) {
      for (let k = 0; k < K; k++) if (k !== spikeIdx) phaseExcluded.add(k);
    } else {
      phaseExcluded.add(spikeIdx);
    }
  }
  // Variety cap composed with phase + hard.
  const excluded = new Set(phaseExcluded);
  if (trialIndex > 0 && core.streakCount >= maxConsecutiveSameDomain
        && !excluded.has(core.lastTaskK)) {
    const otherHasBank = bank.sMean.some(
      (arr, k) => k !== core.lastTaskK && !excluded.has(k) && arr.length > 0,
    );
    if (otherHasBank) excluded.add(core.lastTaskK);
  }

  const pick = (ex: Set<number>): Chosen =>
    trialIndex === 0
      ? chooseFirstItem(core.state, bank, firstItemTopN, core.rng, ex)
      : chooseItem(core.state, bank, ex);
  // Fallbacks: drop the variety cap, then phase — but NEVER revive a
  // resolved/capped task (hardExcluded is the floor).
  let chosen = pick(excluded);
  if (chosen.segId === -1) chosen = pick(phaseExcluded);
  if (chosen.segId === -1) chosen = pick(hardExcluded);
  return chosen;
}

// Process one answer on `core` (MUTATES it): reweight → maybe resample +
// rejuvenate → bookkeeping → AD6 evaluate + per-domain cap → diag → choose the
// next item (skipped when the session is done). Returns the per-trial outputs.
export function advanceCore(
  core: SessionCore, inputs: EngineInputs, chosen: Chosen, y: 0 | 1,
  params: AdvanceParams, trialIndex: number,
): AdvanceResult {
  const { state, rng, policy } = core;

  update(state, chosen.k, chosen.s, y, chosen.sSd);
  let rejuv = false;
  if (ess(state.w) < params.essThresholdFrac * params.nParticles) {
    resampleAndRejuvenate(state, rng, params.nMhSteps, params.proposalScale);
    rejuv = true;
  }

  core.remaining.delete(chosen.segId);
  core.nPerTask[chosen.k] += 1;
  // Snapshot the (post-update) particle cloud for the visualization videos.
  const trajSnapshot = { t: state.t.slice(), l: state.l.slice(), w: state.w.slice() };
  // Variety-cap streak.
  if (chosen.k === core.lastTaskK) core.streakCount += 1;
  else { core.lastTaskK = chosen.k; core.streakCount = 1; }

  const res = policy.evaluate(state, core.nPerTask);
  core.lastVerdicts = res.verdicts;
  for (let k = 0; k < params.K; k++) {
    if (res.verdicts[k] === VERDICT.PENDING && core.nPerTask[k] >= params.perDomainCap) {
      core.cappedTasks.add(k);
    }
  }
  const { tMean, lMean } = posteriorMeans(state);
  const diag: TrialDiag = {
    trialIndex,
    taskK: chosen.k,
    segId: chosen.segId,
    s: chosen.s,
    sSd: chosen.sSd,
    y,
    ess: res.ess,
    rejuv,
    pi: res.pi,
    mcse: res.mcse,
    R: res.R,
    verdicts: res.verdicts,
    nPerTask: core.nPerTask.slice(),
    tMean,
    lMean,
    aurocHw: aurocSummary(state.l, state.w, state.N, params.K).hw,
  };

  // Adaptive stop: every task resolved OR capped-out (capped → REFER at
  // finalize). Subsumes AD6's all-resolved stop.
  const done = res.verdicts.every((v, k) => v !== VERDICT.PENDING || core.cappedTasks.has(k));
  let stopReason = "";
  let nextChosen = NO_ITEM;
  if (done) {
    stopReason = res.stop ? "all_resolved" : "resolved_or_referred";
  } else {
    nextChosen = chooseNext(core, inputs, params, trialIndex + 1);
  }
  return { core, diag, trajSnapshot, servedSegId: chosen.segId, rejuv, nextChosen, done, stopReason };
}

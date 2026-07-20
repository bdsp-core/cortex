// Per-trial engine step, factored out of WebCortexSession.run so it can run
// either INLINE (committed straight onto the live core) or SPECULATIVELY (on a
// clone of the core during the participant's think-time, then adopted when the
// real answer arrives). IIIC branches carry the native six-way task-axis pick;
// spike branches remain binary (task 0 vs the K sentinel).
//
// Both paths call the SAME advanceCore on an exact clone, so a speculative
// session is BIT-IDENTICAL to the inline one — proven in speculative.test.ts.
// Speculation only changes WHEN the work runs (idle think-time vs after the
// answer), never WHAT it produces.

import { ComputeEngineInputs, EngineStepTiming, ParticleState, TrialDiag } from "./types";
import { updateObservation, ess, resampleAndRejuvenate, posteriorMeans, cloneState } from "./particles";
import { aurocSummary } from "./auroc";
import {
  BankArrays, chooseItem, chooseFirstItem, Chosen, sortBankBySignalStable,
} from "./choose_item";
import { makeResponseObservation } from "./nway_likelihood";
import { isNWaySession } from "./nway_profile";
import { chooseFirstNWayItem, chooseNWayItem } from "./nway_selector";
import { EngineTerminationPolicy, VERDICT } from "./policy";
import {
  PRECISION_STATUS, PrecisionDiagnostics, PrecisionPolicy,
} from "./precision_policy";
import { Rng } from "./rng";

// The full MUTABLE per-trial session state — everything the next item's
// selection depends on. Append-only OUTPUTS (trials/traj/served) live on the
// session and are committed after a branch is adopted, so they are NOT part of
// the branchable core.
export interface SessionCore {
  state: ParticleState;
  rng: Rng;
  policy: EngineTerminationPolicy;
  remaining: Set<number>;
  nPerTask: number[];
  cappedTasks: Set<number>;
  lastOutcomes: string[];
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
  nSubsample?: number;
  uncertaintyAwareSubsample?: boolean;
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
  timing: EngineStepTiming;   // internal wall-clock attribution; never policy input
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
    lastOutcomes: c.lastOutcomes.slice(),
    lastTaskK: c.lastTaskK,
    streakCount: c.streakCount,
  };
}

// Per-task candidate arrays over the remaining (unserved) bank. A segment only
// contributes to its applicableTaskIdx (IIIC → tasks 1..6, spike → task 0);
// pre-K=7 bundles omit it and fall back to "all K tasks".
function bankArrays(inputs: ComputeEngineInputs, remaining?: ReadonlySet<number>): BankArrays {
  const K = inputs.taskCodes.length;
  const sMean: number[][] = Array.from({ length: K }, () => []);
  const sSd: number[][] = Array.from({ length: K }, () => []);
  const segId: number[][] = Array.from({ length: K }, () => []);
  const segment = Array.from({ length: K }, () => [] as typeof inputs.segments);
  for (const seg of inputs.segments) {
    if (remaining && !remaining.has(seg.segId)) continue;
    const applicable = seg.applicableTaskIdx;
    if (applicable) {
      for (const k of applicable) {
        sMean[k].push(seg.sMean[k]);
        sSd[k].push(seg.sSd[k]);
        segId[k].push(seg.segId);
        segment[k].push(seg);
      }
    } else {
      for (let k = 0; k < K; k++) {
        sMean[k].push(seg.sMean[k]);
        sSd[k].push(seg.sSd[k]);
        segId[k].push(seg.segId);
        segment[k].push(seg);
      }
    }
  }
  return { sMean, sSd, segId, segment };
}

// Precision always uses the same full, manifest-ordered bank for a session.
// Stable-sorting ~35k candidates into seven domain arrays on every branch of
// every answer was pure repeated work. Cache that immutable expansion once per
// EngineInputs object, then retain only currently available ids. Filtering a
// stable full sort is equivalent to stable-sorting the filtered manifest.
const precisionSortedBankCache = new WeakMap<ComputeEngineInputs, BankArrays>();

function precisionSortedBank(inputs: ComputeEngineInputs): BankArrays {
  let bank = precisionSortedBankCache.get(inputs);
  if (!bank) {
    bank = sortBankBySignalStable(bankArrays(inputs));
    precisionSortedBankCache.set(inputs, bank);
  }
  return bank;
}

export function precisionRemainingBank(
  inputs: ComputeEngineInputs,
  remaining: ReadonlySet<number>,
  additionallyRemove?: number,
): BankArrays {
  const full = precisionSortedBank(inputs);
  const out: BankArrays = { sMean: [], sSd: [], segId: [], segment: [] };
  for (let k = 0; k < full.sMean.length; k++) {
    const means: number[] = [];
    const sds: number[] = [];
    const ids: number[] = [];
    const segments = [] as typeof inputs.segments;
    for (let i = 0; i < full.segId[k].length; i++) {
      const id = full.segId[k][i];
      if (id === additionallyRemove || !remaining.has(id)) continue;
      means.push(full.sMean[k][i]);
      sds.push(full.sSd[k][i]);
      ids.push(id);
      segments.push(full.segment![k][i]);
    }
    out.sMean.push(means);
    out.sSd.push(sds);
    out.segId.push(ids);
    out.segment!.push(segments);
  }
  return out;
}

// Select the item for `trialIndex` from the current core. trialIndex 0 uses the
// top-N uniform opener (consumes core.rng via chooseFirstItem); i≥1 uses the
// deterministic A-optimal chooseItem. Identical hard/phase/variety exclusions +
// defensive fallbacks as the original inline loop.
export function chooseNext(
  core: SessionCore, inputs: ComputeEngineInputs, params: AdvanceParams, trialIndex: number,
  preparedPrecisionBank?: BankArrays,
): Chosen {
  const { K, k7Spike, spikeIdx, maxConsecutiveSameDomain, firstItemTopN } = params;
  const precisionPolicy = core.policy instanceof PrecisionPolicy ? core.policy : null;
  const nway = isNWaySession(inputs);
  let bank = precisionPolicy
    ? (preparedPrecisionBank ?? precisionRemainingBank(inputs, core.remaining))
    : bankArrays(inputs, core.remaining);
  if (precisionPolicy) {
    bank = precisionPolicy.prepareCandidates(bank, core.nPerTask);
    core.lastOutcomes = precisionPolicy.domainStatuses;

    // Precision statuses are fresh-derived, not AD6 verdict locks. Match the
    // controller's exact active-domain/variety semantics: phase A remains
    // spike-only; in phase B, after five consecutive questions, another
    // ACTIVE domain is preferred. If the last domain is the only ACTIVE one,
    // an ESTIMATE_COMPLETE IIIC domain may absorb the variety question (and
    // can consequently reopen). Sticky terminal domains and domains at cap
    // are never revived.
    const statuses = precisionPolicy.domainStatuses;
    let allowed: number[];
    const spikeActive = k7Spike
      && statuses[spikeIdx] === PRECISION_STATUS.ACTIVE
      && core.nPerTask[spikeIdx] < precisionPolicy.perDomainCap
      && bank.sMean[spikeIdx].length > 0;
    if (spikeActive) {
      allowed = [spikeIdx];
    } else {
      const active = Array.from({ length: K }, (_, k) => k).filter(
        (k) => statuses[k] === PRECISION_STATUS.ACTIVE
          && core.nPerTask[k] < precisionPolicy.perDomainCap
          && bank.sMean[k].length > 0,
      );
      allowed = active;
      if (trialIndex > 0 && core.streakCount >= maxConsecutiveSameDomain
          && active.includes(core.lastTaskK)) {
        const others = active.filter((k) => k !== core.lastTaskK);
        if (others.length) {
          allowed = others;
        } else {
          const variety = Array.from({ length: K }, (_, k) => k).filter(
            (k) => k >= 1 && k !== core.lastTaskK
              && statuses[k] === PRECISION_STATUS.ESTIMATE_COMPLETE
              && core.nPerTask[k] < precisionPolicy.perDomainCap
              && bank.sMean[k].length > 0,
          );
          if (variety.length) allowed = variety;
        }
      }
    }
    const excluded = new Set<number>();
    for (let k = 0; k < K; k++) if (!allowed.includes(k)) excluded.add(k);
    return trialIndex === 0
      ? (nway ? chooseFirstNWayItem(
          core.state, inputs, bank, firstItemTopN, core.rng, excluded,
        ) : chooseFirstItem(core.state, bank, firstItemTopN, core.rng, excluded, {
          nSubsample: params.nSubsample,
          uncertaintyAware: params.uncertaintyAwareSubsample,
        }))
      : (nway ? chooseNWayItem(core.state, inputs, bank, excluded) : chooseItem(core.state, bank, excluded, {
          nSubsample: params.nSubsample,
          uncertaintyAware: params.uncertaintyAwareSubsample,
        }));
  }

  // Hard exclusions — survive ALL fallbacks. RESOLVED tasks (verdict locked) +
  // CAPPED tasks (PENDING but spent their budget → REFER) are never reselected.
  const hardExcluded = new Set<number>();
  for (let k = 0; k < K; k++) {
    if (core.lastOutcomes[k] !== core.policy.activeLabel || core.cappedTasks.has(k)) {
      hardExcluded.add(k);
    }
  }
  // Phase exclusion (spike-first sectioning), layered on the hard floor.
  const phaseExcluded = new Set<number>(hardExcluded);
  if (k7Spike) {
    const spikePending = core.lastOutcomes[spikeIdx] === core.policy.activeLabel;
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
      ? (nway ? chooseFirstNWayItem(
          core.state, inputs, bank, firstItemTopN, core.rng, ex,
        ) : chooseFirstItem(core.state, bank, firstItemTopN, core.rng, ex, {
          nSubsample: params.nSubsample,
          uncertaintyAware: params.uncertaintyAwareSubsample,
        }))
      : (nway ? chooseNWayItem(core.state, inputs, bank, ex) : chooseItem(core.state, bank, ex, {
          nSubsample: params.nSubsample,
          uncertaintyAware: params.uncertaintyAwareSubsample,
        }));
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
  core: SessionCore, inputs: ComputeEngineInputs, chosen: Chosen, rawPick: number,
  params: AdvanceParams, trialIndex: number,
  preparedPrecisionBank?: BankArrays,
): AdvanceResult {
  const startedAt = performance.now();
  const { state, rng, policy } = core;

  const segment = chosen.segment
    ?? inputs.segments.find((candidate) => candidate.segId === chosen.segId);
  if (!segment) throw new Error(`chosen segment ${chosen.segId} is unavailable`);
  const response = isNWaySession(inputs)
    ? makeResponseObservation(
        chosen.k, segment, rawPick,
        inputs.taskClasses?.[chosen.k] ?? "iiic",
      )
    : {
        kind: "binary" as const, k: chosen.k, s: chosen.s, sSd: chosen.sSd,
        y: rawPick === chosen.k ? 1 as const : 0 as const, rawPick,
      };
  updateObservation(state, response);
  const y: 0 | 1 = rawPick === chosen.k ? 1 : 0;
  const updatedAt = performance.now();
  let rejuv = false;
  if (ess(state.w) < params.essThresholdFrac * params.nParticles) {
    resampleAndRejuvenate(
      state, rng, params.nMhSteps, params.proposalScale, trialIndex,
    );
    rejuv = true;
  }
  const rejuvenatedAt = performance.now();

  core.remaining.delete(chosen.segId);
  core.nPerTask[chosen.k] += 1;
  if (policy instanceof PrecisionPolicy) policy.recordAdministered(chosen.k, chosen.s);
  // Snapshot the (post-update) particle cloud for the visualization videos.
  const trajSnapshot = { t: state.t.slice(), l: state.l.slice(), w: state.w.slice() };
  // Variety-cap streak.
  if (chosen.k === core.lastTaskK) core.streakCount += 1;
  else { core.lastTaskK = chosen.k; core.streakCount = 1; }
  const bookkeepingAt = performance.now();

  let telemetry: Record<string, unknown> | undefined;
  if (policy instanceof PrecisionPolicy) {
    const rawBank = preparedPrecisionBank
      ?? precisionRemainingBank(inputs, core.remaining);
    telemetry = policy.bankTelemetry(rawBank, state.lastRejuvenation);
  }
  const res = policy.evaluate(state, core.nPerTask, telemetry);
  const policyAt = performance.now();
  core.lastOutcomes = res.selectionStates;
  if (policy.name === "ad6") {
    for (let k = 0; k < params.K; k++) {
      if (res.verdicts[k] === VERDICT.PENDING && core.nPerTask[k] >= params.perDomainCap) {
        core.cappedTasks.add(k);
      }
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
    pick: rawPick,
    responseKind: response.kind,
    matchedAskedTask: y === 1,
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
    terminationPolicy: res.policyName,
    ...(res.domainStatuses ? { domainStatuses: res.domainStatuses } : {}),
    ...(res.determinations ? { determinations: res.determinations } : {}),
    ...(res.terminalReasons ? { terminalReasons: res.terminalReasons } : {}),
    ...(res.streakCounts ? { precisionStreakCounts: res.streakCounts } : {}),
    ...(state.lastRejuvenation ? { lastRejuvenation: { ...state.lastRejuvenation } } : {}),
  };
  const pd = res.diagnostics as PrecisionDiagnostics | undefined;
  if (pd) {
    diag.skillIntervals = pd.skillIntervals.map((x) => [...x] as [number, number]);
    diag.biasIntervals = pd.biasIntervals.map((x) => [...x] as [number, number]);
    diag.skillPointCenteredRadius = pd.skillPointCenteredRadius.slice();
    diag.skillPointCenteredRadiusMcse = pd.skillPointCenteredRadiusMcse.slice();
    diag.guardedPrecisionStatistic = pd.guardedPrecisionStatistic.slice();
    diag.skillTolerance = pd.skillTolerance.slice();
  }
  const diagnosticsAt = performance.now();

  // Adaptive stop: every task resolved OR capped-out (capped → REFER at
  // finalize). Subsumes AD6's all-resolved stop.
  const done = policy.name === "precision_v1"
    ? res.stop
    : res.verdicts.every((v, k) => v !== VERDICT.PENDING || core.cappedTasks.has(k));
  let stopReason = "";
  let nextChosen = NO_ITEM;
  if (done) {
    stopReason = policy.name === "precision_v1"
      ? res.stopReason
      : res.stop ? "all_resolved" : "resolved_or_referred";
  } else {
    nextChosen = chooseNext(
      core, inputs, params, trialIndex + 1, preparedPrecisionBank,
    );
  }
  const finishedAt = performance.now();
  return {
    core, diag, trajSnapshot, servedSegId: chosen.segId, rejuv, nextChosen, done, stopReason,
    timing: {
      kind: "engine_step",
      trialIndex,
      y,
      pick: rawPick,
      rejuvenated: rejuv,
      bankPreparationMs: 0,
      updateMs: updatedAt - startedAt,
      rejuvenationMs: rejuvenatedAt - updatedAt,
      bookkeepingMs: bookkeepingAt - rejuvenatedAt,
      policyMs: policyAt - bookkeepingAt,
      diagnosticsMs: diagnosticsAt - policyAt,
      selectionMs: finishedAt - diagnosticsAt,
      totalMs: finishedAt - startedAt,
      executionMode: "serial",
      speculative: false,
      requiredBranchReadyAtAnswer: false,
    },
  };
}

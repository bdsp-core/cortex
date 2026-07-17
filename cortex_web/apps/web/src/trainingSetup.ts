// Training-session construction, split out of App.tsx so the trainer engine
// (belief filters, candidate bank, label schedule) rides in the lazy training
// chunk instead of the first-paint bundle — only participants who press
// "Start training" download it. App.tsx dynamic-imports this module together
// with the TrainingRunner component; everything trainer-only must be imported
// from HERE (a static trainer import in App.tsx would pull the engine back
// into the main chunk).

import { Bundle, SessionBank } from "./bundle";
import type { RegimenPlan } from "./api";
import { buildTrainerBank, cutScores, seedClouds, seedCloudsFromPrior } from "./trainerBank";
import { ArrayBank, TrainerSession, buildFilters } from "../trainer/session";
import { ServerTrainerSession } from "../trainer/serverSession";
import { seedFromString } from "../trainer/label_schedule";
import type { FilterParams } from "../trainer/filter";
import type { TrainerSessionLike } from "./trainingController";

// The trainer's assumed learner dynamics (anchored EXTSET rates; matches the
// Python production defaults). sigmaInf is per-task, set in buildTrainingState.
const TRAINER_PARAMS: FilterParams = {
  alphaT: 0.097, alphaSigma: 0.047, sigmaInf: 0.4,
  qT: 0.05, qSigma: 0.02, rho: 0.5, rule: "soft",
};

export interface TrainingState {
  session: TrainerSessionLike;
  trainingId: string;
  labels: string[];
  bundle: Bundle;
  // R5: seed-time attainability, prepared for display (engine mode only).
  attainability?: AttainabilityTier[];
}

export interface AttainabilityTier {
  label: string;
  tier: "far" | "mid" | "near";
}

/** Qualitative framing of the engine's per-domain trainability report
 *  (probability the skill ceiling clears the certification bar). Pure. */
export function attainabilityTiers(
  att: Record<string, number> | undefined,
  taskCodes: string[],
  taskLabels: string[],
): AttainabilityTier[] {
  if (!att) return [];
  return taskCodes
    .map((code, k) => ({ code, k }))
    .filter(({ code }) => att[code] !== undefined)
    .map(({ code, k }) => ({
      label: taskLabels[k] ?? code,
      tier: (att[code] < 0.3 ? "far" : att[code] < 0.7 ? "mid" : "near") as
        AttainabilityTier["tier"],
    }));
}

/** Assemble a ready-to-run training sitting from the server pieces: the
 *  regimen (measured cert prior), a spacing-aware balanced draw (the same
 *  session bank the exam uses, so every segment is renderable), and the
 *  server-issued trainingId (which also seeds the label schedule). */
export function buildTrainingState(
  plan: RegimenPlan | null | undefined,
  bank: SessionBank,
  trainingId: string,
): TrainingState {
  const b = Bundle.fromSessionBank(bank);
  const inputs = b.inputs;
  const ellStar = inputs.ellStar ?? inputs.taskCodes.map(() => 0.3);
  const { ellStars, sigmaStars, sigmaInf } = cutScores(ellStar);
  // Real-skill handoff: seed each task's belief from the learner's measured
  // cert posterior (variance-inflated); fall back to a generic prior only if
  // the regimen carries no posterior (legacy result).
  const clouds = plan?.prior && plan.prior.length
    ? seedCloudsFromPrior(plan.prior, ellStars.length, 200, 1)
    : seedClouds(ellStars.map(() => 0.0), 200, 1);
  const filters = buildFilters(clouds, TRAINER_PARAMS, sigmaInf, ellStars, { seed: 7, useMixture: false });
  // Label-schedule randomization (docs/LABEL_SCHEDULE_PECR.md): kills the
  // deterministic pos/neg question pattern. The schedule seed is derived
  // from the server-issued trainingId (§6.4) — unique per session, and the
  // served label sequence is reproducible from the session record. The
  // filter seed (7) is unchanged: the belief engine is untouched.
  const session = new TrainerSession(
    filters, ellStars, sigmaStars,
    new ArrayBank(buildTrainerBank({ segments: inputs.segments })),
    { seed: 7, labelSchedule: 'randomized',
      scheduleSeed: seedFromString(trainingId) });
  return { session, trainingId, labels: inputs.taskLabels, bundle: b };
}

/** Engine-mode assembly (Phase L3): the server-side learning engine drives
 *  the sitting; the client keeps rendering + the checkpoint ledger. The
 *  drawn pool's segIds go up so the engine only serves media this client
 *  can load; the regimen deck (weak tasks) becomes the engine's restrict
 *  set. The belief is seeded SERVER-side by replaying the participant's
 *  latest certification sitting (handoff contract §2a) — the regimen
 *  `prior`/`testStream` payloads are deliberately NOT consumed here
 *  (posterior XOR replay, never both). */
export async function buildServerTrainingState(
  plan: RegimenPlan | null | undefined,
  bank: SessionBank,
  trainingId: string,
): Promise<TrainingState> {
  const b = Bundle.fromSessionBank(bank);
  const segIds = b.inputs.segments.map((s) => s.segId);
  const restrict = plan?.deck?.length
    ? plan.deck.map((d) => d.taskK)
    : undefined;
  const ellStar = b.inputs.ellStar ?? b.inputs.taskCodes.map(() => 0.3);
  const session = await ServerTrainerSession.start(
    trainingId, segIds, restrict, cutScores(ellStar).ellStars);
  return {
    session, trainingId, labels: b.inputs.taskLabels, bundle: b,
    attainability: attainabilityTiers(
      session.attainability, b.inputs.taskCodes, b.inputs.taskLabels),
  };
}

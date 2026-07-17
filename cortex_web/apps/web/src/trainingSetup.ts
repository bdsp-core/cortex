// Training-session construction, split out of App.tsx so the trainer engine
// (belief filters, candidate bank, label schedule) rides in the lazy training
// chunk instead of the first-paint bundle — only participants who press
// "Start training" download it. App.tsx dynamic-imports this module together
// with the TrainingRunner component; everything trainer-only must be imported
// from HERE (a static trainer import in App.tsx would pull the engine back
// into the main chunk).

import { Bundle, SessionBank } from "./bundle";
import type { RegimenPlan } from "./api";
import { ServerTrainerSession } from "../trainer/serverSession";
import type { TrainerSessionLike } from "./trainingController";

// The incumbent client-side trainer was REMOVED 2026-07-17 (user decision:
// no fallback). The server-side learning engine is the sole trainer; this
// module only assembles the server-driven sitting.

// Mastery targets for the reveal screen, derived from the manifest's ℓ*
// (formerly trainerBank.cutScores — retained piece of the removed module).
function ellStarsOf(ellStar: number[]): number[] {
  return ellStar.slice();
}

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
    trainingId, segIds, restrict, ellStarsOf(ellStar));
  return {
    session, trainingId, labels: b.inputs.taskLabels, bundle: b,
    attainability: attainabilityTiers(
      session.attainability, b.inputs.taskCodes, b.inputs.taskLabels),
  };
}

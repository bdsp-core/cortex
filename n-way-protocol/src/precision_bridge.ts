import type { BankArrays } from "../../cortex_web/apps/web/engine/choose_item";
import type { ParticleState } from "../../cortex_web/apps/web/engine/types";
import { PrecisionPolicy } from "../../cortex_web/apps/web/engine/precision_policy";
import type { PolicyResult } from "../../cortex_web/apps/web/engine/policy";
import type {
  Candidate, ProtocolParticleState,
} from "./types";

/** Convert the protocol's global candidate references to the frozen policy's
 * scalar per-domain telemetry view. Cross-domain signals never become direct
 * evidence here; the policy sees only the focal signal for each asked task. */
export function precisionBankArrays(
  taskCount: number,
  candidates: readonly Candidate[],
  remainingSegmentIds: ReadonlySet<number>,
): BankArrays {
  const bank: BankArrays = {
    sMean: Array.from({ length: taskCount }, () => []),
    sSd: Array.from({ length: taskCount }, () => []),
    segId: Array.from({ length: taskCount }, () => []),
  };
  for (const candidate of candidates) {
    if (!remainingSegmentIds.has(candidate.segId)) continue;
    bank.sMean[candidate.askedK].push(candidate.focalSignal);
    bank.sSd[candidate.askedK].push(candidate.focalSignalSd);
    bank.segId[candidate.askedK].push(candidate.segId);
  }
  return bank;
}

export function evaluateFrozenPrecision(args: {
  policy: PrecisionPolicy;
  state: ProtocolParticleState;
  nPerTask: number[];
  remainingCandidates: readonly Candidate[];
  remainingSegmentIds: ReadonlySet<number>;
}): PolicyResult {
  const bank = precisionBankArrays(
    args.state.K, args.remainingCandidates, args.remainingSegmentIds,
  );
  const telemetry = args.policy.bankTelemetry(bank, args.state.lastRejuvenation);
  // PrecisionPolicy reads only numerical particle arrays and diagnostics. Its
  // binary History type is irrelevant to evaluation and is not accessed.
  return args.policy.evaluate(
    args.state as unknown as ParticleState,
    args.nPerTask,
    telemetry,
  );
}


import type {
  ConditionalF1Artifact, ConditionalF1ArtifactDraw, ConditionalF1ArtifactEnsemble,
  ConditionalF1ResponseArtifact,
} from "./types";

const ensembleCache = new WeakMap<
  ConditionalF1ArtifactEnsemble,
  ConditionalF1ArtifactDraw[]
>();
const momentMatchCache = new WeakMap<
  ConditionalF1ArtifactEnsemble,
  ConditionalF1Artifact
>();

export function normalizedArtifactDraws(
  artifact: ConditionalF1ResponseArtifact,
): ConditionalF1ArtifactDraw[] {
  if (artifact.model === "iiic_conditional_f1_v1") {
    const draw = {
      beta: artifact.beta,
      distractorLapse: artifact.distractorLapse,
      weight: 1,
    };
    if (!Number.isFinite(draw.beta) || draw.beta <= 0) {
      throw new Error("artifact beta must be finite and positive");
    }
    if (!Number.isFinite(draw.distractorLapse)
      || draw.distractorLapse < 0 || draw.distractorLapse > 1) {
      throw new Error("artifact distractor lapse must be in [0, 1]");
    }
    return [draw];
  }
  const cached = ensembleCache.get(artifact);
  if (cached) return cached;
  const raw = artifact.draws;
  if (raw.length === 0) throw new Error("artifact ensemble must contain a draw");
  let weightSum = 0;
  for (const draw of raw) {
    if (!Number.isFinite(draw.beta) || draw.beta <= 0) {
      throw new Error("artifact beta must be finite and positive");
    }
    if (!Number.isFinite(draw.distractorLapse)
      || draw.distractorLapse < 0 || draw.distractorLapse > 1) {
      throw new Error("artifact distractor lapse must be in [0, 1]");
    }
    if (!Number.isFinite(draw.weight) || draw.weight <= 0) {
      throw new Error("artifact draw weights must be finite and positive");
    }
    weightSum += draw.weight;
  }
  const normalized = raw.map((draw) => ({ ...draw, weight: draw.weight / weightSum }));
  ensembleCache.set(artifact, normalized);
  return normalized;
}

/** Cheap, deterministic artifact surrogate used only by Fisher screening. */
export function momentMatchedArtifact(
  artifact: ConditionalF1ResponseArtifact | undefined,
): ConditionalF1Artifact | undefined {
  if (!artifact || artifact.model === "iiic_conditional_f1_v1") return artifact;
  const cached = momentMatchCache.get(artifact);
  if (cached) return cached;
  const draws = normalizedArtifactDraws(artifact);
  const matched: ConditionalF1Artifact = {
    schemaVersion: 1,
    artifactId: `${artifact.artifactId}:moment-match-screen`,
    model: "iiic_conditional_f1_v1",
    qualification: artifact.qualification,
    beta: draws.reduce((sum, draw) => sum + draw.weight * draw.beta, 0),
    distractorLapse: draws.reduce(
      (sum, draw) => sum + draw.weight * draw.distractorLapse, 0,
    ),
    binaryLapse: artifact.binaryLapse,
    sha256: artifact.sha256,
    provenance: artifact.provenance,
  };
  momentMatchCache.set(artifact, matched);
  return matched;
}

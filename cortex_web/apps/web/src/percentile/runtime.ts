import type { TrajectoryCloud } from "../features/exam/results";
import {
  PERCENTILE_DOMAINS,
  type PercentileDomain,
  type PercentileDomainScore,
  type PercentileProfile,
  type PercentileReport,
} from "./types";

interface Block { offset: number; count: number; rows?: number; cols?: number }
interface DomainLayout {
  central: Block;
  bootstrap: Block & { rows: number; cols: number };
  sensitivity: Record<string, Block>;
}
interface RuntimeMetadata {
  schemaVersion: "cortex-percentile-runtime/v1";
  scoreSchemaVersion: "cortex-percentile-score/v1";
  normId: string;
  normSha256: string;
  metadataSha256: string;
  runtimeDataSha256: string;
  runtimeMetadataUrl: string;
  runtimeDataUrl: string;
  referenceLabel: string;
  policyStatus: "provisional_historical_calibration_cohort";
  intervalLevel: 0.95;
  intervalResolutionPercentagePoints: number;
  domainOrder: string[];
  floatEncoding: "float32_le";
  floatCount: number;
  requiredWarnings: string[];
  display: PercentileProfile["displayCopy"];
  domains: Record<PercentileDomain, DomainLayout>;
}
interface LoadedRuntime { metadata: RuntimeMetadata; values: Float32Array }

const cache = new Map<string, Promise<LoadedRuntime>>();

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map(
    (key) => `${JSON.stringify(key)}:${canonical(object[key])}`,
  ).join(",")}}`;
}

async function sha256Hex(data: BufferSource): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function assertProfile(metadata: RuntimeMetadata, profile: PercentileProfile): void {
  const exact: Array<[unknown, unknown, string]> = [
    [metadata.scoreSchemaVersion, profile.scoreSchemaVersion, "score schema"],
    [metadata.normId, profile.normId, "norm id"],
    [metadata.normSha256, profile.normSha256, "norm hash"],
    [metadata.metadataSha256, profile.metadataSha256, "metadata hash"],
    [metadata.runtimeDataSha256, profile.runtimeDataSha256, "runtime hash"],
    [metadata.runtimeMetadataUrl, profile.runtimeMetadataUrl, "metadata URL"],
    [metadata.runtimeDataUrl, profile.runtimeDataUrl, "runtime URL"],
    [metadata.referenceLabel, profile.referenceLabel, "reference label"],
    [metadata.policyStatus, profile.policyStatus, "policy status"],
    [metadata.intervalLevel, profile.intervalLevel, "interval level"],
  ];
  for (const [actual, expected, label] of exact) {
    if (actual !== expected) throw new Error(`percentile ${label} mismatch`);
  }
}

function validateLayout(metadata: RuntimeMetadata, values: Float32Array): void {
  if (metadata.schemaVersion !== "cortex-percentile-runtime/v1"
      || metadata.floatEncoding !== "float32_le"
      || metadata.domainOrder.join(",") !== PERCENTILE_DOMAINS.join(",")
      || metadata.floatCount !== values.length) {
    throw new Error("invalid percentile runtime metadata");
  }
  let next = 0;
  for (const domain of PERCENTILE_DOMAINS) {
    const layout = metadata.domains[domain];
    const blocks = [
      layout.central, layout.bootstrap,
      ...Object.keys(layout.sensitivity).sort().map((k) => layout.sensitivity[k]),
    ];
    for (const block of blocks) {
      if (block.offset !== next || block.count <= 0) {
        throw new Error(`invalid percentile layout for ${domain}`);
      }
      next += block.count;
    }
    if (layout.bootstrap.rows * layout.bootstrap.cols !== layout.bootstrap.count) {
      throw new Error(`invalid percentile bootstrap shape for ${domain}`);
    }
  }
  if (next !== values.length) throw new Error("percentile layout length mismatch");
}

async function load(
  profile: PercentileProfile,
  fetcher: typeof fetch = fetch,
): Promise<LoadedRuntime> {
  const response = await fetcher(profile.runtimeMetadataUrl, { cache: "force-cache" });
  if (!response.ok) throw new Error(`percentile metadata HTTP ${response.status}`);
  const metadata = await response.json() as RuntimeMetadata;
  const { metadataSha256: _metadataSha256, ...withoutHash } = metadata;
  const metaHash = await sha256Hex(new TextEncoder().encode(canonical(withoutHash)));
  if (metaHash !== metadata.metadataSha256) {
    throw new Error("percentile metadata SHA-256 mismatch");
  }
  assertProfile(metadata, profile);
  const dataResponse = await fetcher(profile.runtimeDataUrl, { cache: "force-cache" });
  if (!dataResponse.ok) throw new Error(`percentile runtime HTTP ${dataResponse.status}`);
  const buffer = await dataResponse.arrayBuffer();
  if (await sha256Hex(buffer) !== metadata.runtimeDataSha256) {
    throw new Error("percentile runtime SHA-256 mismatch");
  }
  if (buffer.byteLength !== metadata.floatCount * 4) {
    throw new Error("percentile runtime length mismatch");
  }
  const values = new Float32Array(buffer);
  validateLayout(metadata, values);
  return { metadata, values };
}

export function preloadPercentile(profile: PercentileProfile | null): void {
  if (!profile) return;
  if (!cache.has(profile.runtimeDataSha256)) {
    cache.set(profile.runtimeDataSha256, load(profile));
  }
}

async function runtimeFor(profile: PercentileProfile): Promise<LoadedRuntime> {
  preloadPercentile(profile);
  return cache.get(profile.runtimeDataSha256)!;
}

function cdf(x: number, values: Float32Array, start: number, length: number): number {
  if (x < values[start]) return 0;
  if (x >= values[start + length - 1]) return 1;
  let lo = 0, hi = length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (values[start + mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  const right = lo, left = right - 1;
  const x0 = values[start + left], x1 = values[start + right];
  const p0 = left / (length - 1), p1 = right / (length - 1);
  return x1 <= x0 ? (p0 + p1) / 2 : p0 + ((x - x0) / (x1 - x0)) * (p1 - p0);
}

export function scoreDomain(
  metadata: RuntimeMetadata,
  values: Float32Array,
  domain: PercentileDomain,
  samples: ArrayLike<number>,
  rawWeights: ArrayLike<number>,
): PercentileDomainScore {
  const layout = metadata.domains[domain];
  const { rows, cols, offset } = layout.bootstrap;
  let weightSum = 0;
  for (let i = 0; i < samples.length; i++) weightSum += rawWeights[i];
  if (!samples.length || samples.length !== rawWeights.length || !(weightSum > 0)) {
    throw new Error(`invalid candidate cloud for ${domain}`);
  }
  const resolution = metadata.intervalResolutionPercentagePoints;
  const histogram = new Float64Array(Math.round(100 / resolution) + 1);
  let estimate = 0;
  for (let row = 0; row < rows; row++) {
    const curveStart = offset + row * cols;
    for (let i = 0; i < samples.length; i++) {
      const weight = rawWeights[i] / weightSum / rows;
      const rank = 100 * cdf(samples[i], values, curveStart, cols);
      estimate += weight * rank;
      const bin = Math.min(histogram.length - 1, Math.max(
        0, Math.floor(rank / resolution + 0.5),
      ));
      histogram[bin] += weight;
    }
  }
  const quantile = (probability: number): number => {
    let cumulative = 0;
    for (let index = 0; index < histogram.length; index++) {
      cumulative += histogram[index];
      if (cumulative >= probability) return index * resolution;
    }
    return 100;
  };
  const sensitivity: number[] = [];
  for (const name of Object.keys(layout.sensitivity).sort()) {
    const block = layout.sensitivity[name];
    let value = 0;
    for (let i = 0; i < samples.length; i++) {
      value += (rawWeights[i] / weightSum) * 100
        * cdf(samples[i], values, block.offset, block.count);
    }
    sensitivity.push(value);
  }
  const alpha = 1 - metadata.intervalLevel;
  return {
    estimate,
    lower: quantile(alpha / 2),
    upper: quantile(1 - alpha / 2),
    level: metadata.intervalLevel,
    status: metadata.policyStatus,
    candidateMethod: "posterior_particles",
    referenceReplicates: rows,
    sensitivityRange: sensitivity.length
      ? [Math.min(...sensitivity), Math.max(...sensitivity)] : null,
  };
}

export async function scoreFinalTrajectory(
  profile: PercentileProfile,
  trajectory: TrajectoryCloud,
  taskCodes: string[],
): Promise<PercentileReport> {
  const { metadata, values } = await runtimeFor(profile);
  const [steps, particles, tasks] = trajectory.shape;
  if (steps < 1 || particles < 1 || tasks !== taskCodes.length) {
    throw new Error("final trajectory shape does not match task codes");
  }
  const last = steps - 1;
  const weights = new Float64Array(particles);
  for (let i = 0; i < particles; i++) {
    weights[i] = trajectory.w[last * particles + i];
  }
  const domains: Partial<Record<PercentileDomain, PercentileDomainScore>> = {};
  for (let k = 0; k < tasks; k++) {
    const domain = taskCodes[k] as PercentileDomain;
    if (!PERCENTILE_DOMAINS.includes(domain)) continue;
    const samples = new Float64Array(particles);
    for (let i = 0; i < particles; i++) {
      samples[i] = trajectory.l[(last * particles + i) * tasks + k];
    }
    domains[domain] = scoreDomain(metadata, values, domain, samples, weights);
  }
  if (Object.keys(domains).length !== PERCENTILE_DOMAINS.length) {
    throw new Error("percentile scoring requires the canonical seven domains");
  }
  return { status: "available", profile, domains };
}

export async function scoreOrUnavailable(
  profile: PercentileProfile | null,
  trajectory: TrajectoryCloud,
  taskCodes: string[],
): Promise<PercentileReport | null> {
  if (!profile) return null;
  try {
    return await scoreFinalTrajectory(profile, trajectory, taskCodes);
  } catch (error) {
    console.error("[cortex] percentile preview unavailable", error);
    return { status: "unavailable_runtime_error", profile };
  }
}

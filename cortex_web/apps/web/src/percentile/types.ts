export const PERCENTILE_DOMAINS = [
  "spike", "sz", "lpd", "gpd", "lrda", "grda", "iic",
] as const;

export type PercentileDomain = typeof PERCENTILE_DOMAINS[number];

export interface PercentileProfile {
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
  display: boolean;
  displayCopy: {
    label: string;
    intervalLabel: string;
    disclosure: string;
  };
  requiredWarnings: string[];
}

export interface PercentileDomainScore {
  estimate: number;
  lower: number;
  upper: number;
  level: number;
  status: "provisional_historical_calibration_cohort";
  candidateMethod: "posterior_particles";
  referenceReplicates: number;
  sensitivityRange: [number, number] | null;
}

export interface PercentileReport {
  status: "available" | "unavailable_runtime_error";
  profile: PercentileProfile;
  domains?: Partial<Record<PercentileDomain, PercentileDomainScore>>;
}

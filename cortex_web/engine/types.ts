// Shared engine types.

export interface SegmentMeta {
  segId: number;
  family?: string; // "iiic" | "spike" (display only; the engine gates on NaN sMean)
  patternClass: string; // "seizure" | "lpd" | … | "spike"
  // length K, per-task signal mean/SD. The slot for a task this segment is NOT
  // a candidate for is NaN (spike seg → [val, NaN×6]; IIIC → [NaN, 6 vals]).
  // Bundle.load converts the manifest's JSON `null`s to NaN; the engine
  // family-gates the per-task candidate banks on isNaN (mirrors the desktop
  // cortex_engine_inputs_k7.as_engine_arrays NaN masking).
  sMean: number[];
  sSd: number[];
  fsHz: number;
  nCh: number;
  nSamp: number;
  channelNames: string[];
  eeg: string; // relative path / key of the int16 EEG blob
  spec: string; // relative path / key of the spectrogram blob
}

export interface EngineInputs {
  taskCodes: string[]; // ["sz","lpd","gpd","lrda","grda","iic"]
  taskLabels: string[]; // ["Seizure","LPD","GPD","LRDA","GRDA","Other"]
  taskPatternWords: string[]; // ["seizure","lpd","gpd","lrda","grda","other"]
  corrL: number[][]; // (K×K) fitted prior correlation (both blocks)
  ellStar: number[]; // (K) Youden cut-scores from cert_config v13
  segments: SegmentMeta[];
}

// Particle cloud. t,l are row-major Float64Array(N*K).
export interface ParticleState {
  N: number;
  K: number;
  t: Float64Array;
  l: Float64Array;
  w: Float64Array; // (N) normalized weights
  logPrior: Float64Array; // (N)
  logLik: Float64Array; // (N)
  history: { k: number; s: number; y: 0 | 1; sSd: number }[];
  prior: PriorPieces;
}

export interface PriorPieces {
  K: number;
  sigmaInv: number[][]; // K×K (same matrix for t- and l-blocks here)
  logDet: number; // log|Σ|
  L: number[][]; // K×K Cholesky factor (lower)
}

export interface TrialDiag {
  trialIndex: number;
  taskK: number;
  segId: number;
  s: number;
  sSd: number;
  y: 0 | 1;
  ess: number;
  rejuv: boolean;
  pi: number[]; // per-task pass-mass
  mcse: number[];
  R: number[]; // per-task info gate
  verdicts: string[];
  nPerTask: number[];
  tMean: number[];
  lMean: number[];
  aurocHw: number[]; // per-task AUROC credible halfwidth (for the collapse video)
}

// Bundle loader. In local dev the bundle is served by Vite from
// public/bundle/<version>/; in production it's S3+CloudFront (same shape).
// EEG arrives as int16 (µV × eegScale); spectrogram as uint8 over the fixed
// dB range. Blobs are fetched lazily per segment, cached in memory for the
// session, and persisted to IndexedDB (keyed by bundle version) so a reload
// or a repeat sitting doesn't re-download the bundle.

import { ComputeEngineInputs, EngineInputs, SegmentMeta } from "../engine";
import { cachedArrayBuffer } from "./idbcache";

export interface BundleManifest extends Omit<EngineInputs, "segments"> {
  version: string;
  eegScale: number;
  specDbRange: [number, number];
  nSegments: number;
  segments: (SegmentMeta & {
    specShape: number[] | null;
    specTime?: number[] | null; // [t0,t1] s — spectrogram x-axis extent
    specFreq?: number[] | null; // [f0,f1] Hz — spectrogram y-axis extent
  })[];
}

// A server-drawn per-session bank (POST /api/session, goal 3): the engine
// inputs + the ~drawn segment subset + the bundle base, in place of fetching
// the full manifest. Same shape as BundleManifest (minus nSegments) plus the
// base URL and the draw bookkeeping.
export type SessionBank = Omit<BundleManifest, "nSegments"> & {
  bundleUrl: string;
  sampleSeed?: number;
  nPool?: number;
};

// One engine-selected question: which task is being asked, on which segment,
// at what position in the sitting. Shared by Viewer + SpikeViewer (was declared
// identically in both).
export interface Item {
  trialIndex: number;
  taskK: number;
  segId: number;
}

export interface SegmentData {
  eeg: Float32Array; // (nCh × nSamp) row-major, µV
  nCh: number;
  nSamp: number;
  fsHz: number;
  channelNames: string[];
  spec: { data: Uint8Array; shape: number[]; time?: number[] | null; freq?: number[] | null } | null;
}

export class Bundle {
  readonly base: string;
  readonly manifest: BundleManifest;
  private cache = new Map<number, SegmentData>();
  private pending = new Map<number, Promise<SegmentData>>();
  private computeProfile: ComputeEngineInputs | null = null;

  private constructor(base: string, manifest: BundleManifest) {
    this.base = base;
    this.manifest = manifest;
  }

  // Build a Bundle from a server-drawn session bank (goal 3) — no full-manifest
  // fetch. The engine runs over bank.segments; EEG/spec blobs are still fetched
  // lazily from bank.bundleUrl via each segment's `eeg`/`spec` path.
  static fromSessionBank(bank: SessionBank): Bundle {
    return new Bundle(bank.bundleUrl, bank as unknown as BundleManifest);
  }

  get inputs(): EngineInputs {
    const m = this.manifest;
    return {
      taskCodes: m.taskCodes,
      taskLabels: m.taskLabels,
      taskPatternWords: m.taskPatternWords,
      taskClasses: m.taskClasses,   // K=7 routing — Viewer/Results need this
      certBlock: m.certBlock,       // ℓ* lineage provenance for the result file
      corrL: m.corrL,
      corrT: m.corrT,               // v15 OPT-IN (undefined on the frozen-pilot manifest)
      nParticles: m.nParticles,     // v15 OPT-IN (undefined → engine defaults to 600)
      perDomainCap: m.perDomainCap,
      terminationPolicy: m.terminationPolicy,
      nwayProfile: m.nwayProfile,
      precisionBandEdges: m.precisionBandEdges,
      ellStar: m.ellStar,
      segments: m.segments,
    };
  }

  /**
   * Minimal immutable payload sent to the engine worker. The complete manifest
   * remains owned by Bundle so the selected segId can still resolve its exact
   * EEG and spectrogram. Keeping UI-only strings and shape metadata out of the
   * worker prevents the 35k manifest from being duplicated into every compute
   * isolate.
   */
  get computeInputs(): ComputeEngineInputs {
    if (this.computeProfile) return this.computeProfile;
    const profile = this.inputs;
    this.computeProfile = {
      ...profile,
      segments: this.manifest.segments.map((segment) => ({
        segId: segment.segId,
        ...(segment.applicableTaskIdx
          ? { applicableTaskIdx: segment.applicableTaskIdx.slice() }
          : {}),
        sMean: segment.sMean.slice(),
        sSd: segment.sSd.slice(),
      })),
    };
    return this.computeProfile;
  }

  segment(segId: number): Promise<SegmentData> {
    const hit = this.cache.get(segId);
    if (hit) return Promise.resolve(hit);
    const pending = this.pending.get(segId);
    if (pending) return pending;
    const request = this.loadSegment(segId).finally(() => {
      if (this.pending.get(segId) === request) this.pending.delete(segId);
    });
    this.pending.set(segId, request);
    return request;
  }

  private async loadSegment(segId: number): Promise<SegmentData> {
    const meta = this.manifest.segments.find((s) => s.segId === segId);
    if (!meta) throw new Error(`segment ${segId} not in manifest`);

    const ver = this.manifest.version;
    const eegBuf = await cachedArrayBuffer(`${this.base}/${meta.eeg}`, `${ver}/${meta.eeg}`);
    const i16 = new Int16Array(eegBuf);
    const eeg = new Float32Array(i16.length);
    const inv = 1 / this.manifest.eegScale;
    for (let i = 0; i < i16.length; i++) eeg[i] = i16[i] * inv;

    let spec: SegmentData["spec"] = null;
    if (meta.spec) {
      const sBuf = await cachedArrayBuffer(`${this.base}/${meta.spec}`, `${ver}/${meta.spec}`);
      spec = { data: new Uint8Array(sBuf), shape: meta.specShape ?? [],
               time: meta.specTime ?? null, freq: meta.specFreq ?? null };
    }

    const data: SegmentData = {
      eeg,
      nCh: meta.nCh,
      nSamp: meta.nSamp,
      fsHz: meta.fsHz,
      channelNames: meta.channelNames,
      spec,
    };
    this.cache.set(segId, data);
    return data;
  }

  // Prefetch a few segments (called for the next likely items; harmless if
  // they're never used).
  prefetch(segIds: number[]): void {
    for (const id of segIds) void this.segment(id).catch(() => {});
  }
}

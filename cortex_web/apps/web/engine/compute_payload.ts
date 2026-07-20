import type { ComputeEngineInputs, ComputeSegmentMeta } from "./types";

export interface PackedSegments {
  length: number;
  segIds: Float64Array;
  applicableMasks: Uint32Array;
  sMean: Float64Array;
  sSd: Float64Array;
}

/** Structured-clone payload: immutable profile plus a compact segment SoA. */
export interface PackedComputeInputs {
  profile: Omit<ComputeEngineInputs, "segments">;
  segments: PackedSegments;
}

/**
 * Pack UI-independent calculation data into transferable typed arrays.
 * K is frozen at seven in production; the 31-task bound makes applicability
 * masks explicit instead of silently truncating a future profile.
 */
export function packComputeInputs(inputs: ComputeEngineInputs): PackedComputeInputs {
  const K = inputs.taskCodes.length;
  if (K < 1 || K > 31) throw new Error(`unsupported compute task count: ${K}`);
  const length = inputs.segments.length;
  const segIds = new Float64Array(length);
  const applicableMasks = new Uint32Array(length);
  const sMean = new Float64Array(length * K);
  const sSd = new Float64Array(length * K);
  const allTasksMask = (2 ** K - 1) >>> 0;

  for (let i = 0; i < length; i += 1) {
    const segment = inputs.segments[i];
    if (segment.sMean.length !== K || segment.sSd.length !== K) {
      throw new Error(`segment ${segment.segId} signal dimension does not match K=${K}`);
    }
    segIds[i] = segment.segId;
    let mask = segment.applicableTaskIdx === undefined ? allTasksMask : 0;
    for (const taskK of segment.applicableTaskIdx ?? []) {
      if (!Number.isInteger(taskK) || taskK < 0 || taskK >= K) {
        throw new Error(`segment ${segment.segId} has invalid task index ${taskK}`);
      }
      mask |= 1 << taskK;
    }
    applicableMasks[i] = mask >>> 0;
    sMean.set(segment.sMean, i * K);
    sSd.set(segment.sSd, i * K);
  }

  const { segments: _segments, ...profile } = inputs;
  return {
    profile,
    segments: { length, segIds, applicableMasks, sMean, sSd },
  };
}

export function unpackComputeInputs(payload: PackedComputeInputs): ComputeEngineInputs {
  const K = payload.profile.taskCodes.length;
  const packed = payload.segments;
  if (packed.segIds.length !== packed.length
      || packed.applicableMasks.length !== packed.length
      || packed.sMean.length !== packed.length * K
      || packed.sSd.length !== packed.length * K) {
    throw new Error("invalid packed compute payload dimensions");
  }
  const segments = new Array<ComputeSegmentMeta>(packed.length);
  for (let i = 0; i < packed.length; i += 1) {
    const mask = packed.applicableMasks[i];
    const applicableTaskIdx: number[] = [];
    for (let taskK = 0; taskK < K; taskK += 1) {
      if ((mask & (1 << taskK)) !== 0) applicableTaskIdx.push(taskK);
    }
    segments[i] = {
      segId: packed.segIds[i],
      applicableTaskIdx,
      sMean: Array.from(packed.sMean.subarray(i * K, (i + 1) * K)),
      sSd: Array.from(packed.sSd.subarray(i * K, (i + 1) * K)),
    };
  }
  return { ...payload.profile, segments };
}

export function computePayloadTransferables(payload: PackedComputeInputs): Transferable[] {
  return [
    payload.segments.segIds.buffer,
    payload.segments.applicableMasks.buffer,
    payload.segments.sMean.buffer,
    payload.segments.sSd.buffer,
  ];
}

export function packedSegmentBytes(payload: PackedComputeInputs): number {
  const segments = payload.segments;
  return segments.segIds.byteLength
    + segments.applicableMasks.byteLength
    + segments.sMean.byteLength
    + segments.sSd.byteLength;
}

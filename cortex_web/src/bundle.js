// Bundle loader. In local dev the bundle is served by Vite from
// public/bundle/<version>/; in production it's S3+CloudFront (same shape).
// EEG arrives as int16 (µV × eegScale); spectrogram as uint8 over the fixed
// dB range. Blobs are fetched lazily per segment and cached in memory.
//
// IndexedDB persistence (survive reload, prefetch-ahead) is a documented
// follow-up (PLAN §5); the in-memory cache is enough for the dev slice.
export class Bundle {
    base;
    manifest;
    cache = new Map();
    constructor(base, manifest) {
        this.base = base;
        this.manifest = manifest;
    }
    static async load(base) {
        const manifest = (await (await fetch(`${base}/manifest.json`)).json());
        return new Bundle(base, manifest);
    }
    get inputs() {
        const m = this.manifest;
        return {
            taskCodes: m.taskCodes,
            taskLabels: m.taskLabels,
            taskPatternWords: m.taskPatternWords,
            corrL: m.corrL,
            ellStar: m.ellStar,
            segments: m.segments,
        };
    }
    async segment(segId) {
        const hit = this.cache.get(segId);
        if (hit)
            return hit;
        const meta = this.manifest.segments.find((s) => s.segId === segId);
        if (!meta)
            throw new Error(`segment ${segId} not in manifest`);
        const eegBuf = await (await fetch(`${this.base}/${meta.eeg}`)).arrayBuffer();
        const i16 = new Int16Array(eegBuf);
        const eeg = new Float32Array(i16.length);
        const inv = 1 / this.manifest.eegScale;
        for (let i = 0; i < i16.length; i++)
            eeg[i] = i16[i] * inv;
        let spec = null;
        if (meta.spec) {
            const sBuf = await (await fetch(`${this.base}/${meta.spec}`)).arrayBuffer();
            spec = { data: new Uint8Array(sBuf), shape: meta.specShape ?? [] };
        }
        const data = {
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
    prefetch(segIds) {
        for (const id of segIds)
            if (!this.cache.has(id))
                void this.segment(id).catch(() => { });
    }
}

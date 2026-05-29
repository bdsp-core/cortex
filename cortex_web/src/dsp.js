// EEG display filtering — zero-phase Butterworth bandpass + IIR notch.
//
// The desktop applies scipy butter(N=2, btype='band') + iirnotch via
// sosfiltfilt (forward-backward, zero-phase). This is a faithful biquad
// reimplementation: a bandpass built as a cascade of a 2nd-order high-pass
// and 2nd-order low-pass (matching the N=2-per-edge response), and an RBJ
// notch, all run forward then backward for zero phase.
//
// NOTE: visually matches the desktop; exact scipy-coefficient parity is a
// documented refinement (PLAN §6). The cutoffs, zero-phase property, and
// roll-off are correct, which is what the clinician sees.
// Direct-form-II transposed, one pass.
function applyBiquad(x, q, out) {
    let z1 = 0, z2 = 0;
    for (let i = 0; i < x.length; i++) {
        const xi = x[i];
        const yi = q.b0 * xi + z1;
        z1 = q.b1 * xi - q.a1 * yi + z2;
        z2 = q.b2 * xi - q.a2 * yi;
        out[i] = yi;
    }
}
// forward-backward (zero-phase) cascade of biquads, in place on a copy.
export function filtfilt(x, cascade) {
    let cur = x.slice();
    const tmp = new Float32Array(x.length);
    for (const q of cascade) {
        applyBiquad(cur, q, tmp);
        cur.set(tmp);
    }
    // reverse
    cur.reverse();
    for (const q of cascade) {
        applyBiquad(cur, q, tmp);
        cur.set(tmp);
    }
    cur.reverse();
    return cur;
}
// 2nd-order Butterworth high-pass (RBJ cookbook, Q = 1/√2).
export function butterHighpass(fc, fs) {
    const w0 = (2 * Math.PI * fc) / fs;
    const cos = Math.cos(w0), sin = Math.sin(w0);
    const alpha = sin / Math.SQRT2;
    const a0 = 1 + alpha;
    return {
        b0: ((1 + cos) / 2) / a0,
        b1: (-(1 + cos)) / a0,
        b2: ((1 + cos) / 2) / a0,
        a1: (-2 * cos) / a0,
        a2: (1 - alpha) / a0,
    };
}
// 2nd-order Butterworth low-pass.
export function butterLowpass(fc, fs) {
    const w0 = (2 * Math.PI * fc) / fs;
    const cos = Math.cos(w0), sin = Math.sin(w0);
    const alpha = sin / Math.SQRT2;
    const a0 = 1 + alpha;
    return {
        b0: ((1 - cos) / 2) / a0,
        b1: (1 - cos) / a0,
        b2: ((1 - cos) / 2) / a0,
        a1: (-2 * cos) / a0,
        a2: (1 - alpha) / a0,
    };
}
// RBJ notch at f0 with quality Q (desktop uses iirnotch Q=30).
export function notch(f0, fs, Q = 30) {
    const w0 = (2 * Math.PI * f0) / fs;
    const cos = Math.cos(w0), sin = Math.sin(w0);
    const alpha = sin / (2 * Q);
    const a0 = 1 + alpha;
    return {
        b0: 1 / a0,
        b1: (-2 * cos) / a0,
        b2: 1 / a0,
        a1: (-2 * cos) / a0,
        a2: (1 - alpha) / a0,
    };
}
// Parse the desktop bandpass dropdown string ("0.5-70 Hz" | "1-70 Hz" | "off")
// into [lo, hi] Hz or null.
export function parseBandpass(s) {
    if (s === "off")
        return null;
    const m = s.match(/([\d.]+)-([\d.]+)/);
    return m ? [parseFloat(m[1]), parseFloat(m[2])] : null;
}
// Build the filter cascade for a (bandpass, notch) choice.
export function buildCascade(bandpass, notchHz, fs) {
    const cascade = [];
    const bp = parseBandpass(bandpass);
    if (bp) {
        const [lo, hi] = bp;
        cascade.push(butterHighpass(lo, fs));
        cascade.push(butterLowpass(Math.min(hi, fs / 2 - 1), fs));
    }
    if (notchHz !== "off") {
        const f0 = parseFloat(notchHz);
        if (f0 < fs / 2)
            cascade.push(notch(f0, fs));
    }
    return cascade;
}

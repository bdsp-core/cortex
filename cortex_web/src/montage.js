// Montage transforms — port of apply_bipolar / apply_average / apply_laplacian
// from scripts/eeg_bank_viewer.py. NaN-separator rows become gaps; the EKG/ECG
// channel is appended at the end of each montage.
//
// Input: the bundle's channel-ordered EEG (nCh × nSamp, row-major Float32),
// plus the channel-name list. Output: an array of { name, data|null } rows,
// where data===null is a blank-separator row.
import { BIPOLAR_PAIRS, BIPOLAR_SEPARATOR_AFTER } from "../ui/theme";
function rowOf(eeg, idx, nSamp) {
    return eeg.subarray(idx * nSamp, idx * nSamp + nSamp);
}
function diff(a, b) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++)
        out[i] = a[i] - b[i];
    return out;
}
function chMap(names) {
    const m = new Map();
    names.forEach((n, i) => m.set(n, i));
    return m;
}
function ekgRow(names, eeg, nSamp) {
    const idx = names.findIndex((n) => /^(ekg|ecg)$/i.test(n));
    if (idx < 0)
        return null;
    return { name: "EKG", data: rowOf(eeg, idx, nSamp), isEkg: true };
}
export function bipolar(eeg, names, nSamp) {
    const cm = chMap(names);
    const rows = [];
    BIPOLAR_PAIRS.forEach(([a, b], i) => {
        const ia = cm.get(a), ib = cm.get(b);
        const data = ia !== undefined && ib !== undefined
            ? diff(rowOf(eeg, ia, nSamp), rowOf(eeg, ib, nSamp))
            : new Float32Array(nSamp);
        rows.push({ name: `${a}-${b}`, data });
        if (BIPOLAR_SEPARATOR_AFTER.includes(i))
            rows.push({ name: "", data: null });
    });
    const ekg = ekgRow(names, eeg, nSamp);
    if (ekg) {
        rows.push({ name: "", data: null });
        rows.push(ekg);
    }
    return rows;
}
// Average reference: each EEG channel minus the 19-ch mean.
export function average(eeg, names, nSamp) {
    const nEeg = Math.min(19, names.length);
    const mean = new Float32Array(nSamp);
    for (let c = 0; c < nEeg; c++) {
        const r = rowOf(eeg, c, nSamp);
        for (let i = 0; i < nSamp; i++)
            mean[i] += r[i] / nEeg;
    }
    const rows = [];
    for (let c = 0; c < nEeg; c++) {
        const r = rowOf(eeg, c, nSamp);
        const data = new Float32Array(nSamp);
        for (let i = 0; i < nSamp; i++)
            data[i] = r[i] - mean[i];
        rows.push({ name: `${names[c]}-Av`, data });
    }
    const ekg = ekgRow(names, eeg, nSamp);
    if (ekg) {
        rows.push({ name: "", data: null });
        rows.push(ekg);
    }
    return rows;
}
// Laplacian placeholder: until the per-channel neighbourhoods are ported,
// fall back to the average montage so the control still works end-to-end.
// TODO(PLAN §6): port the exact neighbour sets from eeg_bank_viewer.py L91-103.
export function laplacian(eeg, names, nSamp) {
    return average(eeg, names, nSamp);
}
export function applyMontage(kind, eeg, names, nSamp) {
    if (kind === "average")
        return average(eeg, names, nSamp);
    if (kind === "laplacian")
        return laplacian(eeg, names, nSamp);
    return bipolar(eeg, names, nSamp);
}

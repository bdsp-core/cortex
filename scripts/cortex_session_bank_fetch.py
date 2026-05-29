"""Phase 9 Layer 6b — per-session CORTEX bank fetcher.

Called at session start by CORTEX. Reads the production bank's MANIFEST
(local or cloud), stratified-random samples N segments per task/family
seeded by `session_id`, and materialises a session-local HDF5 bank that
the engine reads as if it were the local `data/eeg_bank.h5`.

Per Eli's directive: "no two tests are the same due to our large bank of
questions." Per-session sampling from ~36,700-seg production pool gives:

  Per-session draw: per_task × 7 = 1400 (default 200/task)
  P(any two sessions share a seg in task k):
    spike  = (200/16,681)² × 16,681 ≈ 2.4 ⟶ ~2 expected shared / session-pair
    IIIC   = (200/3,200)² × 3,200    ≈ 12.5 ⟶ ~13 expected shared / pair
                                              (per-class; class-sizes vary)
  At the SESSION level (1400 segs / 36,700 pool), expected overlap ≈ 53 segs
  / session-pair. The adaptive QUEST+ engine selects a different subset to
  ASK from this 1400, so the effective per-session test sequence overlap
  is far smaller still.

Backend interface (deferred-choice per Eli's Q3 Layer-6b):
  manifest entries declare a `group_path` ("iiic/<sid>" or "spike/<sid>");
  `bank_path` is interpreted as:
    - local path → open via h5py and copy the group across
    - http(s):// URL pointing at a single h5 → download with byte-range
      requests (NOT implemented in Phase 9 — placeholder; needs an HTTP
      range-capable server on the cloud side)
    - http(s):// URL pointing at a manifest with per-seg payload_url
      entries → download per-file (NOT implemented in Phase 9 — placeholder)

The Phase-9 ship state supports LOCAL bank_path. Cloud backends are
trivial extensions once Eli picks a cloud backend; the BankBackend
interface insulates the fetcher from that choice.

Offline-fallback contract:
  If the manifest URL is unreachable OR the per-session fetch fails,
  CORTEX falls back to `cortex_app/cortex_offline_fallback.h5`
  (built by build_cortex_bank_k7_production.py --mode fallback).
  Fetcher returns the path of whichever bank materialised.

Usage:
    .venv/bin/python scripts/cortex_session_bank_fetch.py \
        --manifest data/production_bank/MANIFEST.json \
        --out /tmp/cortex_session_42.h5 \
        --session-id ses_2026_05_29_eli_001 \
        --per-task 200
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OFFLINE_FALLBACK = REPO / "cortex_app" / "cortex_offline_fallback.h5"
DEFAULT_OFFLINE_FALLBACK_MANIFEST = (
    REPO / "cortex_app" / "cortex_offline_fallback.manifest.json")

DEFAULT_PER_TASK = 60  # v1.2.0: lowered from 200 — internal-test bandwidth budget.
                       # Empirical AD6 production settings (N_MIN=15, ALPHA=0.10)
                       # produce ~30 asks/task median; 60 gives ~2× choice margin
                       # for the adaptive selector. ~500 MB/session download.
N_STRATA = 10
TASK_CODES = ("spike", "sz", "lpd", "gpd", "lrda", "grda", "iic")
IIIC_CODES = TASK_CODES[1:]
# Pattern-class plurality → engine task code (for IIIC stratification).
# spike segments are family=spike (the pattern_class field is set to "spike"
# in the manifest); IIIC segments carry pattern_class ∈ {seizure, lpd, ...}.
_CLS_TO_CODE = {
    "spike":   "spike",
    "seizure": "sz",
    "lpd":     "lpd",
    "gpd":     "gpd",
    "lrda":    "lrda",
    "grda":    "grda",
    "other":   "iic",
}


# ── manifest loading ───────────────────────────────────────────────────────

def _load_manifest(source: str) -> dict:
    """Load MANIFEST.json from a local path or http(s) URL."""
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(source, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {source}")
    return json.loads(p.read_text())


# ── sampling ───────────────────────────────────────────────────────────────

def _session_seed(session_id: str) -> int:
    """Deterministic seed derived from session_id (stable across runs)."""
    h = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


def _bucket_segments(segments: list[dict]) -> dict[str, list[dict]]:
    """Bucket segments by engine task code (spike + 6 IIIC pattern_class)."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        if seg["family"] == "spike":
            buckets["spike"].append(seg)
            continue
        cls = seg.get("pattern_class", "").strip().lower()
        code = _CLS_TO_CODE.get(cls)
        if code is None:
            # unknown pattern_class — skip (manifest schema bug)
            continue
        buckets[code].append(seg)
    return buckets


def _stratified_sample(segs: list[dict], n: int, key: str,
                        rng: np.random.Generator,
                        n_strata: int = N_STRATA) -> list[dict]:
    """Stratified-by-quantile sample of `segs` on the `key` attribute,
    drawing n total. Falls back to uniform random if `key` is missing
    or has too few finite values to stratify."""
    if not segs:
        return []
    values = np.array([seg.get(key, np.nan) for seg in segs], dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2 * n_strata:
        # Not enough finite values to stratify — uniform sample.
        idx = rng.choice(len(segs), size=min(n, len(segs)), replace=False)
        return [segs[int(i)] for i in idx]
    fvals = values[finite]
    quantiles = np.linspace(0, 1, n_strata + 1)
    bins = np.unique(np.quantile(fvals, quantiles))
    if len(bins) < 2:
        idx = rng.choice(len(segs), size=min(n, len(segs)), replace=False)
        return [segs[int(i)] for i in idx]
    per_stratum = max(1, n // (len(bins) - 1))
    out: list[dict] = []
    # Map each seg to its stratum (segs with NaN key → uniform pool).
    stratum_of = np.full(len(segs), -1, dtype=int)
    stratum_of[finite] = np.clip(
        np.digitize(values[finite], bins[1:-1], right=False), 0, len(bins) - 2)
    for s in range(len(bins) - 1):
        idxs_s = np.flatnonzero(stratum_of == s)
        if idxs_s.size == 0:
            continue
        k = min(per_stratum, idxs_s.size)
        chosen = rng.choice(idxs_s, size=k, replace=False)
        out.extend(segs[int(i)] for i in chosen)
    # If under-target due to small strata, top up uniformly from remainder.
    if len(out) < n:
        chosen_ids = {seg["seg_id"] for seg in out}
        remainder = [seg for seg in segs if seg["seg_id"] not in chosen_ids]
        if remainder:
            top_up = rng.choice(
                len(remainder),
                size=min(n - len(out), len(remainder)),
                replace=False)
            out.extend(remainder[int(i)] for i in top_up)
    return out[:n]


def sample_session_segments(manifest: dict, session_id: str,
                             per_task: int = DEFAULT_PER_TASK) -> list[dict]:
    """Return the session's per-task stratified sample of segments.

    Stratifies spike by `s_mean`; IIIC tasks by `s_mean_<task>`. Both come
    from the K=7 V_B SVI posterior means written into the production
    manifest by build_cortex_bank_k7_production.py.

    Returns a flat list of segment dicts (manifest entries) in deterministic
    order (sorted by `seg_id` within each task bucket).
    """
    buckets = _bucket_segments(manifest["segments"])
    seed = _session_seed(session_id)
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for code in TASK_CODES:
        segs = buckets.get(code, [])
        if not segs:
            continue
        stratify_key = "s_mean" if code == "spike" else f"s_mean_{code}"
        sample = _stratified_sample(segs, per_task, stratify_key, rng)
        sample.sort(key=lambda seg: seg["seg_id"])
        out.extend(sample)
    return out


# ── backend abstraction (cloud-deferred) ───────────────────────────────────

class BankBackend:
    """Reads h5 groups from a bank (local file or cloud — future)."""
    def open(self): ...
    def close(self): ...
    def copy_group(self, group_path: str, dst: h5py.File): ...


class LocalBankBackend(BankBackend):
    """Reads from a local h5 file (Phase-9 ship state)."""
    def __init__(self, bank_path: Path):
        self.bank_path = Path(bank_path)
        self._f: h5py.File | None = None

    def open(self):
        if not self.bank_path.exists():
            raise FileNotFoundError(f"bank not found: {self.bank_path}")
        self._f = h5py.File(self.bank_path, "r")

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def copy_group(self, group_path: str, dst: h5py.File):
        if self._f is None:
            raise RuntimeError("backend not open")
        if group_path not in self._f:
            raise KeyError(f"group missing from bank: {group_path}")
        # Make sure parent group exists in dst
        parent = "/".join(group_path.split("/")[:-1])
        if parent and parent not in dst:
            dst.create_group(parent)
        # h5py copies the dataset + attrs intact.
        self._f.copy(group_path, dst, name=group_path)


class HTTPBankBackend(BankBackend):
    """Placeholder: would fetch h5 groups over HTTP byte-range or per-file.
    Not implemented in Phase 9. Selecting an HTTP backend raises so callers
    fall back to offline. To be wired up when Eli picks a cloud backend."""
    def __init__(self, bank_url: str):
        self.bank_url = bank_url

    def open(self):
        raise NotImplementedError(
            "HTTP bank backend not implemented in Phase 9 — pick a cloud "
            "backend (S3 / Dropbox / etc.) and add a concrete subclass.")

    def close(self): pass
    def copy_group(self, *a, **kw): pass


def _make_backend(manifest: dict) -> BankBackend:
    bank_path = manifest.get("bank_path")
    bank_url = manifest.get("bank_url")
    if bank_url:
        return HTTPBankBackend(bank_url)
    if bank_path:
        # Resolve relative to repo
        return LocalBankBackend(REPO / bank_path)
    raise KeyError("manifest has neither bank_path nor bank_url")


# ── session bank materialisation ───────────────────────────────────────────

def materialise_session_bank(segments: list[dict],
                              backend: BankBackend,
                              out_path: Path,
                              *, session_id: str,
                              source_manifest_path: str | None = None,
                              ) -> Path:
    """Write the per-session h5 bank from the chosen `segments` list."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    backend.open()
    try:
        with h5py.File(out_path, "w") as dst:
            dst.attrs["session_id"] = session_id
            dst.attrs["phase"] = 9
            dst.attrs["K"] = 7
            dst.attrs["schema_version"] = "session_v1"
            dst.attrs["materialised_utc"] = (
                datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds"))
            if source_manifest_path:
                dst.attrs["source_manifest"] = str(source_manifest_path)
            for seg in segments:
                backend.copy_group(seg["group_path"], dst)
    finally:
        backend.close()
    return out_path


def fetch_session_bank(manifest_source: str, session_id: str,
                        *, out_path: Path,
                        per_task: int = DEFAULT_PER_TASK,
                        offline_fallback_path: Path | None = DEFAULT_OFFLINE_FALLBACK,
                        ) -> Path:
    """Top-level entry. Try the configured manifest; fall back to offline.

    Returns the path of the materialised session bank.
    """
    try:
        manifest = _load_manifest(manifest_source)
        backend = _make_backend(manifest)
        sampled = sample_session_segments(manifest, session_id, per_task=per_task)
        materialise_session_bank(
            sampled, backend, out_path,
            session_id=session_id,
            source_manifest_path=manifest_source)
        return out_path
    except (FileNotFoundError, NotImplementedError, KeyError, OSError) as e:
        # Cloud unreachable / backend not yet wired / manifest missing — fall back.
        print(f"  WARN: session fetch failed ({type(e).__name__}: {e}); "
              f"falling back to offline bundle", flush=True)
        if offline_fallback_path is None or not offline_fallback_path.exists():
            raise
        # Just copy the offline bundle to out_path (session "is" the offline
        # bundle — no per-session sampling possible without manifest).
        shutil.copy(offline_fallback_path, out_path)
        # Stamp the copy
        with h5py.File(out_path, "r+") as dst:
            dst.attrs["session_id"] = session_id
            dst.attrs["offline_fallback"] = True
        return out_path


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="path or URL to MANIFEST.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="destination h5 path for the session bank")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK)
    ap.add_argument("--offline-fallback", type=Path,
                    default=DEFAULT_OFFLINE_FALLBACK)
    args = ap.parse_args(argv)

    print("=== Phase 9 Layer 6b — per-session bank fetch ===")
    print(f"  manifest:    {args.manifest}")
    print(f"  session_id:  {args.session_id}")
    print(f"  per_task:    {args.per_task}  (×7 tasks = "
          f"{args.per_task * 7} target segs)")
    print(f"  out:         {args.out}")
    print(f"  fallback:    {args.offline_fallback}")

    out = fetch_session_bank(
        args.manifest, args.session_id,
        out_path=args.out,
        per_task=args.per_task,
        offline_fallback_path=args.offline_fallback if
                              args.offline_fallback.exists() else None)
    sz_mb = out.stat().st_size / 1024 / 1024
    with h5py.File(out, "r") as f:
        n_spike = len(f.get("spike", {}))
        n_iiic = len(f.get("iiic", {}))
        is_fallback = bool(f.attrs.get("offline_fallback", False))
    print(f"  written:     {n_spike + n_iiic} segs ({n_spike} spike + "
          f"{n_iiic} iiic) → {sz_mb:.1f} MB"
          f"{'   [OFFLINE FALLBACK]' if is_fallback else ''}")
    print("=== Phase 9 Layer 6b — per-session bank fetch COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

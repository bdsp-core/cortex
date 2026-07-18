"""Server-side per-session question-bank draw (goal 3).

The full ~35k bank's signal index — the bundle ``manifest.json`` that
``prepare_web_bundle.py`` emits — is loaded ONCE at boot (read-only, shared).
AD6 session starts draw a seeded, class-balanced, difficulty-stratified subset
(~``target`` segments), excluding the participant's recently-seen segments.
The account-scoped Precision pilot instead receives the complete exposure-
eligible served bank required by its frozen coarse-to-fine selector profile.

This is a faithful Python port of ``apps/web/src/sampleSession.ts`` operating on
the same web manifest schema (``patternClass`` groups, difficulty = the
segment's signal on its OWN true task, 8 contiguous difficulty bins drawn
round-robin). Pure-Python ``random.Random(seed)`` makes the draw reproducible
from the stored ``sample_seed``.
"""
from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Optional

# Difficulty strata per class (mirrors sampleSession.ts N_BINS).
_N_BINS = 8

# Top-level engine-input keys the client needs alongside the drawn segments.
# Optional ones (corrT / nParticles / perDomainCap / taskClasses) are only
# present on the v15+ manifests; absent → the client engine uses its defaults.
_ENGINE_KEYS = (
    "version", "eegScale", "specDbRange", "taskCodes", "taskLabels",
    "taskPatternWords", "taskClasses", "certBlock", "ellStar", "corrL",
    "corrT", "nParticles", "perDomainCap", "precisionBandEdges",
)


def _linear_quantile(values: list[float], q: float) -> float:
    """NumPy-compatible default linear quantile over a finite vector."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty vector")
    position = (len(ordered) - 1) * float(q)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


class SessionBank:
    """Loads one web bundle manifest and draws per-session subsets from it."""

    def __init__(self, manifest_path: Path | str, bundle_url: str):
        self.bundle_url = bundle_url
        manifest_bytes = Path(manifest_path).read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        self.version = manifest.get("version")
        self.engine = {k: manifest[k] for k in _ENGINE_KEYS if k in manifest}
        self.segments: list[dict] = manifest["segments"]
        # PrecisionPolicy's three-per-tercile floor is defined against the full
        # served bank, not a participant's random session draw. Older manifests
        # do not carry the edges, so derive and attach them once at bank load.
        if "precisionBandEdges" not in self.engine:
            edges: list[list[float]] = []
            valid_edges = True
            for k in range(len(manifest["taskCodes"])):
                values = [
                    float(seg["sMean"][k])
                    for seg in self.segments
                    if (not seg.get("applicableTaskIdx")
                        or k in seg["applicableTaskIdx"])
                ]
                if len(values) < 3:
                    valid_edges = False
                    break
                q1 = _linear_quantile(values, 1.0 / 3.0)
                q2 = _linear_quantile(values, 2.0 / 3.0)
                if not q1 < q2:
                    valid_edges = False
                    break
                edges.append([q1, q2])
            if valid_edges:
                self.engine["precisionBandEdges"] = edges
        words = manifest["taskPatternWords"]
        self._word_idx = {w: i for i, w in enumerate(words)}
        # Group segment indices by pattern class once, at load.
        self._by_class: dict[str, list[int]] = {}
        for i, seg in enumerate(self.segments):
            self._by_class.setdefault(seg["patternClass"], []).append(i)
        self._by_id = {seg["segId"]: seg for seg in self.segments}

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def _difficulty(self, seg: dict) -> float:
        # The segment's signal on its OWN (true) task: high = clear/easy.
        k = self._word_idx.get(seg["patternClass"], -1)
        return seg["sMean"][k] if 0 <= k < len(seg["sMean"]) else 0.0

    def draw(self, seed: int, target: int,
             exclude: Optional[set] = None) -> dict:
        """Return ``{**engine_inputs, bundleUrl, sampleSeed, nPool, segments}``.

        Stratified, class-balanced, difficulty-spread sample of ~``target``
        segments seeded by ``seed``, with any seg_id in ``exclude`` removed
        from the candidate pool first (spacing)."""
        exclude = exclude or set()
        rng = random.Random(seed)
        classes = list(self._by_class.keys())
        per_class = -(-target // max(1, len(classes)))  # ceil division

        chosen: list[int] = []
        for cls in classes:
            pool = [i for i in self._by_class[cls]
                    if self.segments[i]["segId"] not in exclude]
            # sort by difficulty, split into N_BINS contiguous bins, draw
            # round-robin across bins (shuffled within bin) until per_class.
            pool.sort(key=lambda i: self._difficulty(self.segments[i]))
            n = len(pool)
            bins: list[list[int]] = [[] for _ in range(_N_BINS)]
            for j, idx in enumerate(pool):
                b = min(_N_BINS - 1, (j * _N_BINS) // n) if n else 0
                bins[b].append(idx)
            for b in bins:
                rng.shuffle(b)
            picks: list[int] = []
            bi = exhausted = 0
            while len(picks) < per_class and exhausted < _N_BINS:
                b = bins[bi % _N_BINS]
                if b:
                    picks.append(b.pop())
                    exhausted = 0
                else:
                    exhausted += 1
                bi += 1
            chosen.extend(picks)

        # Trim to target with a shuffle so the trim doesn't bias to the last
        # class (mirrors sampleSession.ts).
        rng.shuffle(chosen)
        chosen = chosen[:target]
        segs = [self.segments[i] for i in chosen]
        n_pool = sum(1 for s in self.segments if s["segId"] not in exclude)
        return {
            **self.engine,
            "bundleUrl": self.bundle_url,
            "sampleSeed": seed,
            "nPool": n_pool,
            "segments": segs,
        }

    def subset(self, seg_ids: list[int]) -> Optional[dict]:
        """The draw()-shaped payload for an EXACT stored seg_id list, in the
        stored order — powers session resume (GET /api/session/active). The
        ORDER is load-bearing: the browser engine's item selection is
        deterministic over the segment array, so replaying the original draw
        verbatim reproduces the original item sequence. Returns None when any
        seg_id is missing from this bank (the bundle changed since the
        sitting started — not resumable)."""
        segs = [self._by_id.get(sid) for sid in seg_ids]
        if any(s is None for s in segs):
            return None
        return {
            **self.engine,
            "bundleUrl": self.bundle_url,
            "nPool": len(segs),
            "segments": segs,
        }

    def full(self, seed: int, exclude: Optional[set] = None) -> dict:
        """Full exposure-eligible bank for the frozen Precision profile.

        Manifest order is retained exactly; per-domain browser candidates are
        then stable-sorted by signal. Resume can reconstruct this payload from
        only the exclusion list after verifying ``manifest_sha256``.
        """
        exclude = exclude or set()
        segs = [seg for seg in self.segments if seg["segId"] not in exclude]
        return {
            **self.engine,
            "bundleUrl": self.bundle_url,
            "sampleSeed": seed,
            "nPool": len(segs),
            "segments": segs,
        }

    def example(self) -> dict:
        """A single representative IIIC segment for the in-context tutorial (the
        spectrogram + red-box walkthrough), in the same shape as draw() so the
        client builds a minimal Bundle without fetching the full manifest."""
        seg = next((s for s in self.segments if s.get("testClass") != "spike"),
                   self.segments[0] if self.segments else None)
        return {
            **self.engine,
            "bundleUrl": self.bundle_url,
            "segments": [seg] if seg is not None else [],
        }

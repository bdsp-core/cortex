"""Server-side per-session question-bank draw (goal 3).

The full ~35k bank's signal index — the bundle ``manifest.json`` that
``prepare_web_bundle.py`` emits — is loaded ONCE at boot (read-only, shared).
Each session start draws a seeded, class-balanced, difficulty-stratified subset
(~``target`` segments), excluding the participant's recently-seen segments
(spacing, goal 5). The browser SMC engine runs over the returned subset exactly
as before, so the client never downloads the 35k manifest and per-question
selection stays bounded by the subset size, independent of bank size.

This is a faithful Python port of ``apps/web/src/sampleSession.ts`` operating on
the same web manifest schema (``patternClass`` groups, difficulty = the
segment's signal on its OWN true task, 8 contiguous difficulty bins drawn
round-robin). Pure-Python ``random.Random(seed)`` makes the draw reproducible
from the stored ``sample_seed``.
"""
from __future__ import annotations

import json
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
    "corrT", "nParticles", "perDomainCap",
)


class SessionBank:
    """Loads one web bundle manifest and draws per-session subsets from it."""

    def __init__(self, manifest_path: Path | str, bundle_url: str):
        self.bundle_url = bundle_url
        manifest = json.loads(Path(manifest_path).read_text())
        self.version = manifest.get("version")
        self.engine = {k: manifest[k] for k in _ENGINE_KEYS if k in manifest}
        self.segments: list[dict] = manifest["segments"]
        words = manifest["taskPatternWords"]
        self._word_idx = {w: i for i, w in enumerate(words)}
        # Group segment indices by pattern class once, at load.
        self._by_class: dict[str, list[int]] = {}
        for i, seg in enumerate(self.segments):
            self._by_class.setdefault(seg["patternClass"], []).append(i)

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

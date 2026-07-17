"""Credentialed-panel high-confidence training bank builder (D-INT-6).

A training item is a segment whose **credentialed-panel plurality label is
supported by > 3 concordant reads (>= 4 agree)**. The label panel is the full
credentialed-expert set (65 named experts, zero crowd). The **training label is
the panel's own plurality** — the credentialed panel is the authority, so where
it disagrees with the pre-existing MANIFEST pattern_class (the hardest, lowest-
agreement segments) we defer to the panel. Spike is a detection task, so a
segment is a confident spike-domain item when the panel confidently agrees on
present-vs-absent.

The v15 cut ell* is UNTOUCHED — it stays calibrated on Super8 ∪ Bonobo. The cut
(a threshold on the latent skill scale) and the label (a per-segment fact) are
different objects, so widening the label panel does not change the cut. See
docs/TRAINER_INTEGRATION_PLAN.md D-INT-6.

Run to (re)build the committed artifacts under `data/trainer_bank/`:

    python3 -m trainer.bank            # from the repo root

Outputs:
  * data/trainer_bank/MANIFEST.json      — provenance + per-domain counts +
                                           exposure budget.
  * data/trainer_bank/training_items.csv — per-segment panel agreement (auditable).

The fast G0 test asserts the committed MANIFEST counts; a `slow`-marked test
rebuilds from raw data and checks they reproduce.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from trainer import domains

# ── criterion constants (D-INT-6) ─────────────────────────────────────────
CREDENTIALED_GROUPS = ("Super8", "Bonobo", "profiler_iiic", "New28",
                       "spikeed_expert", "centaur_iiic_expert")
AGREE_THRESHOLD = 4                      # > 3 concordant panel reads
SPIKE_POSITIVE_VALUES = frozenset({"1", "ied"})   # spike-present vote values
DEFAULT_C = 600                          # no-repeat within-domain window (D-INT-7)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_IN = os.path.join(_REPO, "data", "production_bank", "MANIFEST.json")
_LABELS_IN = os.path.join(_REPO, "data", "labels", "labels.csv")
_RATERS_IN = os.path.join(_REPO, "data", "labels", "raters.csv")
_OUT_DIR = os.path.join(_REPO, "data", "trainer_bank")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel_rater_ids(raters_csv: str = _RATERS_IN) -> set[int]:
    """rater_ids whose `groups` cell names any credentialed expert group."""
    import pandas as pd
    rat = pd.read_csv(raters_csv, low_memory=False)
    g = rat["groups"].fillna("")
    mask = g.str.contains("|".join(CREDENTIALED_GROUPS))
    return set(rat.loc[mask, "rater_id"].astype(int))


def _iiic_plurality(counter: Counter):
    """(plurality_code, votes) with a deterministic canonical-order tie-break."""
    if not counter:
        return None, 0
    order = {d.pattern_class: d.index for d in domains.DOMAINS}
    cls, votes = max(counter.items(), key=lambda kv: (kv[1], -order[kv[0]]))
    return domains.code_for_pattern_class(cls), votes


def load_panel_votes(manifest_in=_MANIFEST_IN, labels_in=_LABELS_IN,
                     raters_in=_RATERS_IN):
    """Scan the vote table (credentialed panel only, production segments only)
    and return (prod, spike_pos, spike_neg, iiic_votes, manifest):
      prod       : {seg_id: (family, pattern_class)}
      spike_pos  : {seg_id: # panel spike-present votes}
      spike_neg  : {seg_id: # panel spike-absent votes}
      iiic_votes : {seg_id: Counter(pattern_class -> panel votes)}
    Shared by the G0 bank builder and the trainer bank_adapter (one scan)."""
    import pandas as pd

    with open(manifest_in) as fh:
        man = json.load(fh)
    prod = {int(s["seg_id"]): (s["family"], s["pattern_class"])
            for s in man["segments"]}
    panel = load_panel_rater_ids(raters_in)
    lab = pd.read_csv(labels_in, low_memory=False)
    lab = lab[lab.seg_id.isin(prod) & lab.rater_id.isin(panel)]

    iiic_classes = {d.pattern_class for d in domains.DOMAINS if d.family == "iiic"}
    spike_pos = defaultdict(int)
    spike_neg = defaultdict(int)
    iiic_votes: dict[int, Counter] = defaultdict(Counter)
    sp = lab[lab.label_type == "spike"]
    for sid, v in zip(sp.seg_id.values, sp.value.astype(str).values):
        if prod[sid][0] != "spike":
            continue
        (spike_pos if v in SPIKE_POSITIVE_VALUES else spike_neg)[sid] += 1
    pc = lab[lab.label_type == "pattern_class"]
    for sid, v in zip(pc.seg_id.values, pc.value.astype(str).values):
        if prod[sid][0] != "iiic":
            continue
        cls = v if v in iiic_classes else "other"
        iiic_votes[sid][cls] += 1
    return prod, spike_pos, spike_neg, iiic_votes, man


def compute(manifest_in=_MANIFEST_IN, labels_in=_LABELS_IN, raters_in=_RATERS_IN,
            threshold: int = AGREE_THRESHOLD):
    """Return (rows, per_domain, manifest_dict). `rows` = per-segment dicts;
    `per_domain` maps code → {testing, positives} where `testing` is bucketed by
    the MANIFEST class (the exam bank) and `positives` by the PANEL plurality
    label (the training authority)."""
    prod, spike_pos, spike_neg, iiic_votes, man = load_panel_votes(
        manifest_in, labels_in, raters_in)

    rows = []
    per_domain = {c: {"testing": 0, "positives": 0} for c in domains.CODES}
    for sid, (fam, pcl) in prod.items():
        manifest_code = domains.code_for_pattern_class(pcl)
        per_domain[manifest_code]["testing"] += 1
        if fam == "spike":
            p, n = spike_pos.get(sid, 0), spike_neg.get(sid, 0)
            reads = p + n
            bucket_code = "spike"                      # detection task, one domain
            panel_label = "spike_present" if p >= n else "spike_absent"
            plur_votes = max(p, n)                      # confident present/absent
        else:
            c = iiic_votes.get(sid, Counter())
            reads = sum(c.values())
            bucket_code, plur_votes = _iiic_plurality(c)
            panel_label = bucket_code if bucket_code else ""
        positive = plur_votes >= threshold             # panel label is confident
        if positive:
            per_domain[bucket_code]["positives"] += 1   # bucket by PANEL label
        rows.append({
            "seg_id": sid, "family": fam, "pattern_class": pcl,
            "manifest_code": manifest_code, "panel_reads": reads,
            "panel_label": panel_label, "panel_label_votes": plur_votes,
            "positive_eligible": int(positive),
        })
    rows.sort(key=lambda r: r["seg_id"])
    return rows, per_domain, man


def exposure_budget(per_domain, C: int = DEFAULT_C):
    """Per-domain no-repeat headroom against the C-window. A domain enters the
    D-INT-8 least-recently-seen fallback only if its confident-positive pool
    <= C (the trainer also draws confident negatives one-vs-rest, so the true
    supply is larger — positives are the scarce input)."""
    out = {}
    for code, d in per_domain.items():
        pool = d["positives"]
        out[code] = {
            "positive_pool": pool, "C": C, "headroom": pool - C,
            "enters_fallback_under_count_only": pool <= C,
        }
    return out


def build(out_dir: str = _OUT_DIR, threshold: int = AGREE_THRESHOLD) -> dict:
    import csv

    rows, per_domain, man = compute(threshold=threshold)
    os.makedirs(out_dir, exist_ok=True)

    items_path = os.path.join(out_dir, "training_items.csv")
    with open(items_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_pos = sum(d["positives"] for d in per_domain.values())
    total_test = sum(d["testing"] for d in per_domain.values())
    manifest = {
        "schema_version": "trainer_bank_v1",
        "decision": "D-INT-6",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "criterion": ("credentialed-panel plurality label supported by >= %d "
                      "concordant reads (> 3 agree); training label = panel "
                      "plurality; spike = confident present/absent" % threshold),
        "label_panel_groups": list(CREDENTIALED_GROUPS),
        "label_panel_size": len(load_panel_rater_ids()),
        "agree_threshold": threshold,
        "cut_note": ("v15 cut ell* UNCHANGED (Super8 union Bonobo); the label "
                     "panel and the cut panel are deliberately distinct objects"),
        "K": domains.K,
        "codes": list(domains.CODES),
        "per_domain": per_domain,
        "totals": {"testing": total_test, "training_positives": total_pos,
                   "retained_pct": round(100.0 * total_pos / total_test, 2)},
        "exposure_budget": exposure_budget(per_domain),
        "provenance": {
            "manifest_in": os.path.relpath(_MANIFEST_IN, _REPO),
            "manifest_sha256": _sha256(_MANIFEST_IN),
            "raters_in": os.path.relpath(_RATERS_IN, _REPO),
            "raters_sha256": _sha256(_RATERS_IN),
            "labels_in": os.path.relpath(_LABELS_IN, _REPO),
            "n_production_segments": len(rows),
        },
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    m = build()
    print("built trainer bank →", _OUT_DIR)
    print("label panel size:", m["label_panel_size"], "| threshold >=", m["agree_threshold"])
    print("totals:", m["totals"])
    for code in domains.CODES:
        d = m["per_domain"][code]
        flag = "  <-- <= 1000" if d["positives"] <= 1000 else ""
        print(f"  {code:5s} testing={d['testing']:6d}  positives={d['positives']:6d}{flag}")

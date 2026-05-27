"""Phase 6 — invariant drift-guards (gating the audit doc).

The bulk of Phase 6 is the documented audit at
`docs/INVARIANT_AUDIT.md` (referenced + signed-off by the user
2026-05-19). This file adds the targeted programmatic guards the plan
explicitly requests:

  * §3 — the two in-repo `fit_sdt_per_domain.py` copies (under
    `pipeline/reference_calibration/` and `pipeline/_calib_work/src/`)
    must remain byte-equivalent. The plan: "add a test asserting it
    (real consistency bug if they drift)."
  * Audit-doc consistency: `docs/INVARIANT_AUDIT.md` exists, covers
    all 9 invariants, and records the 4 signed-off deviations
    (§6 GRAY_ZONE_DELTA N/A, §7 EXPERTS wording clarified, §8 T_TOL
    N/A, §9 r_ℓ updated to the K=7 derived value).

Other invariants in the audit are already programmatically gated by
existing tests (Phase-3 byte-identity for the reference fitters;
Phase-4 LAPSE_RATE single-source via `core`; Phase-3 v13 / Phase-4
v13 ℓ* drift-guards; Phase-5 D3 shared-lineage). Avoiding duplication.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUDIT = REPO / "docs" / "INVARIANT_AUDIT.md"
_FITSDT_A = (REPO / "pipeline" / "reference_calibration"
             / "fit_sdt_per_domain.py")
_FITSDT_B = (REPO / "pipeline" / "_calib_work" / "src"
             / "fit_sdt_per_domain.py")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def test_fit_sdt_two_copies_byte_equivalent():
    """Plan §3: the two in-repo fit_sdt_per_domain.py copies (carried
    verbatim from the methodology reference) MUST stay byte-identical.
    A drift between them = a real consistency bug — different lapse
    NLL / fit logic in two paths that the calibration pipeline reads.

    _FITSDT_B lives under pipeline/_calib_work/src/ — a derived path
    produced by running pipeline/run_unified_calibration.py (hours of
    runtime). Skip cleanly when it hasn't been built yet; the in-repo
    md5 pin on _FITSDT_A is the still-active drift guard."""
    import pytest
    assert _FITSDT_A.is_file(), f"{_FITSDT_A} (in-repo reference) missing"
    if not _FITSDT_B.is_file():
        pytest.skip(
            f"{_FITSDT_B.relative_to(REPO)} not produced yet "
            "(run pipeline/run_unified_calibration.py first)")
    a, b = _md5(_FITSDT_A), _md5(_FITSDT_B)
    assert a == b, (
        f"fit_sdt_per_domain.py drifted between copies:\n"
        f"  {_FITSDT_A.relative_to(REPO)}  md5={a}\n"
        f"  {_FITSDT_B.relative_to(REPO)}  md5={b}")
    # also pinned to the reference md5 the Phase-3 byte-identity test
    # asserts (defence in depth — if both copies drift in lock-step,
    # this check still catches it)
    assert a == "6b90d59dcd0aaa878f9a52802254044b"


def test_invariant_audit_doc_covers_all_invariants_and_deviations():
    """The Phase-6 audit is the gate; this asserts the doc is present
    and records all 9 invariants + the 4 signed-off deviations (so the
    doc can't silently lose anchors)."""
    assert AUDIT.is_file(), "docs/INVARIANT_AUDIT.md missing"
    txt = AUDIT.read_text()
    # 9 invariants by their h2 headings (## 1. … ## 9.)
    for i in range(1, 10):
        assert f"## {i}." in txt, (
            f"INVARIANT_AUDIT.md missing '## {i}.' section")
    # 4 documented deviations (signed off 2026-05-19)
    for anchor in ("DEVIATION", "CLARIFICATION", "STALE",
                   "Mode-B", "raters.csv", "0.367"):
        assert anchor in txt, (
            f"INVARIANT_AUDIT.md missing anchor {anchor!r}")
    # the gate-satisfied sentence
    assert "Phase-6 gate satisfied" in txt


def test_phase6_passing_invariants_grounded_in_actual_code():
    """Spot-check the 5 PASS invariants point at code that ACTUALLY
    defines them (so the audit doc doesn't lose touch with reality):
    LOGIT_TO_PROBIT=1.0/1.7 at the cited line, LAPSE_RATE=0.025,
    EXPERT_TRAIN_FRAC=0.70."""
    src = _FITSDT_A.read_text().splitlines()
    assert src[37].strip() == "LOGIT_TO_PROBIT = 1.0 / 1.7", (
        "fit_sdt_per_domain.py:38 LOGIT_TO_PROBIT literal moved/edited")
    import sys
    if str(REPO / "engine") not in sys.path:
        sys.path.insert(0, str(REPO / "engine"))
    import core
    assert core.LAPSE_RATE == 0.025
    yref = (REPO / "pipeline" / "reference_calibration"
            / "youden_sigma_star_ref.py").read_text()
    assert "EXPERT_TRAIN_FRAC = 0.70" in yref

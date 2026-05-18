"""Phase-3 orchestrator: reference-faithful 7-task calibration on the
unified corpus.

The reference scripts are carried BYTE-IDENTICAL into
`pipeline/reference_calibration/` and are NEVER edited. This orchestrator
only controls the DATA ENTRY POINT (staging working dirs / argv) so the
verbatim scripts resolve their relative-path expectations:

  * fit_main_effects.py — uses DATA_DIR = Path("data/prepared") relative to
    CWD, and its DATASETS list IS exactly our 7 tasks (combined_spike +
    6 sparcnet_*). We invoke it as a subprocess with cwd = the staging
    root `pipeline/_calib_work/` which contains `data/prepared/`. Fully
    verbatim, no import gymnastics.

  * fit_sdt_per_domain.py — uses PREPARED = __file__.parent.parent /
    "data"/"prepared". Its module-level DOMAINS list rejects
    `combined_spike` ONLY inside main()'s argv guard; the per-domain
    function `fit_domain(domain)` is domain-name-agnostic. To fit ALL 7
    tasks (including combined_spike) WITHOUT editing the file, we place a
    byte-identical copy at `pipeline/_calib_work/src/fit_sdt_per_domain.py`
    (so PREPARED resolves to the staging root's data/prepared) and import
    `fit_domain` from it, calling fit_domain(task) directly per task. This
    bypasses only main()'s restrictive DOMAINS guard — the numerical body
    (probit_lapse_nll, fit_rater, LAMBDA=0.025, LOGIT_TO_PROBIT=1/1.7) is
    executed verbatim.

  * run_youden_calibration.py — uses _ENGINE_INPUTS = __file__.parent /
    "data"/"engine_inputs". Carried byte-identical; we stage its two
    inputs (cross_domain_rater_matrix.csv, sdt_fits.csv) into a sibling
    `data/engine_inputs/` next to the carried copy and invoke it as a
    subprocess. Only the two input paths are redirected.

Steps a..i per the Phase-3 spec.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
PIPELINE = REPO / "pipeline"
REF_CALIB = PIPELINE / "reference_calibration"
WORK = PIPELINE / "_calib_work"
WORK_PREPARED = WORK / "data" / "prepared"
WORK_SRC = WORK / "src"

LABELS_CSV = REPO / "data" / "labels" / "labels.csv"
RATERS_CSV = REPO / "data" / "labels" / "raters.csv"
ENGINE_INPUTS = REPO / "data" / "engine_inputs"
CROSSWALK_CSV = (
    REPO.parent / "name_crosswalk_audit_v3.0.csv"
)
CALIB_DIR = REPO / "calibration"

ALL_TASKS = [
    "combined_spike",
    "sparcnet_sz",
    "sparcnet_lpd",
    "sparcnet_gpd",
    "sparcnet_lrda",
    "sparcnet_grda",
    "sparcnet_iic",
]
# task name -> youden domain key (run_youden_calibration DOMAINS)
TASK_TO_DOMAIN = {
    "sparcnet_sz": "sz",
    "sparcnet_lpd": "lpd",
    "sparcnet_gpd": "gpd",
    "sparcnet_lrda": "lrda",
    "sparcnet_grda": "grda",
    "sparcnet_iic": "iic",
}
IIIC_DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]

# Centaur 4-expert gold panel (independent IIIC expert panel, step h / D7)
CENTAUR_GOLD_RATER_IDS = {97, 99000001, 99000002, 99000003}

sys.path.insert(0, str(PIPELINE))
from build_calibration_inputs import build_all  # noqa: E402


def _banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ───────────────────────────── step a ──────────────────────────────────────
def step_a_build_inputs() -> dict:
    _banner("STEP a — build reference-schema inputs (7 tasks)")
    WORK_PREPARED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = build_all(LABELS_CSV, RATERS_CSV, WORK_PREPARED, tasks=ALL_TASKS)
    dt = time.time() - t0
    print(f"  build time: {dt:.1f}s")
    print(
        f"  {'task':16s} {'n_obs':>9s} {'n_case':>7s} "
        f"{'n_rater':>7s} {'pos_rate':>9s}"
    )
    for task in ALL_TASKS:
        v = res[task]
        print(
            f"  {v['task']:16s} {v['n_obs']:>9d} {v['n_case']:>7d} "
            f"{v['n_rater']:>7d} {v['pos_rate']:>9.4f}"
        )
    print(f"  CHECKPOINT a: 7/7 tasks staged @ {WORK_PREPARED}")
    return res


# ───────────────────────────── step b ──────────────────────────────────────
def step_b_fit_main_effects() -> dict:
    _banner("STEP b — carried fit_main_effects.py (VERBATIM, all 7 tasks)")
    # fit_main_effects.py: DATA_DIR=Path("data/prepared") relative to CWD;
    # its DATASETS list IS our 7 tasks. Run verbatim with cwd=WORK.
    script = REF_CALIB / "fit_main_effects.py"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(WORK),
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"fit_main_effects.py failed (rc={proc.returncode})"
        )
    summaries = {}
    for task in ALL_TASKS:
        sj = WORK_PREPARED / f"{task}_fit" / "summary.json"
        s = json.load(open(sj))
        summaries[task] = s
        print(
            f"  {task:16s} sigma_c={s['sigma_c']:.4f} "
            f"sigma_l={s['sigma_l']:.4f} iters={s['iterations']:>3d} "
            f"n_obs={s['n_obs']}"
        )
    print(f"  fit_main_effects total wall time: {dt:.1f}s")
    print(f"  CHECKPOINT b: 7/7 Rasch fits written")
    return summaries


# ───────────────────────────── step c ──────────────────────────────────────
def _load_fit_domain():
    """Place a byte-identical copy of fit_sdt_per_domain.py at
    WORK/src/ so PREPARED resolves to WORK/data/prepared, then import
    `fit_domain` (verbatim per-domain logic; bypasses only main()'s
    DOMAINS argv guard)."""
    WORK_SRC.mkdir(parents=True, exist_ok=True)
    src = REF_CALIB / "fit_sdt_per_domain.py"
    dst = WORK_SRC / "fit_sdt_per_domain.py"
    shutil.copy2(src, dst)
    assert _md5(src) == _md5(dst), "staged fit_sdt copy not byte-identical"
    spec = importlib.util.spec_from_file_location(
        "_staged_fit_sdt_per_domain", dst
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Sanity: PREPARED must point at the staging prepared dir.
    assert mod.PREPARED == WORK_PREPARED, (
        f"PREPARED resolved to {mod.PREPARED}, expected {WORK_PREPARED}"
    )
    return mod


def step_c_fit_sdt() -> dict:
    _banner(
        "STEP c — carried fit_sdt_per_domain.fit_domain (VERBATIM, 7 tasks)"
    )
    mod = _load_fit_domain()
    print(
        f"  carried invariants: LAMBDA={mod.LAMBDA} "
        f"LOGIT_TO_PROBIT={mod.LOGIT_TO_PROBIT!r} "
        f"MIN_FIT_TRIALS={mod.MIN_FIT_TRIALS}"
    )
    out = {}
    for task in ALL_TASKS:
        t0 = time.time()
        mod.fit_domain(task)  # verbatim per-domain logic
        dt = time.time() - t0
        fits = pd.read_csv(WORK_PREPARED / f"{task}_sdt_fits.csv")
        conv = fits[fits["converged"]]
        med = conv["sigma"].median() if len(conv) else float("nan")
        out[task] = {
            "n_total": int(len(fits)),
            "n_converged": int(fits["converged"].sum()),
            "sigma_median": float(med),
            "seconds": round(dt, 1),
        }
        print(
            f"  {task:16s} converged={out[task]['n_converged']:>4d}/"
            f"{out[task]['n_total']:<4d} sigma_median={med:.4f} "
            f"({dt:.1f}s)"
        )
    print(f"  CHECKPOINT c: 7/7 SDT fits written")
    return out


# ───────────────────────────── step d ──────────────────────────────────────
def step_d_assemble_sdt_fits() -> Path:
    _banner("STEP d — assemble unified data/engine_inputs/sdt_fits.csv")
    # Preserve the existing carried sdt_fits.csv (old corpus) as legacy.
    cur = ENGINE_INPUTS / "sdt_fits.csv"
    legacy = ENGINE_INPUTS / "sdt_fits.legacy_oldcorpus.csv"
    if cur.exists() and not legacy.exists():
        shutil.copy2(cur, legacy)
        print(f"  preserved old corpus -> {legacy.name}")
    elif legacy.exists():
        print(f"  legacy already present ({legacy.name}); not overwriting")

    frames = []
    for task, dom in TASK_TO_DOMAIN.items():
        df = pd.read_csv(WORK_PREPARED / f"{task}_sdt_fits.csv")
        df.insert(0, "domain", dom)
        frames.append(df)
    unified = pd.concat(frames, ignore_index=True)
    unified.to_csv(cur, index=False)
    print(
        f"  wrote {cur}  ({len(unified)} rows, "
        f"domains={sorted(unified['domain'].unique())})"
    )

    # spike sdt_fits (separate; combined_spike not an IIIC domain)
    spike = pd.read_csv(WORK_PREPARED / "combined_spike_sdt_fits.csv")
    spike.insert(0, "domain", "spike")
    spike_out = ENGINE_INPUTS / "sdt_fits.spike.csv"
    spike.to_csv(spike_out, index=False)
    print(f"  wrote {spike_out}  ({len(spike)} rows)")
    print(f"  CHECKPOINT d: unified sdt_fits.csv assembled (6 IIIC domains)")
    return cur


# ───────────────────────────── step e ──────────────────────────────────────
def _alias_to_canonical(raters: pd.DataFrame) -> dict[str, str]:
    import ast

    a2c: dict[str, str] = {}
    for _, row in raters.iterrows():
        cn = str(row["canonical_name"]).strip()
        a2c.setdefault(cn, cn)
        raw = row.get("aliases")
        if isinstance(raw, str) and raw.strip():
            try:
                aliases = ast.literal_eval(raw)
            except Exception:
                aliases = []
            for a in aliases:
                a2c.setdefault(str(a).strip().strip("'").strip(), cn)
    return a2c


def step_e_reconcile_matrix(sdt_fits_csv: Path) -> dict:
    _banner(
        "STEP e — reconcile Q2 candidate pool (29) to unified canonical "
        "names"
    )
    raters = pd.read_csv(RATERS_CSV, low_memory=False)
    uni_canon = set(raters["canonical_name"].astype(str).str.strip())
    a2c = _alias_to_canonical(raters)

    crosswalk = pd.read_csv(CROSSWALK_CSV)
    # crosswalk: confirmed_sparcnet_name -> confirmed_canonical_name
    cw_sn2cn: dict[str, str] = {}
    for _, r in crosswalk.iterrows():
        sn = r.get("confirmed_sparcnet_name")
        cn = r.get("confirmed_canonical_name")
        if isinstance(sn, str) and isinstance(cn, str):
            cw_sn2cn.setdefault(sn.strip(), cn.strip())

    src_matrix = ENGINE_INPUTS / "cross_domain_rater_matrix.csv"
    m = pd.read_csv(src_matrix)

    fit_names = set(
        pd.read_csv(sdt_fits_csv)["rater_name"].astype(str).str.strip()
    )

    rows = []
    unresolved = []
    applied = []
    for _, row in m.iterrows():
        sn = str(row["confirmed_sparcnet_name"]).strip()
        cn = str(row["confirmed_canonical_name"]).strip()
        method = None
        resolved = None
        # 1. sparcnet name directly matches unified canonical (= fit name)
        if sn in uni_canon:
            resolved, method = sn, "sn_direct"
        # 2. via unified alias on sparcnet name
        elif sn in a2c:
            resolved, method = a2c[sn], "sn_via_unified_alias"
        # 3. crosswalk sparcnet->canonical, then unified
        elif sn in cw_sn2cn and cw_sn2cn[sn] in uni_canon:
            resolved, method = cw_sn2cn[sn], "crosswalk_sn2cn"
        elif sn in cw_sn2cn and cw_sn2cn[sn] in a2c:
            resolved, method = a2c[cw_sn2cn[sn]], "crosswalk_sn2cn_alias"
        # 4. matrix canonical directly / via alias
        elif cn in uni_canon:
            resolved, method = cn, "cn_direct"
        elif cn in a2c:
            resolved, method = a2c[cn], "cn_via_unified_alias"

        in_fits = resolved in fit_names if resolved else False
        if resolved is None:
            unresolved.append({"matrix_sparcnet": sn, "matrix_canonical": cn})
            print(
                f"  UNRESOLVED  sparcnet={sn!r} canonical={cn!r}"
            )
            continue
        if method != "sn_direct" or sn != resolved:
            applied.append(
                {
                    "matrix_sparcnet": sn,
                    "matrix_canonical": cn,
                    "resolved_unified_canonical": resolved,
                    "method": method,
                    "in_new_sdt_fits": in_fits,
                }
            )
            print(
                f"  RECONCILED  {sn!r:30s} / {cn!r:24s} -> "
                f"{resolved!r:24s} [{method}] in_fits={in_fits}"
            )
        # Reconciled matrix: confirmed_sparcnet_name MUST equal the
        # unified canonical name so the carried run_youden_calibration.py
        # (which keys candidates by confirmed_sparcnet_name against
        # sdt_fits.rater_name == canonical_name) joins correctly.
        new_row = dict(row)
        new_row["confirmed_canonical_name"] = resolved
        new_row["confirmed_sparcnet_name"] = resolved
        rows.append(new_row)

    out = pd.DataFrame(rows)
    dst = ENGINE_INPUTS / "cross_domain_rater_matrix.csv"
    src_legacy = ENGINE_INPUTS / "cross_domain_rater_matrix.q2locked.csv"
    if not src_legacy.exists():
        shutil.copy2(src_matrix, src_legacy)
        print(f"  preserved Q2-locked source -> {src_legacy.name}")
    out.to_csv(dst, index=False)
    print(
        f"  wrote reconciled matrix ({len(out)}/{len(m)} resolved) -> "
        f"{dst}"
    )
    n_in_fits = sum(
        1
        for _, r in out.iterrows()
        if str(r["confirmed_sparcnet_name"]).strip() in fit_names
    )
    print(
        f"  CHECKPOINT e: {len(out)}/{len(m)} candidates resolved; "
        f"{n_in_fits} present in new sdt_fits.csv; "
        f"{len(unresolved)} unresolved"
    )
    return {
        "n_candidates": int(len(m)),
        "n_resolved": int(len(out)),
        "n_in_new_sdt_fits": int(n_in_fits),
        "unresolved": unresolved,
        "applied_mappings": applied,
    }


# ───────────────────────────── step f ──────────────────────────────────────
def step_f_run_youden() -> dict:
    _banner("STEP f — carried run_youden_calibration.py (VERBATIM)")
    # The carried script resolves _ENGINE_INPUTS = __file__.parent /
    # data/engine_inputs. Stage the two redirected inputs there.
    script = REF_CALIB / "run_youden_calibration.py"
    stage_ei = REF_CALIB / "data" / "engine_inputs"
    stage_ei.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ENGINE_INPUTS / "sdt_fits.csv", stage_ei / "sdt_fits.csv"
    )
    shutil.copy2(
        ENGINE_INPUTS / "cross_domain_rater_matrix.csv",
        stage_ei / "cross_domain_rater_matrix.csv",
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REF_CALIB),
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError("run_youden_calibration.py failed")
    out_json = REF_CALIB / "youden_ell_star.json"
    payload = json.load(open(out_json))
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    saved = CALIB_DIR / "youden_ell_star.json"
    shutil.copy2(out_json, saved)
    print(f"  saved -> {saved}")
    print(f"  CHECKPOINT f: 6 IIIC ell* computed")
    return payload


# ───────────────────────────── step g ──────────────────────────────────────
def step_g_spike_sigma_star() -> dict:
    _banner("STEP g — spike σ* (reference 70/30 train split, verbatim Youden)")
    from reference_calibration.youden_sigma_star_ref import (  # noqa: E402
        EXPERT_TRAIN_FRAC,
        SEED,
        youden_sigma_star,
    )

    fits = pd.read_csv(WORK_PREPARED / "combined_spike_sdt_fits.csv")
    raters = pd.read_csv(RATERS_CSV, low_memory=False)
    name2exp = dict(
        zip(
            raters["canonical_name"].astype(str).str.strip(),
            raters["expertise_level"].astype(str),
        )
    )

    # converged & n_trials>=20
    f = fits[
        fits["converged"] & (fits["n_trials"] >= 20)
    ].copy()
    f["rater_name"] = f["rater_name"].astype(str).str.strip()
    f["expertise_level"] = f["rater_name"].map(name2exp).fillna("")

    is_expert = f["expertise_level"] == "expert"
    experts = f[is_expert]["sigma"].to_numpy(dtype=np.float64)
    nonexperts = f[~is_expert]["sigma"].to_numpy(dtype=np.float64)
    print(
        f"  converged & n>=20: {len(f)}  "
        f"experts={len(experts)}  non-experts={len(nonexperts)}"
    )

    # Reference 70/30 TRAIN split (train_val_split_and_fit.py:357-360):
    #   shuf = rng.permutation(ids); half = round(n*(1-FRAC)) -> val
    #   train = shuf[: n-half]
    rng = np.random.default_rng(SEED)
    e_idx = np.arange(len(experts))
    shuf_e = rng.permutation(e_idx)
    half_e = round(len(shuf_e) * (1 - EXPERT_TRAIN_FRAC))
    exp_train = experts[shuf_e[: len(shuf_e) - half_e]]
    # Non-experts: reference uses 50/50 train split (experienced/novice/
    # crowd blocks, lines 365-390). TRAIN pool only.
    n_idx = np.arange(len(nonexperts))
    shuf_n = rng.permutation(n_idx)
    half_n = len(shuf_n) // 2
    non_train = nonexperts[shuf_n[:half_n]]
    print(
        f"  TRAIN pool: experts={len(exp_train)} (70%), "
        f"non-experts={len(non_train)} (50%)"
    )

    sigma_star, J = youden_sigma_star(exp_train, non_train)
    ell_star = float(-np.log(sigma_star))
    print(
        f"  spike σ* = {sigma_star:.4f}  ℓ* = {ell_star:+.4f}  "
        f"Youden J = {J:.4f}"
    )
    print(
        f"  reference frozen spike σ* = 0.815  (ℓ* = "
        f"{-np.log(0.815):+.4f})  Δσ* = {sigma_star - 0.815:+.4f}"
    )
    print(f"  CHECKPOINT g: spike σ* computed")
    return {
        "sigma_star": float(sigma_star),
        "ell_star": ell_star,
        "youden_j": float(J),
        "n_expert_train": int(len(exp_train)),
        "n_non_expert_train": int(len(non_train)),
        "n_expert_total": int(len(experts)),
        "n_non_expert_total": int(len(nonexperts)),
        "reference_frozen_sigma_star": 0.815,
        "delta_vs_frozen": float(sigma_star - 0.815),
    }


# ───────────────────────────── step h ──────────────────────────────────────
def step_h_d7_independent_panel(youden_payload: dict) -> dict:
    _banner(
        "STEP h — D7 independent-panel reproduction (Centaur 4-expert gold)"
    )
    labels = pd.read_csv(LABELS_CSV, low_memory=False)
    raters = pd.read_csv(RATERS_CSV, low_memory=False)
    rid2name = dict(
        zip(
            raters["rater_id"].astype(int),
            raters["canonical_name"].astype(str).str.strip(),
        )
    )
    gold_names = {
        rid2name.get(rid) for rid in CENTAUR_GOLD_RATER_IDS
    }
    gold_names.discard(None)
    print(
        f"  Centaur gold panel rater_ids={sorted(CENTAUR_GOLD_RATER_IDS)} "
        f"-> names={sorted(gold_names)}"
    )

    # Recompute the 6 IIIC ell* using ONLY gold-panel raters as the expert
    # set vs the same non-expert pool (all non-candidate raters), keyed by
    # canonical name in the new sdt_fits.csv. This re-uses the carried
    # Youden machinery's empirical-CDF definition (identical to
    # run_youden_calibration._youden_one), applied with the gold panel as
    # the expert set. The numeric Youden logic is unchanged.
    sdt = pd.read_csv(ENGINE_INPUTS / "sdt_fits.csv")
    SIGMA_FLOOR = 0.025

    def youden_one(e, ne):
        e = np.sort(e[np.isfinite(e)])
        ne = np.sort(ne[np.isfinite(ne)])
        if e.size == 0 or ne.size == 0:
            return float("nan"), float("nan")
        grid = np.sort(np.unique(np.concatenate([e, ne])))
        F_e = np.searchsorted(e, grid, side="right") / e.size
        F_ne = np.searchsorted(ne, grid, side="right") / ne.size
        diff = F_ne - F_e
        idx = int(np.argmax(diff))
        return float(grid[idx]), float(diff[idx])

    # D7 re-scoped (decision 2026-05-18): the n=4 Centaur gold panel is too
    # thin for tight point-reproduction of ℓ*. Report it HONESTLY as a
    # consistency/direction check: (1) bootstrap the gold panel (B=2000,
    # resample gold experts AND the non-expert pool with replacement — same
    # bootstrap as run_youden_calibration._run_calibration) → gold ℓ* 95% CI;
    # (2) does the gold-ℓ* 95% CI OVERLAP the CV-ℓ* 95% CI (step f)?;
    # (3) across the 6 IIIC tasks, Spearman rank concordance + sign
    # concordance of (gold − CV). The n=4 limitation is stated explicitly.
    from scipy.stats import spearmanr  # noqa: E402

    BOOTSTRAP_N = 2000
    D7_SEED = 42

    def ci_overlap(a_lo, a_hi, b_lo, b_hi):
        lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
        return (lo <= hi), (float(lo), float(hi)) if lo <= hi else None

    out = {"per_task": {}}
    gold_vec, cv_vec = [], []
    for dom in IIIC_DOMAINS:
        d = sdt[(sdt["domain"] == dom)]
        d = d[
            (d["converged"].astype(str).str.lower() == "true")
            & (d["sigma"] > SIGMA_FLOOR)
        ].copy()
        d["rater_name"] = d["rater_name"].astype(str).str.strip()
        d["ell"] = -np.log(d["sigma"])
        is_gold = d["rater_name"].isin(gold_names)
        exp = d[is_gold]["ell"].to_numpy(dtype=np.float64)
        non = d[~is_gold]["ell"].to_numpy(dtype=np.float64)
        ell_star, J = youden_one(exp, non)

        # bootstrap gold-panel ℓ* (resample experts + non-experts w/ replace)
        rng = np.random.default_rng(D7_SEED + hash(dom) % 1000)
        boots = np.empty(BOOTSTRAP_N)
        for b in range(BOOTSTRAP_N):
            eb = rng.choice(exp, size=exp.size, replace=True) if exp.size else exp
            nb = rng.choice(non, size=non.size, replace=True) if non.size else non
            boots[b], _ = youden_one(eb, nb)
        bg = boots[np.isfinite(boots)]
        g_lo = float(np.percentile(bg, 2.5)) if bg.size else float("nan")
        g_hi = float(np.percentile(bg, 97.5)) if bg.size else float("nan")

        f_res = youden_payload["results"].get(dom, {})
        ref_ell = f_res.get("l_star", float("nan"))
        cv_lo = f_res.get("ci_low", float("nan"))
        cv_hi = f_res.get("ci_high", float("nan"))
        overlaps, ov_int = (
            ci_overlap(g_lo, g_hi, cv_lo, cv_hi)
            if all(np.isfinite([g_lo, g_hi, cv_lo, cv_hi]))
            else (False, None)
        )
        delta = (ell_star - ref_ell
                 if np.isfinite(ell_star) and np.isfinite(ref_ell)
                 else float("nan"))
        if np.isfinite(ell_star) and np.isfinite(ref_ell):
            gold_vec.append(ell_star)
            cv_vec.append(ref_ell)
        out["per_task"][dom] = {
            "ell_star_gold": float(ell_star),
            "gold_ci_low": g_lo, "gold_ci_high": g_hi,
            "youden_j_gold": float(J),
            "n_gold_expert": int(is_gold.sum()),
            "n_non_expert": int((~is_gold).sum()),
            "ell_star_cv_step_f": float(ref_ell)
            if np.isfinite(ref_ell) else None,
            "cv_ci_low": float(cv_lo) if np.isfinite(cv_lo) else None,
            "cv_ci_high": float(cv_hi) if np.isfinite(cv_hi) else None,
            "ci_95_overlap": bool(overlaps),
            "ci_overlap_interval": ov_int,
            "delta_point": float(delta) if np.isfinite(delta) else None,
        }
        print(
            f"  {dom:>5s}  ℓ*_gold={ell_star:+.4f} "
            f"CI[{g_lo:+.3f},{g_hi:+.3f}]  vs CV ℓ*={ref_ell:+.4f} "
            f"CI[{cv_lo:+.3f},{cv_hi:+.3f}]  overlap={overlaps}"
        )

    n_overlap = sum(
        1 for v in out["per_task"].values() if v["ci_95_overlap"])
    rho = (float(spearmanr(gold_vec, cv_vec).statistic)
           if len(gold_vec) >= 3 else float("nan"))
    sign_conc = (
        float(np.mean(np.sign(np.array(gold_vec) - np.array(cv_vec))
                      == np.sign(np.array(gold_vec) - np.array(cv_vec))))
        if gold_vec else float("nan"))
    # sign concordance of the SHIFT direction (is gold consistently higher?)
    shift = np.array(gold_vec) - np.array(cv_vec) if gold_vec else np.array([])
    frac_gold_higher = float(np.mean(shift > 0)) if shift.size else float("nan")
    out["summary"] = {
        "n_iiic_tasks": len(IIIC_DOMAINS),
        "n_ci95_overlap": int(n_overlap),
        "spearman_rho_gold_vs_cv": rho,
        "frac_tasks_gold_higher_than_cv": frac_gold_higher,
        "bootstrap_n": BOOTSTRAP_N,
        "interpretation": (
            "D7 is a consistency/direction check, NOT tight point "
            "reproduction: the Centaur gold panel is n=4 experts, so ℓ* is "
            "high-variance. Report = bootstrap-CI overlap + rank/shift "
            "concordance. A systematic gold>CV shift indicates the gold "
            "panel is stricter (more discriminating) than the CV-top-14 "
            "pool; this is a known limitation of the n=4 independent panel "
            "and is documented as such (not a tight-reproducibility claim)."
        ),
        "n4_limitation": (
            "Centaur gold panel = 4 experts (mbw/cal/matt/tianyu); "
            "bootstrap CIs are wide by construction."
        ),
    }
    print(
        f"  D7 summary: {n_overlap}/{len(IIIC_DOMAINS)} tasks gold-CI ∩ "
        f"CV-CI overlap | Spearman ρ(gold,CV)={rho:+.3f} | "
        f"gold>CV in {frac_gold_higher:.0%} of tasks (n=4 — directional)"
    )
    print(f"  CHECKPOINT h: D7 (bootstrap CI-overlap + concordance) complete")
    return out


# ───────────────────────────── step i ──────────────────────────────────────
def step_i_emit_cert_config_v12(
    youden_payload: dict,
    spike: dict,
    d7: dict,
    recon: dict,
    sdt_summary: dict,
) -> Path:
    _banner("STEP i — emit calibration/cert_config.yaml v12")
    src_cfg = REPO / "cert_config.yaml"
    with open(src_cfg) as fh:
        cfg = yaml.safe_load(fh)
    cfg["config_version"] = 12

    ell_block = {}
    for dom in IIIC_DOMAINS:
        r = youden_payload["results"].get(dom, {})
        # task name for this domain
        task = next(
            t for t, dd in TASK_TO_DOMAIN.items() if dd == dom
        )
        ell_block[task] = {
            "domain": dom,
            "ell_star": float(r.get("l_star", float("nan"))),
            "sigma_star": float(r.get("sigma_star", float("nan"))),
            "youden_j": float(r.get("youden_J", float("nan"))),
            "ci_low": float(r.get("ci_low", float("nan"))),
            "ci_high": float(r.get("ci_high", float("nan"))),
            "n_expert": int(r.get("n_expert", 0)),
            "n_non_expert": int(r.get("n_non_expert", 0)),
        }
    ell_block["combined_spike"] = {
        "domain": "spike",
        "ell_star": spike["ell_star"],
        "sigma_star": spike["sigma_star"],
        "youden_j": spike["youden_j"],
        "ci_low": None,
        "ci_high": None,
        "n_expert": spike["n_expert_train"],
        "n_non_expert": spike["n_non_expert_train"],
    }

    cfg["ell_star_unified_v12"] = {
        "tasks": ell_block,
        "provenance": {
            "labels_csv_sha256": _sha256(LABELS_CSV),
            "raters_csv_sha256": _sha256(RATERS_CSV),
            "method_iiic": youden_payload.get("method"),
            "method_iiic_description": youden_payload.get("description"),
            "panel_size_N": youden_payload.get("panel_size_N"),
            "bootstrap_n": youden_payload.get("bootstrap_n"),
            "seed": youden_payload.get("seed"),
            "sigma_floor": youden_payload.get("sigma_floor"),
            "method_spike": (
                "reference youden_sigma_star (verbatim) on 70/30 expert "
                "train split + 50/50 non-expert train split; "
                "EXPERT_TRAIN_FRAC=0.70, SEED=42"
            ),
            "spike_reference_frozen_sigma_star": 0.815,
            "spike_delta_vs_frozen": spike["delta_vs_frozen"],
            "carried_script_md5": {
                "fit_main_effects.py": _md5(
                    REF_CALIB / "fit_main_effects.py"
                ),
                "fit_sdt_per_domain.py": _md5(
                    REF_CALIB / "fit_sdt_per_domain.py"
                ),
                "run_youden_calibration.py": _md5(
                    REF_CALIB / "run_youden_calibration.py"
                ),
            },
            "candidate_pool": {
                "n_candidates": recon["n_candidates"],
                "n_resolved": recon["n_resolved"],
                "n_in_new_sdt_fits": recon["n_in_new_sdt_fits"],
                "unresolved": recon["unresolved"],
                "applied_mappings": recon["applied_mappings"],
            },
            "d7_independent_panel": d7,
            "sdt_fit_summary": sdt_summary,
            "note": (
                "Mode-A (Paper-1, Multi-AUROC) does NOT consume ell*; these "
                "feed the deployment runtime + Mode-B / Paper-2. The "
                "mode_b_legacy block below is retained unchanged."
            ),
        },
    }

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    dst = CALIB_DIR / "cert_config.yaml"
    header = (
        "# Certified configuration — Phase-3 unified recompute (v12).\n"
        "# COPY of repo-root cert_config.yaml (config_version bumped 11->12)\n"
        "# plus the new ell_star_unified_v12 block. mode_b_legacy preserved.\n"
        "# The repo-root cert_config.yaml is left UNTOUCHED.\n"
    )
    with open(dst, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
    print(f"  wrote {dst}  (config_version={cfg['config_version']})")
    assert "mode_b_legacy" in cfg, "mode_b_legacy lost!"
    print("  mode_b_legacy preserved: OK")
    print(f"  CHECKPOINT i: cert_config v12 emitted")
    return dst


def main() -> None:
    t_start = time.time()
    _banner("PHASE-3 UNIFIED REFERENCE-FAITHFUL CALIBRATION — START")
    step_a_build_inputs()
    step_b_fit_main_effects()
    sdt_summary = step_c_fit_sdt()
    sdt_fits_csv = step_d_assemble_sdt_fits()
    recon = step_e_reconcile_matrix(sdt_fits_csv)
    youden_payload = step_f_run_youden()
    spike = step_g_spike_sigma_star()
    d7 = step_h_d7_independent_panel(youden_payload)
    cfg_path = step_i_emit_cert_config_v12(
        youden_payload, spike, d7, recon, sdt_summary
    )
    dt = time.time() - t_start
    _banner(f"PHASE-3 COMPLETE in {dt / 60:.1f} min  ->  {cfg_path}")


if __name__ == "__main__":
    main()

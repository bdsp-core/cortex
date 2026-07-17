"""Canonical K=7 domain registry — one source of truth for task naming.

Three naming conventions coexist in this repo, and the trainer integration has
to bridge all three. Before G0 they were duplicated as literals across ~15
files with no central registry (see the slowing-domain audit); this module is
that registry. New trainer-integration code imports from here; a G0 test
(`tests/test_trainer_g0_ell_star.py`) cross-checks it against the production
`cert_config` keys so the two can never silently drift.

The three conventions, in the canonical order used by the policy + ℓ* arrays:

  idx | code  | pattern_class | anon    | cert_key        | family
  ----+-------+---------------+---------+-----------------+-------
   0  | spike | spike         | domain1 | combined_spike  | spike
   1  | sz    | seizure       | domain2 | sparcnet_sz     | iiic
   2  | lpd   | lpd           | domain3 | sparcnet_lpd    | iiic
   3  | gpd   | gpd           | domain4 | sparcnet_gpd    | iiic
   4  | lrda  | lrda          | domain5 | sparcnet_lrda   | iiic
   5  | grda  | grda          | domain6 | sparcnet_grda   | iiic
   6  | iic   | other         | domain7 | sparcnet_iic    | iiic

  * `code`          — production task code used by the policy + ℓ* loader
                      (`scripts/cortex_policy_k7._KEY_FOR_CODE_K7`). Uses `sz`
                      and `iic`.
  * `pattern_class` — the label used in `data/production_bank/MANIFEST.json`
                      and the vote tables. Uses `seizure` and `other`.
  * `anon`          — the anonymized code the frozen `trainer_rd/` banks use
                      (`domain1`..`domain7`).
  * `cert_key`      — the YAML key under `cert_config.yaml`'s
                      `ell_star_unified_v1x.tasks`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    index: int          # 0..6, canonical position (matches ℓ* array order)
    code: str           # production task code: spike, sz, lpd, gpd, lrda, grda, iic
    pattern_class: str  # MANIFEST.json / vote-table class: spike, seizure, ..., other
    anon: str           # trainer_rd anonymized code: domain1..domain7
    cert_key: str       # cert_config YAML task key: combined_spike / sparcnet_*
    family: str         # "spike" | "iiic"
    label: str          # human-readable label


DOMAINS: tuple[Domain, ...] = (
    Domain(0, "spike", "spike",   "domain1", "combined_spike", "spike", "Spike / IED"),
    Domain(1, "sz",    "seizure", "domain2", "sparcnet_sz",    "iiic",  "Seizure"),
    Domain(2, "lpd",   "lpd",     "domain3", "sparcnet_lpd",   "iiic",  "LPD"),
    Domain(3, "gpd",   "gpd",     "domain4", "sparcnet_gpd",   "iiic",  "GPD"),
    Domain(4, "lrda",  "lrda",    "domain5", "sparcnet_lrda",  "iiic",  "LRDA"),
    Domain(5, "grda",  "grda",    "domain6", "sparcnet_grda",  "iiic",  "GRDA"),
    Domain(6, "iic",   "other",   "domain7", "sparcnet_iic",   "iiic",  "Other / IIC"),
)

K: int = len(DOMAINS)

CODES: tuple[str, ...] = tuple(d.code for d in DOMAINS)
PATTERN_CLASSES: tuple[str, ...] = tuple(d.pattern_class for d in DOMAINS)
ANON_CODES: tuple[str, ...] = tuple(d.anon for d in DOMAINS)
CERT_KEYS: tuple[str, ...] = tuple(d.cert_key for d in DOMAINS)

_BY_CODE = {d.code: d for d in DOMAINS}
_BY_PATTERN_CLASS = {d.pattern_class: d for d in DOMAINS}
_BY_ANON = {d.anon: d for d in DOMAINS}
_BY_CERT_KEY = {d.cert_key: d for d in DOMAINS}


def by_index(index: int) -> Domain:
    return DOMAINS[index]


def by_code(code: str) -> Domain:
    return _BY_CODE[code]


def by_pattern_class(pattern_class: str) -> Domain:
    return _BY_PATTERN_CLASS[pattern_class]


def by_anon(anon: str) -> Domain:
    return _BY_ANON[anon]


def by_cert_key(cert_key: str) -> Domain:
    return _BY_CERT_KEY[cert_key]


def code_for_pattern_class(pattern_class: str) -> str:
    """MANIFEST/vote pattern_class → production task code (e.g. 'seizure'→'sz')."""
    return _BY_PATTERN_CLASS[pattern_class].code

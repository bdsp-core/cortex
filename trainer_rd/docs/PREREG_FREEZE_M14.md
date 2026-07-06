# PREREG_FREEZE_M14 — analysis freeze manifest (A5, 2026-06-12)

Purpose: fix the confirmatory/exploratory boundary BEFORE further analysis;
this file is the local equivalent of an OSF registration (post externally
when the user is ready — content is self-contained).

## Confirmatory (pre-specified in EXTSET_SAP_M13.md v2 BEFORE fits; results
in PROJECT_MEMORY M13.3): P1, P1b, P2, P2b, P3, P4 — all PASS.

## Exploratory / post-hoc (M13.2–M14, labeled as such in any manuscript):
- F40 signal-circularity discovery + LOO protocol (data-driven, then frozen).
- M5/M5b joint-softmax comparison (model refinement after seeing marginals).
- Qscore volume-confound partial correlation; tier known-groups failure.
- Specification curve (robustness exhibit, not hypothesis test):
  task1 ρ ∈ [0.836, 0.904] (6 branches), task2 ρ ∈ [0.346, 0.736]
  (36 branches), all positive.
- Disattenuated P2b = 1.00 (capped), P2 = 0.648 (split-half replay
  reliabilities 0.781/0.776; `data_extset_disatten.npz`).
- Shadow uplift ×6.0; pilot power table (design inputs, not claims).

## Code/document hashes at freeze (SHA-256)

| file | sha256 |
|---|---|
| training/extset_adapter.py | e02cf46fb69e59e145b4a05628ace462a61888bb175ff22a7151319cfcb10834 |
| studies/study_extset_link.py | 8985cfd58d6c2d55411625c1b9d2097bc052fc8ff93881091b3919811ab41fc9 |
| studies/study_extset_replay.py | 28311bdb993cc735100e1fae141985fa93d4f081acd21aa3f6e19e54cd3626b4 |
| studies/study_extset_xdomain.py | b822f4921034afb070affd74bc3191f1a50b2c64c729045e662aa1fc8c5de383 |
| studies/study_extset_stress.py | 23d77494ebf14d61af75ad98761b9da6af77782fdb53e2518681a060daccf7a0 |
| studies/study_extset_m5.py | 2b2ee49f05b7dec63eaf0a2896fcef7b58f83d99787406c53e40a4a5da7452e9 |
| studies/study_extset_bayes.py | 1535af625e23a31c9fc95d8f13c6640d3d1b4fc69ec6eeb43cbdd10e9e49a9ba |
| studies/study_extset_speccurve.py | 7bf72d33424f355442673da7fd842d80f0ac934fb94845810af422a8e874f01f |
| studies/study_extset_shadow.py | 04ae5627dfdcac5c419f2b01b59767cf4f62fa09496ac6923b470a672176dfa4 |
| studies/study_pilot_power.py | f5a8616f5475872dd57903b5c597c21003a051e727e1a3ebc0e18ede7a3f3f21 |
| docs/EXTSET_SAP_M13.md | 8a3c4d4c0775109d66b24a62090d21f88e556ae2da24adbe6de15860c65d733b |

Data files are the *_scrubbed.csv release + extset_task1_*.csv as received
2026-06-12 (immutability assumed; re-hash on any re-delivery).

## Confirmatory analyses REMAINING under this freeze (not yet run):
1. Stage-2 pilot endpoints per PILOT_SAP_M14.md (requires pilot data).
2. Any re-pinned claims after the K=7 re-cert replication (B8).
Everything else from this point is exploratory unless a new SAP version is
frozen first.

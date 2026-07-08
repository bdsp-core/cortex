# Label-schedule randomization — G-C OC campaign artifacts

Companion artifacts for `docs/LABEL_SCHEDULE_PECR.md` (gate G-C). All runs
are legacy-vs-randomized on MATCHED seeds, real K=7 bank
(`SimBankAdapter(build_k7_engine_inputs())`), mixture filter stack,
ROUND-3b (paired-bin) build of `trainer/label_schedule.py`.

| File | What |
|---|---|
| `oc_campaign_results.json` + `oc_campaign.py` | Single-task sessions, weak task 2: trainable (l_true 0.0/0.006, soft, 2×10 seeds, budget 240) + static-below-cut FG arm (l_true 0.15, 15 seeds). |
| `oc_campaign2_results.json` + `oc_campaign2.py` | Sharp-margin slice: near-cut soft (l_true 0.25 = 0.056 below cut, 10 seeds) + sharp static FG arm (l_true 0.25, 15 seeds, budget 240). |
| `closed_loop_pair16_r3.json` + `closed_loop_pair16_r3.py` | Full test→train→retest closed loop (`run_closed_loop`, real exam engine), 16 matched seed pairs, per-task final verdicts + true (ℓ, t). |
| `fg_budget_check.py` | Time-rescaling check: legacy sharp-margin FG at the evidence-equivalent budget 325 (measured 0.73 vs randomized 0.67 @ 240). |
| `realbank_adversaries.py` + `realbank_adversaries.json` | Real-bank served-stream adversary measurement, both arms: label-history oracle, \|s\|-cluster, anti-repeat, and the sd-cluster perceived-ambiguity channel (the bank-content limitation recorded as the PECR ledger's open owner item; measured here: legacy 0.935 with anti-repeat 1.000, randomized 0.998 with all label/|s| channels at chance). |

Data is from the ROUND-3b build (paired-bin serving, precision-aware —
PECR §5c). Headline: delivered value +25–38% AND median declaration
lateness 96–97 → 69–70 (−28%, the round-2 lateness cost flipped to a
gain), terminal |t_true| 0.197–0.207 → 0.093–0.131; false graduation
within ±1 seed of legacy per arm (0.40→0.47, 0.60→0.67, n=15 arms;
pooled 15/30 → 17/30, n.s.) with false declarations arriving earlier
(172→127) — consistent with a time-rescaled process at a fixed
240-trial window (see the ledger's budget-equalized legacy check).
Closed loop (16 matched pairs): pass delta −0.31 (t=−0.49), |t_true|
+0.007 (t=+0.47), ℓ_true +0.013 (t=+0.89) — all null; the round-2
short-stream |t_true| effect (+0.0246, p=0.042) vanished with the
equalizer's removal.

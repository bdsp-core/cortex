# V15 engine staging (OPT-IN)

Phase 5 of the CORTEX web migration stages the **v15 engine parameters** behind
an opt-in flag. The default (pilot) path is provably **bit-identical** to the
frozen pilot instrument that an in-flight pilot relies on, so this change is
safe to land before the pilot completes.

## What "v15" is

Two parameter changes relative to the frozen pilot instrument:

1. **`N = 1200` particles** (up from the pilot's `N = 600`). More particles →
   lower Monte-Carlo error in the per-task posteriors and the AUROC credible
   half-widths.
2. **A separate `Corr_t` prior for the t-block.** The pilot instrument uses the
   single fitted `Corr_l` (K×K) for *both* the t-block and the l-block. v15
   supplies a distinct `Corr_t` (K×K) for the **t-block only**; the **l-block
   always keeps `Corr_l`**.

The third v15 ingredient — the **credentialed-panel ℓ\*** cut-scores
(`ell_star_unified_v15`) — is **already live** in the shipped bundle (the
default `--cert-block` is `ell_star_unified_v15`). It is not gated by this flag;
only `N` and `Corr_t` are staged here.

## Why this is safe (bit-identical default)

The engine carries the prior as a `{tPieces, lPieces}` pair
(`engine/types.ts:PriorPair`):

- **Pilot path** (`corrT` absent): `tPieces = precomputePrior(corrL)` and
  `lPieces = precomputePrior(corrL)` — i.e. both blocks use `corrL`, exactly as
  the single-`PriorPieces` era did. Every float op in `samplePrior` /
  `logPriorOne` runs in the same order on the same matrices, and the RNG draw
  order (t-block ε then l-block ε per particle) is unchanged. The particle count
  resolves to `N_PARTICLES = 600` (`this.inputs.nParticles ?? N_PARTICLES`).
- **v15 path** (`corrT` present): `tPieces = precomputePrior(corrT)` for the
  t-block; the l-block still uses `precomputePrior(corrL)`. `N` resolves to the
  manifest's `nParticles` (1200).

The bit-identical contract is gated by `engine/v15_drift.test.ts`, whose
default-path snapshot (`engine/__snapshots__/v15_drift.test.ts.snap`) was
recorded from the **pre-refactor** code. It must keep matching with **no `-u`**;
if it ever fails to match, the default path has drifted and the refactor — not
the snapshot — is wrong. The same test proves the opt-in flag actually changes
behaviour (a different `Corr_t` changes the t-block while leaving the l-block
byte-identical, and `nParticles` flows through to the cloud size).

## How to enable it

Build the bundle with the `v15` profile:

```sh
python cortex_web/scripts/prepare_web_bundle.py --version <ver> --profile v15
```

`--profile v15` makes the manifest additionally carry:

| key          | value                              | consumed by |
|--------------|------------------------------------|-------------|
| `engineProfile` | `"v15"`                          | provenance only |
| `corrT`      | the `Corr_t` K×K from `Sigma_l_fitted_k7.npy` | `bundle.ts → EngineInputs.corrT → precomputePriorPair` |
| `nParticles` | `1200`                             | `bundle.ts → EngineInputs.nParticles → session N` |

The default `--profile pilot` emits **no** `corrT` and **no** `nParticles` (only
a descriptive `engineProfile: "pilot-frozen"`), so the browser engine resolves
`N = 600` and uses `corrL` for both prior blocks — the frozen-pilot instrument.

No code switch is needed to flip profiles: the engine reads the manifest. Ship a
`--profile v15` bundle and the SPA picks up `N = 1200 + Corr_t` automatically;
ship a `--profile pilot` bundle and it reverts.

## Post-pilot flip procedure

The v15 parameters are **staged, not active**. Flipping the live instrument to
v15 is a governed change:

1. **PI sign-off** on the v15 parameter set (N = 1200, `Corr_t`) and on closing
   the in-flight pilot at the frozen instrument.
2. **Re-freeze** the instrument at v15 (the freeze pipeline / instrument-freeze
   tests are out of scope for this staging change and were intentionally **not
   touched** here: `calibration/cert_config.yaml`,
   `scripts/freeze_instrument_v1_3_6.py`, and the instrument-freeze tests).
3. **Rebuild + republish** the production bundle with `--profile v15` and update
   the SPA's bundle version pointer.
4. Keep the `--profile pilot` build reproducible for audit (the frozen
   instrument any pilot results were scored against).

Until step 1, production continues to ship `--profile pilot`.

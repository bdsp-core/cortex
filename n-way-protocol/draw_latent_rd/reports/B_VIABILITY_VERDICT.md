# Construction-B viability verdict: VIABLE (2026-07-31)

Seven 192-seed CRN cells at the production profile, binary control from
the shipping harness verbatim, draw-latent oracle isolated in this
sandbox (byte-identical to the shipping engine in the single-atom limit).

| cell | world | B skill cov [Wilson] | static baseline | binary |
|---|---|---|---|---|
| ongrid9_exact | atoms incl. 0.15 lapse | **0.9436 [0.929, 0.955]** | — | 0.9557 |
| ongrid9_floor | atoms, world λ=0 | **0.9479 [0.934, 0.959]** | 0.9375 | 0.9497 |
| continuum9 | pre-reg gate world | 0.9392 [0.924, 0.952] | 0.9227 | 0.9488 |
| **continuum17** | pre-reg gate world | **0.9505 [0.936, 0.962]** | 0.9227 | 0.9488 |
| **continuum33** | pre-reg gate world | **0.9523 [0.938, 0.963]** | 0.9227 | 0.9488 |
| stress190 | β = 1.903 point mass | 0.9062 [0.888, 0.922] | 0.8108 | 0.9505 |
| collapse_nest | λ_d = 1 uniform | **0.9323 [0.916, 0.945]** | 0.4288 | 0.9566 |

## Findings

1. **The implementation is correct**: with truth on the engine's own
   atoms (lapse included) coverage is nominal within noise — the joint
   treatment is calibrated exactly where static averaging (0.9375 with a
   λ-mismatch confound) was not, and the 0.15 floor costs ~nothing
   under B (0.9479).
2. **The pre-registered gate world passes at 17+ atoms**: 0.9505/0.9523
   with Wilson containing nominal, paired noninferiority
   confidence-passing (Δ +0.002/+0.004), and the full accuracy prize
   intact (skill RMSE 0.356 vs binary 0.538, −34%). Atom count is
   statistically near-free under selection (no blur penalty), exactly as
   designed.
3. **Graceful degradation is real**: with a λ=1 binary-equivalent atom,
   the uniform-collapse world recovers from static's 0.4288 to 0.9323 —
   sessions concentrate 0.92 posterior mass on the nesting atom and
   effectively become binary without waiting for the monitor. The
   monitor/floor/replay stack remains as defense in depth.
4. **Open edge — point-mass tail worlds**: β = 1.903 improves 0.8108 →
   0.9062 but stays sub-nominal; the continuum weights such tails by
   their prevalence and passes, but a sitter exactly at the independent
   expert median is still ~4.4% undercovered. Options for the locked
   design: heavier tail atoms (grid from the OOS τ with explicit tail
   mass), more particles, or accepting the cell as informational with
   the documented residual. Owner call at pre-registration amendment.

## Status

Research-only sandbox result. Next steps per the approved staged plan:
TS engine port with parity artifacts, then the locked campaign under the
amended pre-registration (same gates, same margins, same park rule).

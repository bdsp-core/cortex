# Conditional F1 protocol contract

## Scope

The spike task remains binary. Each registered IIIC group consumes one raw
pick among its member tasks. No module assumes that spike is index 0, that
IIIC starts at index 1, or that a categorical group always contains six tasks.

For focal class `k`:

```text
p_k = lambda + (1 - 2 lambda) Phi(z_k)
P(R=k) = p_k
P(R=j) = (1-p_k) [lambda_d/(G-1) + (1-lambda_d) softmax(beta z_-k)_j]
z_j = exp(l_j) (s_j+t_j) / sqrt(1 + (exp(l_j) s_sd,j)^2)
```

`lambda` is fixed at 0.025. `beta` and `lambda_d` are artifact fields. The
focal probability and its complement use the same stable probit-lapse
implementation as the binary protocol.

## Required invariants

1. Response probabilities are finite, non-negative, and sum to one.
2. `P(R=k)` equals the binary `P(y=1)` for all finite inputs.
3. Summed wrong-pick probability equals binary `P(y=0)`.
4. Spike never enters a categorical group.
5. History retains raw picks and immutable segment indices.
6. MH replay evaluates the exact model that created the sitting.
7. One response removes one segment and increments one direct-domain count.
8. Speculative branches are immutable clones; only the actual outcome lands.
9. An n-way session never falls back to a binary update.
10. Result and resume profiles must exactly match the server session stamp.

## F1 versus full softmax

F1 is the certification bridge because it preserves the current binary
marginal and cut coordinate. The trainer's full-softmax learning model is not
used here. Promoting full softmax into certification would be a different
program requiring a validated reduction map, new calibration, and cut/decision
qualification.

## Selector

The exact response-marginalized loss is computed from the law of total
variance over every `t` and `l` coordinate. The full-bank selector first builds
a deterministic shortlist from focal-difficulty representatives, minimum
signal-uncertainty representatives, high predictive-entropy candidates, and
forced content-deficit segments. Exact categorical loss decides only among
that shortlist.

The shortlist constants are protocol defaults, not promoted values. A selector
regret and latency audit must freeze them before production integration.

## Standard integrated protocol

`research_selector.ts` augments the frozen shortlist with a local Fisher
information screen, then still uses the exact six-outcome total-variance loss
for the final decision. `research_artifact.ts` evaluates an exact weighted
mixture of conditional-F1 artifact draws on the probability scale. Both are
opt-in exports and are not imported by `session.ts`; they do not alter the
frozen protocol by being present.

The standard isolated n-way approach combines these extensions. Its immutable
profile uses `categorical_fisher_totalvar_v1`; scalar artifacts and
`categorical_totalvar_v1` remain valid for exact frozen-session replay. The
profile dispatcher adds Fisher candidates to the baseline shortlist and then
uses exact ensemble-marginalized total-variance loss. Bias-weighted loss,
mutual information, alternate resampling, and the RQMC hybrid remain comparison
arms only. `artifacts/nway_standard_rd.json` is the machine-readable standard
pointer. The frozen archive remains the rollback and recovery authority.

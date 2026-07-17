"""Tier-3 item-level rollout placement (trainer_rd trainer_rollout) — DEFERRED.

Only reached from `trainer.policy` when `ModeThresholds.rollout_placement=True`
(an opt-in ablation, default False). The shipped tier-2 default never calls it, so
the G1 port and its bit-parity tests do not exercise this path. Port it here if the
rollout ablation is needed downstream.
"""
from __future__ import annotations


def rollout_item_q(*args, **kwargs):  # pragma: no cover - opt-in, not ported
    raise NotImplementedError(
        "tier-3 rollout placement (rollout_placement=True) is not ported in G1; "
        "the shipped tier-2 default (rollout_placement=False) does not need it.")

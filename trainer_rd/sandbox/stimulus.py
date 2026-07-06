"""M20 — terminal perceptual stimulus for the manual sandbox.

The scrubbed bank carries latent signals, not renderable domain traces, so
the sandbox presents a SYNTHETIC PERCEPTUAL TASK whose difficulty maps
monotonically to the item's latent signal: a noisy trace with a target
DEFLECTION of amplitude RENDER_GAIN·s_real inside a marked window. The
human's answer is a genuine signal-detection decision — their internal
noise and criterion become real (σ, t) in the SAME latent units the whole
stack operates in (any perceptual gain mismatch is absorbed into the
human's effective σ, which the filter estimates; that is the point).

Rendering is DETERMINISTIC given (seg_id, session_no): reproducible for
analytics, and s_real is drawn once and logged (sandbox-only ground truth —
production never observes it).
"""
from __future__ import annotations

import numpy as np

from sandbox.config import (RENDER_GAIN, RENDER_N, RENDER_NOISE,
                            TEMPLATE_HI, TEMPLATE_LO)

_BLOCKS = "▁▂▃▄▅▆▇█"


def draw_s_real(s_mean, s_sd, seg_id, session_no, y_star=None):
    """Item's realized latent signal. With y_star given (M21/F74), the draw
    is TRUNCATED to the label-consistent sign: production feedback items are
    margin-filtered so the trace an expert labeled agrees with its label
    (D11); the unconditioned M20 draw broke that contract at the render
    layer — 10% of served tester trials showed evidence OPPOSING the
    feedback label, training the wrong mapping. Inverse-CDF truncation keeps
    the draw deterministic per (seg, session)."""
    rng = np.random.default_rng(1_000_003 * int(seg_id) + 97 * int(session_no))
    u = rng.uniform()
    if y_star is None or s_sd <= 0:
        z = _norm_ppf(u)
        s = s_mean + s_sd * z if s_sd > 0 else s_mean
        return float(s)
    # truncate N(s_mean, s_sd²) to (0,∞) if y*=1 else (−∞,0)
    from scipy.stats import norm
    lo = norm.cdf((0.0 - s_mean) / s_sd)
    u_t = lo + u * (1.0 - lo) if y_star == 1 else u * lo
    u_t = min(max(u_t, 1e-12), 1.0 - 1e-12)
    return float(s_mean + s_sd * norm.ppf(u_t))


def _norm_ppf(u):
    from scipy.stats import norm
    return float(norm.ppf(min(max(u, 1e-12), 1.0 - 1e-12)))


def render(s_real, seg_id, session_no):
    """Return (trace_string, marker_string, samples). The target window is
    marked under the trace; 'target present' = upward deflection there."""
    rng = np.random.default_rng(7_000_003 * int(seg_id)
                                + 101 * int(session_no) + 13)
    x = RENDER_NOISE * rng.standard_normal(RENDER_N)
    x[TEMPLATE_LO:TEMPLATE_HI] += RENDER_GAIN * s_real
    lo, hi = x.min(), x.max()
    span = max(hi - lo, 1e-9)
    idx = np.clip(((x - lo) / span * (len(_BLOCKS) - 1)).round().astype(int),
                  0, len(_BLOCKS) - 1)
    trace = "".join(_BLOCKS[i] for i in idx)
    marker = "".join("^" if TEMPLATE_LO <= i < TEMPLATE_HI else " "
                     for i in range(RENDER_N))
    return trace, marker, x


def window_elevation(samples):
    """Diagnostic: mean elevation of the target window vs the surround —
    the statistic a monotonicity test checks against s_real."""
    win = samples[TEMPLATE_LO:TEMPLATE_HI].mean()
    out = np.r_[samples[:TEMPLATE_LO], samples[TEMPLATE_HI:]].mean()
    return float(win - out)

"""G0 — ℓ* identity: the trainer reads the SAME cut-scores as the exam, and the
registry's cert_config keys resolve for v14 and v15. The label panel (65 experts)
and the cut panel (Super8 ∪ Bonobo) are deliberately distinct; the cut values are
pinned here so neither drifts."""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)

from trainer import domains  # noqa: E402

# Pinned Youden ℓ*_k (drift guards). v14 == trainer_rd ELL_STAR exactly.
V14 = [0.3251213139414513, 0.2559949138064387, 0.5337430687087749,
       0.3297002787536504, 0.4792595265871899, 0.4864510172880300,
       0.4418131962513707]
V15 = [0.2382694370243232, 0.1548131161373049, 0.3059231418723579,
       0.2569706571684701, 0.3213979957403151, 0.3539561460242777,
       0.3042300471351564]


def test_ell_star_v14_pinned():
    from cortex_policy_k7 import load_ell_star_k7
    got = load_ell_star_k7(list(domains.CODES), block_name="ell_star_unified_v14")
    np.testing.assert_allclose(got, V14, rtol=0, atol=1e-12)


def test_ell_star_v15_pinned():
    from cortex_policy_k7 import load_ell_star_k7
    got = load_ell_star_k7(list(domains.CODES), block_name="ell_star_unified_v15")
    np.testing.assert_allclose(got, V15, rtol=0, atol=1e-12)


def test_registry_cert_keys_match_loader_mapping():
    # the registry's cert_key for each code == the key the production loader uses
    from cortex_policy_k7 import _KEY_FOR_CODE_K7
    for d in domains.DOMAINS:
        assert _KEY_FOR_CODE_K7[d.code] == d.cert_key


def test_ell_star_ordering_is_registry_ordering():
    # loading in registry order must give the pinned array in that same order
    from cortex_policy_k7 import load_ell_star_k7
    got = load_ell_star_k7(list(domains.CODES))  # default block = v15 (2026-07-16)
    assert len(got) == domains.K
    np.testing.assert_allclose(got, V15, rtol=0, atol=1e-12)

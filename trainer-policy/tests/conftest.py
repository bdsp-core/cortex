"""Test isolation for the root trainer-policy representation."""
from __future__ import annotations

import os
import sys
from pathlib import Path


for _name in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
PRODUCTION_PACKAGE = REPO_ROOT / "cortex_web" / "learning-engine-cleaned"

for _path in (PACKAGE_ROOT, PRODUCTION_PACKAGE, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

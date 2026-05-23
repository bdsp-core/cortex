"""Combine the per-session CORTEX result CSVs into two master tables.

Each completed session writes its own `<id>_<name>_trials.csv` and
`<id>_<name>_summary.csv` into the synced folder (so multi-machine cloud
sync never collides). This script merges them:

    all_trials.csv    every question of every session
    all_summary.csv   one row per test-taker — the roster

Usage:
    python scripts/combine_results.py [folder]

`folder` defaults to `synced_results_dir` from cortex_config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))
from cortex_storage import load_synced_dir  # noqa: E402


def combine(folder: Path) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    for kind in ("trials", "summary"):
        files = sorted(f for f in folder.glob(f"*_{kind}.csv")
                       if not f.name.startswith("all_"))
        if not files:
            print(f"  no *_{kind}.csv files in {folder}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out = folder / f"all_{kind}.csv"
        df.to_csv(out, index=False)
        print(f"  {kind}: {len(files)} session(s) -> {out}  ({len(df)} rows)")


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else load_synced_dir()
    if folder is None:
        raise SystemExit(
            "No folder given and synced_results_dir is unset in "
            "cortex_config.yaml — pass a folder explicitly.")
    combine(Path(folder))


if __name__ == "__main__":
    main()

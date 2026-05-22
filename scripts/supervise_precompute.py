"""Supervisor for precompute_iiic.py — survives the Extreme SSD vanishing.

The IIIC precompute writes to a side file on an external (encrypted APFS)
SSD. If that volume unmounts mid-run — sleep, USB hiccup, reboot, accidental
eject — the worker pool wedges and the side-file path disappears. This
supervisor:

  1. Waits for the volume + side file directory to be present and writable.
  2. Launches precompute_iiic.py (which resumes by skipping already-done
     segments) and tees its output into the same log the monitor tails.
  3. While it runs, polls the volume every CHECK_S seconds. If the volume
     vanishes, it kills the precompute process tree immediately (rather than
     letting it hang or error-loop) and goes back to step 1.
  4. When precompute exits cleanly AND no reachable segments remain, stops.

It cannot unlock an encrypted volume (no password) — on a remount that needs
a password it simply waits until you unlock it in Finder, then resumes.

USAGE
    caffeinate -dimsu python3 scripts/supervise_precompute.py --workers 8
    # or background:
    nohup caffeinate -dimsu python3 scripts/supervise_precompute.py \
        --workers 8 > /tmp/iiic_supervisor.log 2>&1 &
"""
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import pandas as pd

ROOT = Path("/Users/mwestover/GithubRepos/ideal-test-multi")
SPEC_PATH = Path("/Volumes/Extreme SSD/eeg_bank_spec.h5")
MANIFEST = ROOT / "data/iiic_precompute_manifest.csv"
PRECOMPUTE_LOG = Path("/tmp/iiic_precompute.log")

CHECK_S = 10          # how often to poll the volume while running
WAIT_S = 30           # how often to re-check for the volume when it's gone
STRATEGIES = {"morgoth1_10min", "kong_contest_h5"}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[supervisor {ts}] {msg}", flush=True)


def volume_ready(spec: Path) -> bool:
    """True iff the side-file directory exists and is writable."""
    d = spec.parent
    if not d.is_dir():
        return False
    probe = d / ".supervisor_write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def count_reachable(manifest: Path) -> int:
    man = pd.read_csv(manifest, low_memory=False, dtype={"notes": "str"})
    return int(man["fetch_strategy"].isin(STRATEGIES).sum())


def count_done(spec: Path) -> int:
    """Segments with both data30s + sdata written. 0 if file absent/unreadable."""
    if not spec.exists():
        return 0
    try:
        with h5py.File(spec, "r") as f:
            grp = f.get("segments")
            if grp is None:
                return 0
            n = 0
            for k in grp.keys():
                g = grp[k]
                if "sdata" in g and "data30s" in g:
                    n += 1
            return n
    except Exception as e:
        log(f"WARN: could not read side file to count done: {e}")
        return -1  # unknown — don't treat as complete


def wait_for_volume(spec: Path) -> None:
    announced = False
    while not volume_ready(spec):
        if not announced:
            log(f"volume not ready ({spec.parent}). Waiting for remount/unlock "
                f"(checking every {WAIT_S}s). If encrypted, unlock it in Finder.")
            announced = True
        time.sleep(WAIT_S)
    if announced:
        log("volume is back and writable. Resuming.")


def run_once(workers: int) -> str:
    """Launch precompute, supervise the volume while it runs.

    Returns one of: 'exited' (process ended on its own),
    'volume_lost' (we killed it because the SSD vanished).
    """
    # Append to the same log the milestone monitor tails.
    logf = open(PRECOMPUTE_LOG, "a", buffering=1)
    logf.write(f"\n===== supervisor launch {time.strftime('%Y-%m-%d %H:%M:%S')} "
               f"(workers={workers}) =====\n")
    logf.flush()
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts/precompute_iiic.py"),
         "--workers", str(workers)],
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=str(ROOT), start_new_session=True,  # own process group → clean kill
    )
    log(f"launched precompute pid={proc.pid} (output → {PRECOMPUTE_LOG})")
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                log(f"precompute exited (rc={rc}).")
                return "exited"
            if not volume_ready(SPEC_PATH):
                log("VOLUME VANISHED — killing precompute process group.")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
                return "volume_lost"
            time.sleep(CHECK_S)
    finally:
        logf.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-restarts", type=int, default=1000,
                    help="Safety cap on relaunch count.")
    args = p.parse_args()

    reachable = count_reachable(MANIFEST)
    log(f"reachable segments in manifest: {reachable:,}")

    restarts = 0
    while True:
        wait_for_volume(SPEC_PATH)

        done = count_done(SPEC_PATH)
        if done >= 0:
            log(f"side file holds {done:,}/{reachable:,} done "
                f"({reachable - done:,} remaining).")
            if done >= reachable:
                log("All reachable segments complete. Supervisor exiting.")
                return
        done_before_run = done

        outcome = run_once(args.workers)
        restarts += 1

        if outcome == "exited":
            # Did it finish, or just crash? Re-check completion.
            time.sleep(3)
            if not volume_ready(SPEC_PATH):
                log("precompute exited and volume is gone — treating as "
                    "volume loss; will wait for remount.")
                continue
            done = count_done(SPEC_PATH)
            if done >= reachable:
                log(f"Complete: {done:,}/{reachable:,}. Supervisor exiting.")
                return
            # A clean pass re-attempts every not-yet-done segment. If it added
            # zero, the remainder is unfetchable (genuinely-missing source
            # files, e.g. pd_rda_profiler with no 10-min recording) — keep
            # relaunching forever would be a tight infinite loop. Stop instead.
            if done_before_run >= 0 and done <= done_before_run:
                log(f"Clean pass added 0 segments — the remaining "
                    f"{reachable - done:,} have no fetchable source. "
                    f"Stopping (see data/iiic_unfetchable.csv).")
                return
            log(f"precompute exited; {reachable - done:,} remain but "
                f"{done - done_before_run:,} were recovered this pass. "
                f"Relaunching in 15s.")
            time.sleep(15)
        else:  # volume_lost
            log("Waiting for the volume to come back before relaunching.")
            time.sleep(5)

        if restarts >= args.max_restarts:
            log(f"Hit --max-restarts={args.max_restarts}. Stopping.")
            return


if __name__ == "__main__":
    main()

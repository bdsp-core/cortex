"""/api/videos — server-side MP4 rendering of the session's particle-cloud
trajectory, as an asynchronous JOB.

The browser posts the trajectory (t/l/w Float32 blobs + a JSON meta) captured
during the session; we reconstruct the desktop session-dir schema
(trajectory.npz + trials.jsonl + certificate.json) and run the EXACT desktop
renderers (scripts/cortex_render_videos.py + render_engine_explainer.py).

Contract (replaces the old synchronous one, which held the HTTP connection
open for the full ~10-minute render — fragile through NATs/proxies, and a
disconnected client kept burning CPU invisibly):

    POST /api/videos                  → 202 {jobId, status: "queued"}
    GET  /api/videos/{jobId}          → {jobId, status: queued|running|done|error[, error]}
    GET  /api/videos/{jobId}/download → the zip (409 until status == "done")

Jobs live in in-process memory (single-worker design — see run.py): a service
restart forgets them; the client sees 404 and re-submits. Renders run ONE at
a time (semaphore, matplotlib is memory-heavy and not thread-safe on the
2-vCPU/2-GB box); one extra job may queue behind the running one, further
submissions get an honest 503. Artifacts are swept _JOB_TTL_SECONDS after a
job finishes (lazily, on submit/poll).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from ..config import SCRIPTS_DIR
from ..deps import require_auth

router = APIRouter(prefix="/api")

# One matplotlib render at a time (see module docstring).
_RENDER_SEM = threading.Semaphore(1)
# Upload sanity caps for the client-supplied trajectory shape (T, N, K).
# Prod sessions are ≈(≤420, 1200, 7) ⇒ ~3.5M elements; the caps are generous
# multiples, and the PRODUCT cap is what actually bounds the allocation
# (T·N·K float64 ×2 arrays), so a forged meta can't OOM the box.
_MAX_VIDEO_T, _MAX_VIDEO_N, _MAX_VIDEO_K = 2000, 4096, 16
_MAX_VIDEO_ELEMS = 8_000_000

# Job registry (in-process; single worker by design).
_MAX_ACTIVE_JOBS = 2            # 1 rendering + 1 queued; beyond that → 503
_JOB_TTL_SECONDS = 30 * 60      # finished jobs (zip incl.) kept this long
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


_ORPHANS_SWEPT = False


def _sweep_orphan_dirs(now: Optional[float] = None) -> None:
    """Reclaim cortex_viz_* temp dirs left behind by a PREVIOUS process. Jobs
    live in memory, so a service restart forgets them but leaves their dirs —
    an unbounded /tmp leak on the small box. Only dirs older than the job TTL
    and not owned by a live job in THIS process are touched."""
    now = now if now is not None else time.time()
    with _JOBS_LOCK:
        live = {str(j["sd"]) for j in _JOBS.values()}
    for d in Path(tempfile.gettempdir()).glob("cortex_viz_*"):
        try:
            if str(d) not in live and now - d.stat().st_mtime > _JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def _sweep_jobs(now: Optional[float] = None) -> None:
    """Drop finished jobs past their TTL and delete their temp dirs. Called
    lazily from submit/status — no background janitor thread needed. The first
    call per process also reclaims orphan dirs from before the last restart."""
    global _ORPHANS_SWEPT
    now = now if now is not None else time.time()
    if not _ORPHANS_SWEPT:
        _ORPHANS_SWEPT = True
        _sweep_orphan_dirs(now)
    with _JOBS_LOCK:
        for jid, job in list(_JOBS.items()):
            done_ts = job.get("done_ts")
            if done_ts is not None and now - done_ts > _JOB_TTL_SECONDS:
                shutil.rmtree(job["sd"], ignore_errors=True)
                del _JOBS[jid]


def _active_job_count() -> int:
    with _JOBS_LOCK:
        return sum(1 for j in _JOBS.values() if j["status"] in ("queued", "running"))


def _execute_render(sd: Path) -> Path:
    """Run the desktop renderers over the reconstructed session dir and return
    the zip path. Module-level so tests can monkeypatch the heavy part."""
    scripts = SCRIPTS_DIR
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from cortex_render_videos import render_all
    from render_engine_explainer import render_engine_explainer
    outs = list(render_all(sd).values()) + [render_engine_explainer(sd)]
    zpath = sd / "cortex_visualizations.zip"
    import zipfile
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in outs:
            z.write(p, arcname=Path(p).name)
    return zpath


def _write_inputs(sd: Path, tb, lb, wb, info: dict, T: int) -> None:
    """Reconstruct the desktop session-dir schema the renderers expect."""
    import numpy as np
    np.savez_compressed(
        sd / "trajectory.npz", t_traj=tb, l_traj=lb, w_traj=wb,
        task_codes=np.array(list(info["taskCodes"])),
        seg_ids=np.array(list(info["segIds"]), dtype=np.int64),
        delta_auroc=np.float64("nan"), n_questions=int(T))
    with open(sd / "trials.jsonl", "w") as fh:
        for tr in info["trials"]:
            fh.write(json.dumps(tr) + "\n")
    (sd / "certificate.json").write_text(json.dumps(info["certificate"]))
    (sd / "participant.json").write_text(
        json.dumps({"identity": {"name": info.get("participantName", "Anonymous")}}))


def _job_worker(job: dict[str, Any], tb, lb, wb, info: dict, T: int) -> None:
    """Background thread: write inputs, wait for the render slot, render.
    Every exit path stamps done_ts so the TTL sweep reclaims the temp dir."""
    try:
        _write_inputs(job["sd"], tb, lb, wb, info, T)
        with _RENDER_SEM:               # blocking: a queued job waits its turn
            with _JOBS_LOCK:
                job["status"] = "running"
            zpath = _execute_render(job["sd"])
        with _JOBS_LOCK:
            job["zip"] = zpath
            job["status"] = "done"
            job["done_ts"] = time.time()
    except Exception as e:
        with _JOBS_LOCK:
            job["error"] = str(e)[:500]
            job["status"] = "error"
            job["done_ts"] = time.time()


@router.post("/videos")
async def videos_submit(meta: str = Form(...), t: UploadFile = File(...),
                        l: UploadFile = File(...), w: UploadFile = File(...),
                        code: str = Depends(require_auth)):
    try:
        import numpy as np
    except Exception as e:  # heavy render deps are optional at boot
        raise HTTPException(503, f"render deps unavailable: {e}")
    # Client-supplied meta: malformed JSON / missing or non-3-int shape is a
    # 400, not an unhandled 500.
    try:
        info = json.loads(meta)
        T, N, K = (int(x) for x in info["shape"])
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "malformed meta: expected JSON with a 3-int 'shape'")
    # The client-supplied shape dictates the allocation below — bound it
    # before touching the uploads (blobs are read to exactly the byte count
    # the shape implies, so a mismatched/oversized upload is rejected
    # instead of buffered).
    if not (0 < T <= _MAX_VIDEO_T and 0 < N <= _MAX_VIDEO_N
            and 0 < K <= _MAX_VIDEO_K and T * N * K <= _MAX_VIDEO_ELEMS):
        raise HTTPException(413, f"trajectory shape {T}x{N}x{K} exceeds limits")

    async def _read_exact(up: UploadFile, n_elem: int, name: str) -> bytes:
        raw = await up.read(n_elem * 4 + 1)
        if len(raw) != n_elem * 4:
            raise HTTPException(400, f"upload '{name}' does not match meta shape")
        return raw

    raw_t = await _read_exact(t, T * N * K, "t")
    raw_l = await _read_exact(l, T * N * K, "l")
    raw_w = await _read_exact(w, T * N, "w")

    def _parse_blobs():
        # Up to two ~64 MB float64 copies per array — real CPU work, so it
        # runs on the threadpool: this is the ONLY async route, and doing it
        # inline stalled the event loop (and every concurrent request) on the
        # single-worker box.
        tb = np.frombuffer(raw_t, dtype="<f4").reshape(T, N, K).astype(np.float64)
        lb = np.frombuffer(raw_l, dtype="<f4").reshape(T, N, K).astype(np.float64)
        wb = np.frombuffer(raw_w, dtype="<f4").reshape(T, N).astype(np.float64)
        return tb, lb, wb

    tb, lb, wb = await run_in_threadpool(_parse_blobs)

    _sweep_jobs()
    if _active_job_count() >= _MAX_ACTIVE_JOBS:
        raise HTTPException(503, "the render queue is full — try again in a few minutes")

    job_id = uuid.uuid4().hex
    job: dict[str, Any] = {
        "id": job_id, "code": code, "sd": Path(tempfile.mkdtemp(prefix="cortex_viz_")),
        "status": "queued", "error": None, "zip": None,
        "created_ts": time.time(), "done_ts": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    threading.Thread(target=_job_worker, args=(job, tb, lb, wb, info, T),
                     daemon=True, name=f"cortex-viz-{job_id[:8]}").start()
    return JSONResponse(status_code=202, content={"jobId": job_id, "status": "queued"})


def _get_owned_job(job_id: str, code: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None or job["code"] != code:
        raise HTTPException(404, "unknown render job")
    return job


@router.get("/videos/{job_id}")
def videos_status(job_id: str, code: str = Depends(require_auth)):
    _sweep_jobs()
    job = _get_owned_job(job_id, code)
    resp: dict[str, Any] = {"jobId": job_id, "status": job["status"]}
    if job["status"] == "error" and job.get("error"):
        resp["error"] = job["error"]
    return resp


@router.get("/videos/{job_id}/download")
def videos_download(job_id: str, code: str = Depends(require_auth)):
    job = _get_owned_job(job_id, code)
    if job["status"] != "done":
        raise HTTPException(409, f"render not finished (status: {job['status']})")
    # No cleanup on download — the TTL sweep reclaims the dir, so the client
    # can re-download after a flaky connection.
    return FileResponse(job["zip"], media_type="application/zip",
                        filename="cortex_visualizations.zip")

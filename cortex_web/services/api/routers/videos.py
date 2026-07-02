"""POST /api/videos — server-side MP4 rendering of the session's particle-
cloud trajectory.

The browser posts the trajectory (t/l/w Float32 blobs + a JSON meta) captured
during the session; we reconstruct the desktop session-dir schema
(trajectory.npz + trials.jsonl + certificate.json) and run the EXACT desktop
renderers (scripts/cortex_render_videos.py + render_engine_explainer.py),
then stream back a zip of the 4 MP4s.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from ..config import SCRIPTS_DIR
from ..deps import require_auth

router = APIRouter(prefix="/api")

# One matplotlib render at a time: renders take minutes, peak at hundreds of
# MB, and matplotlib's global state is not thread-safe — a second concurrent
# render on the 2-vCPU/2-GB box risks an OOM kill, so it gets a 503 instead.
_RENDER_SEM = threading.Semaphore(1)
# Upload sanity caps for the client-supplied trajectory shape (T, N, K).
# Prod sessions are ≈(≤420, 1200, 7) ⇒ ~3.5M elements; the caps are generous
# multiples, and the PRODUCT cap is what actually bounds the allocation
# (T·N·K float64 ×2 arrays), so a forged meta can't OOM the box.
_MAX_VIDEO_T, _MAX_VIDEO_N, _MAX_VIDEO_K = 2000, 4096, 16
_MAX_VIDEO_ELEMS = 8_000_000


@router.post("/videos")
async def videos(meta: str = Form(...), t: UploadFile = File(...),
                 l: UploadFile = File(...), w: UploadFile = File(...),
                 _code: str = Depends(require_auth)):
    try:
        import numpy as np
    except Exception as e:  # heavy render deps are optional at boot
        raise HTTPException(503, f"render deps unavailable: {e}")
    info = json.loads(meta)
    T, N, K = (int(x) for x in info["shape"])
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

    tb = np.frombuffer(await _read_exact(t, T * N * K, "t"), dtype="<f4").reshape(T, N, K).astype(np.float64)
    lb = np.frombuffer(await _read_exact(l, T * N * K, "l"), dtype="<f4").reshape(T, N, K).astype(np.float64)
    wb = np.frombuffer(await _read_exact(w, T * N, "w"), dtype="<f4").reshape(T, N).astype(np.float64)

    sd = Path(tempfile.mkdtemp(prefix="cortex_viz_"))

    def _render():
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

    # Serialize renders (see _RENDER_SEM). Non-blocking: a second user gets
    # an immediate, honest 503 rather than queueing ~10 min behind the
    # first and risking an OOM. Acquired immediately before the guarded
    # block so every subsequent path — success or failure — releases it.
    if not _RENDER_SEM.acquire(blocking=False):
        shutil.rmtree(sd, ignore_errors=True)
        raise HTTPException(503, "another visualization render is in progress — try again in a few minutes")
    try:
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
        zpath = await run_in_threadpool(_render)
    except Exception as e:
        shutil.rmtree(sd, ignore_errors=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(500, f"render failed: {e}")
    finally:
        _RENDER_SEM.release()
    return FileResponse(
        zpath, media_type="application/zip", filename="cortex_visualizations.zip",
        background=BackgroundTask(shutil.rmtree, sd, True))

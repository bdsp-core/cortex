"""Engine-trainer endpoints (Phase L3): the vendored learning engine as the
server-side training decision-maker. Pure decision surface — the per-trial
ledger keeps its single writer (the client checkpoint outbox). Gated by
CORTEX_TRAINER_ENGINE (off|cohort|all; default off — the incumbent
client-side trainer remains the production posture until promotion)."""
from fastapi import APIRouter, Depends, HTTPException, Request

from .. import engine_trainer
from ..deps import require_auth
from ..models import EngineRecordIn, EngineStartIn

router = APIRouter(prefix="/api")

_manager = engine_trainer.EngineManager()


def _gate(req: Request, code: str):
    cfg = req.app.state.cfg
    participant = (req.app.state.db.get_participant(code)
                   if cfg.get("trainer_engine") == "cohort" else None)
    if not engine_trainer.enabled(cfg, code, participant):
        raise HTTPException(403, "engine trainer is not enabled "
                                 "for this account")


def _own(req: Request, training_id: str, code: str):
    if req.app.state.db.training_session_owner(training_id) != code:
        raise HTTPException(404, "unknown training session")


@router.post("/training-engine/start")
def engine_start(body: EngineStartIn, req: Request,
                 code: str = Depends(require_auth)):
    """Build (or rebuild from the ledger) the sitting's engine session:
    replay-seed the belief from the latest certification sitting
    (contract §2a), then return the first item + belief snapshot."""
    _gate(req, code)
    _own(req, body.trainingId, code)
    bank = req.app.state.get_bank()
    if bank is None:
        raise HTTPException(503, "question bank unavailable")
    try:
        engine_trainer._engine()
    except ImportError as e:
        raise HTTPException(503, f"engine numeric stack unavailable: {e}")
    es, item, snap = _manager.start(
        req.app.state.db, bank, body.trainingId, code,
        body.segIds, body.restrictTaskKs)
    # L4: sitting-level engine metadata lands in the ledger (server-side
    # writer) so analyses never re-derive it from HTTP logs.
    req.app.state.db.set_training_engine_meta(body.trainingId, dict(
        seeded=es.n_seeded, seedUnique=es.seed_unique,
        attainability=es.attain0, alpha=engine_trainer.ALPHA,
        nway=engine_trainer.NWAY, artifact="nway_dynamics_v1_1"))
    return {"item": item, "snapshot": snap,
            "allMastered": es.all_mastered(), "seeded": es.n_seeded,
            "rebuiltSeq": es.seq,
            # roadmap §2.1: attainability is a REPORT; in practice mode
            # (ALPHA=0, the pilot default) it never blocks serving.
            "attainability": es.attain0,
            "seedUnique": es.seed_unique}


@router.post("/training-engine/record")
def engine_record(body: EngineRecordIn, req: Request,
                  code: str = Depends(require_auth)):
    """Apply one answered item to the belief and return the next item +
    snapshot. item=None ⇒ the sitting is done (mastered / futile /
    candidates exhausted). The response record itself lands through the
    EXISTING client checkpoint outbox (single ledger writer)."""
    _gate(req, code)
    _own(req, body.trainingId, code)
    try:
        es, item, snap = _manager.record(body.trainingId, body.segId,
                                         body.pick)
    except LookupError:
        raise HTTPException(409, "engine session not started "
                                 "(POST /api/training-engine/start first)")
    except KeyError as e:
        raise HTTPException(409, str(e))
    return {"item": item, "snapshot": snap,
            "allMastered": es.all_mastered(), "done": item is None}

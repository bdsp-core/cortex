"""Badges + milestones surface: the Settings collection page reads the full
ledger; the dashboard milestone banner reads/acknowledges pending ones."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..deps import require_auth
from ..models import AwardAckIn

router = APIRouter(prefix="/api")


def _row(r: dict) -> dict:
    return {"awardId": r["award_id"], "kind": r["kind"], "key": r["key"],
            "label": r["label"], "detail": r["detail"],
            "awardedUtc": r["awarded_utc"], "revokedUtc": r["revoked_utc"]}


def awards_payload(db, code: str) -> dict:
    rows = db.list_awards(code)
    return {"badges": [_row(r) for r in rows if r["kind"] == "badge"],
            "milestones": [_row(r) for r in rows if r["kind"] == "milestone"]}


def pending_payload(db, code: str) -> dict:
    """The LIGHT bootstrap section: milestones awaiting their banner moment."""
    return {"pending": [
        {"awardId": r["award_id"], "key": r["key"], "label": r["label"],
         "detail": r["detail"], "awardedUtc": r["awarded_utc"]}
        for r in db.pending_milestones(code)]}


@router.get("/awards")
def awards_list(req: Request, code: str = Depends(require_auth)):
    return awards_payload(req.app.state.db, code)


@router.post("/awards/ack")
def awards_ack(body: AwardAckIn, req: Request, code: str = Depends(require_auth)):
    if not req.app.state.db.acknowledge_award(code, body.awardId):
        raise HTTPException(404, "unknown award")
    return {"ok": True}

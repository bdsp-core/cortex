"""Pydantic request models for every endpoint, in one place."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AuthIn(BaseModel):
    email: str
    password: str
    tzOffset: Optional[int] = None   # getTimezoneOffset() — for the local login day


class AuthGoogleIn(BaseModel):
    credential: str   # Google ID token (JWT) returned by Sign in with Google
    tzOffset: Optional[int] = None


class RegisterIn(BaseModel):
    """Public signup payload. `honeypot` should be empty — it's a hidden form
    field most bots auto-fill. Any non-empty value gets a 200 OK with no
    side-effects so the bot moves on without learning it was blocked."""
    email: str
    password: str
    displayName: str
    expertise: str = ""        # optional self-reported expertise dropdown
    profile: dict[str, Any] = Field(default_factory=dict)  # demographic/clinical fields collected at signup
    honeypot: str = ""         # bot trap; must be empty


class VerifyIn(BaseModel):
    email: str
    code: str


class EmailIn(BaseModel):
    email: str


class ResetIn(BaseModel):
    email: str
    code: str
    newPassword: str


class SessionIn(BaseModel):
    participant: dict[str, Any] = Field(default_factory=dict)
    sampleSeed: Optional[int] = None


class ProgressIn(BaseModel):
    sessionId: str
    trial: dict[str, Any]


class ProgressBatchIn(BaseModel):
    sessionId: str
    trials: list[dict[str, Any]]


class ResultsIn(BaseModel):
    sessionId: str
    result: dict[str, Any] = Field(default_factory=dict)
    stopReason: Optional[str] = None
    nQuestions: Optional[int] = None


class AdminGenIn(BaseModel):
    count: int = Field(ge=1, le=1000)
    prefix: str = "cortex"


class TrainingStartIn(BaseModel):
    taskFocus: Optional[str] = None


class TrainingFinalizeIn(BaseModel):
    trainingId: str
    nItems: Optional[int] = None
    summary: Optional[dict[str, Any]] = None


class TrainingProgressIn(BaseModel):
    """One or more real training trials to persist (L1). Each point carries
    {taskK, segId?, ell?, theta?, sd?, rt?, seqInSession?} plus, since Phase
    L2, the response record {pick?, yStar?, isCorrect?, feedbackShown?,
    rtMs?, shownClientUtc?, answeredClientUtc?}. The server sets
    is_real/phase/code from the authenticated training session."""
    trainingId: str
    points: list[dict[str, Any]] = Field(default_factory=list)


class EngineStartIn(BaseModel):
    """Start (or rebuild) an engine-trainer sitting (Phase L3). segIds =
    the client's drawn training-bank pool (media the client can load);
    restrictTaskKs = the regimen's weak-task set (None = all tasks)."""
    trainingId: str
    segIds: list[int] = Field(default_factory=list)
    restrictTaskKs: Optional[list[int]] = None


class EngineRecordIn(BaseModel):
    """One answered engine-served item: the pick updates the belief; the
    full response record lands via the existing checkpoint outbox.

    The item's task is not accepted from the client — the trainer already
    knows which item it served, so a client-supplied taskK could only
    disagree with it. Older SPA builds still post one; pydantic ignores
    unknown fields, so those keep working."""
    trainingId: str
    segId: int
    pick: int


class ReportIn(BaseModel):
    """Public support/feedback report. `client` carries browser-collected
    diagnostics (OS, timezone, screen, etc.) for troubleshooting."""
    username: str = ""
    email: str = ""
    message: str
    client: dict[str, Any] = Field(default_factory=dict)


class ConsentIn(BaseModel):
    """Record a participant's consent acceptance (Phase O1)."""
    consentType: str = "research_irb"
    consentVersion: str
    irbProtocolId: Optional[str] = None


class ConsentWithdrawIn(BaseModel):
    consentType: Optional[str] = None    # None → withdraw all consent types


class ProfileIn(BaseModel):
    """Edit account/profile details (Settings page). All fields optional so a
    partial update is fine; None/omitted leaves a field unchanged. `profile`
    must default to None (NOT {}): a `{}` default would silently wipe the
    stored profile on any request that omits the field."""
    displayName: Optional[str] = None
    expertise: Optional[str] = None
    profile: Optional[dict[str, Any]] = None
    trainingReminders: Optional[bool] = None   # digest opt-in/out (Settings toggle)


class AwardAckIn(BaseModel):
    awardId: str


class PasswordChangeIn(BaseModel):
    currentPassword: str
    newPassword: str


class EmailChangeIn(BaseModel):
    newEmail: str
    password: str


class CohortCreateIn(BaseModel):
    name: str


class CohortMemberIn(BaseModel):
    """Manager-side member operations address users by their 9-digit public
    id only — the internal participant code never crosses the API."""
    publicId: str


class ClientErrorIn(BaseModel):
    """SPA crash report (src/telemetry.ts → POST /api/client-error). Length
    caps tame hostile payloads; the endpoint is public (crashes can happen
    pre-auth) and rate-limited."""
    message: str = Field(max_length=500)
    stack: str = Field("", max_length=4000)
    url: str = Field("", max_length=300)       # pathname only (no query params)
    surface: str = Field("", max_length=20)    # "desktop" | "mobile"
    ua: str = Field("", max_length=300)

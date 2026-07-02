// Backend API client. The server is tiny (PLAN §8): authenticate once, fetch
// the bundle URL, ingest results — no per-question round-trips. All gated
// calls carry the JWT as a Bearer token.
//
// Base URL: VITE_API_BASE if set, else same-origin "" (the SPA is served by
// the same uvicorn that serves /api in the single-process local setup). In
// `vite dev` the dev server proxies /api → :8000 (see vite.config.ts), so the
// empty base works there too.

import type { SessionBank } from "./bundle";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const TOKEN_KEY = "cortex_token";
const DISPLAY_NAME_KEY = "cortex_display_name";

export interface Manifest {
  bundleUrl: string;
  version: string | null;   // real bundle id (null only if no bank configured)
  sessionSample: number;    // advisory; the question SET comes from startSession
}

export interface StartSessionResult {
  sessionId: string;
  sampleSeed: number;
  bank: SessionBank;        // the server-drawn per-session question subset
}

export interface TrialCheckpoint {
  trialIndex: number;
  segId?: number;
  taskK?: number;
  pick?: number;
  isCorrect?: boolean;
  reactionMs?: number;
  diag?: unknown;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// Raised by login() when the server says the account exists but the email
// isn't verified yet (403 {error:"email_not_verified"}). The UI catches this
// to route to the verify screen rather than show a generic credentials error.
export class EmailNotVerifiedError extends Error {
  constructor(public email: string) {
    super("email_not_verified");
  }
}

// The session token lives in sessionStorage, NOT localStorage: it is scoped to
// the tab and is cleared when the tab/window is closed, so reopening the app
// requires signing in again (no persistent cached session). One-time cleanup of
// any token left in localStorage by the previous (persistent) scheme.
try { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(DISPLAY_NAME_KEY); } catch { /* private mode */ }

export function getToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}
export function setToken(t: string): void {
  sessionStorage.setItem(TOKEN_KEY, t);
}
export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}
export function isAuthed(): boolean {
  return !!getToken();
}

// Logged-in display name, kept alongside the token (same tab-scoped lifetime) so
// the shell can greet the clinician without an extra round-trip.
export function getDisplayName(): string | null {
  return sessionStorage.getItem(DISPLAY_NAME_KEY);
}
export function setDisplayName(name: string): void {
  if (name) sessionStorage.setItem(DISPLAY_NAME_KEY, name);
}

// Sign out: drop the token AND the display name. Pending results stay queued in
// localStorage (they belong to the device, not the session) and flush on the
// next sign-in.
export function logout(): void {
  clearToken();
  sessionStorage.removeItem(DISPLAY_NAME_KEY);
  try { sessionStorage.removeItem("cortex-welcome-seen"); } catch { /* private mode */ }
}

async function parse(res: Response): Promise<any> {
  const text = await res.text();
  let body: any = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { error: text };
  }
  if (!res.ok) {
    throw new ApiError(res.status, body?.error || res.statusText);
  }
  return body;
}

async function authedFetch(path: string, init: RequestInit = {}): Promise<any> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 401) clearToken();
  return parse(res);
}

// ── public ───────────────────────────────────────────────────────
// Sign in. On success stores the JWT. A 403 {error:"email_not_verified"} is
// re-thrown as EmailNotVerifiedError so the UI can route to the verify screen
// instead of showing a credentials error.
export async function login(email: string, password: string): Promise<{
  email: string; displayName: string;
}> {
  const res = await fetch(`${API_BASE}/api/auth`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // tzOffset = getTimezoneOffset() (minutes UTC is ahead of local) so the
    // sign-in is logged on the user's LOCAL day, not the UTC day.
    body: JSON.stringify({ email, password, tzOffset: new Date().getTimezoneOffset() }),
  });
  if (res.status === 403) {
    const body = await res.json().catch(() => null);
    if (body?.error === "email_not_verified") throw new EmailNotVerifiedError(email);
    throw new ApiError(403, body?.error || res.statusText);
  }
  const body = await parse(res);
  setToken(body.token);
  setDisplayName(body.displayName);
  return { email: body.email, displayName: body.displayName };
}

// Sign in with Google. Sends the Google ID token to the backend, which
// verifies it and creates-or-signs-in the account; stores the app JWT on
// success. Same response shape as login().
export async function loginWithGoogle(credential: string): Promise<{
  email: string; displayName: string;
}> {
  const res = await fetch(`${API_BASE}/api/auth/google`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ credential, tzOffset: new Date().getTimezoneOffset() }),
  });
  const body = await parse(res);
  setToken(body.token);
  setDisplayName(body.displayName);
  return { email: body.email, displayName: body.displayName };
}

// Public open-signup. `honeypot` is the hidden form field; humans leave it
// empty. The backend accepts a non-empty value silently to avoid tipping off
// scanners that a bot trap exists. Register no longer returns a token: the
// participant must verify their email then sign in. `devCode` is only present
// in dev/CI (CORTEX_EMAIL_EXPOSE_CODE=1) and is never relied on in real UX.
export async function register(
  email: string, password: string, displayName: string, expertise: string,
  profile: Record<string, string>, honeypot: string,
): Promise<{ needsVerification: boolean; email: string; displayName: string; devCode?: string }> {
  const res = await fetch(`${API_BASE}/api/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, displayName, expertise, profile, honeypot }),
  });
  const body = await parse(res);
  setDisplayName(body?.displayName ?? displayName);
  return {
    needsVerification: !!body?.needsVerification,
    email: body?.email ?? email,
    displayName: body?.displayName ?? displayName,
    devCode: body?.devCode,
  };
}

// Confirm the 6-digit email-verification code. Throws ApiError(400) on an
// invalid/expired code, ApiError(429) on rate limiting.
export async function verifyCode(email: string, code: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/verify/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, code }),
  });
  await parse(res);
}

// Resend the email-verification code. `devCode` only present in dev/CI.
export async function resendCode(email: string): Promise<{ devCode?: string }> {
  const res = await fetch(`${API_BASE}/api/verify/resend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  const body = await parse(res);
  return { devCode: body?.devCode };
}

// Request a password-reset code. Always 200 (does not reveal whether the email
// is registered); `devCode` only present in dev/CI.
export async function requestReset(email: string): Promise<{ devCode?: string }> {
  const res = await fetch(`${API_BASE}/api/forgot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  const body = await parse(res);
  return { devCode: body?.devCode };
}

// Complete a password reset with the code + new password. Throws ApiError(400)
// on an invalid/expired code or a too-short password.
export async function resetPassword(
  email: string, code: string, newPassword: string,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, code, newPassword }),
  });
  await parse(res);
}

// Submit a support/feedback report (public, rate-limited). `client` carries
// browser diagnostics collected by the caller. Throws ApiError on failure
// (400 empty message, 429 rate limit).
export async function submitReport(payload: {
  username: string;
  email: string;
  message: string;
  client: Record<string, unknown>;
}): Promise<void> {
  const res = await fetch(`${API_BASE}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await parse(res);
}

export async function health(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}

// ── gated ────────────────────────────────────────────────────────
export function getManifest(): Promise<Manifest> {
  return authedFetch("/api/manifest");
}

// ── account / profile (Settings page) ────────────────────────────
export interface AccountProfile {
  email: string;
  displayName: string;
  expertise: string;
  authProvider: string;
  profile: Record<string, string>;
}

export function getProfile(): Promise<AccountProfile> {
  return authedFetch("/api/profile");
}

export function updateProfile(
  displayName: string, expertise: string, profile: Record<string, string>,
): Promise<{ ok: boolean }> {
  return authedFetch("/api/profile", {
    method: "PUT",
    body: JSON.stringify({ displayName, expertise, profile }),
  });
}

export function changePassword(currentPassword: string, newPassword: string): Promise<{ ok: boolean }> {
  return authedFetch("/api/account/password", {
    method: "POST",
    body: JSON.stringify({ currentPassword, newPassword }),
  });
}

export function changeEmail(newEmail: string, password: string): Promise<{ ok: boolean; email: string }> {
  return authedFetch("/api/account/email", {
    method: "POST",
    body: JSON.stringify({ newEmail, password }),
  });
}

// ── dashboard / learning-protocol (Phase 2 backend) ───────────────
// Read-only surfaces backing the shell. All real cert-result data — no sample
// data. Learning-protocol surfaces (streak/deck/trajectories) carry no data
// until the trainer ships.
// Cert-summary KPIs derived from the latest result.
export interface DashboardKpis {
  tasksCertified: number;
  tasksTotal: number;
  lastAssessed: string | null;   // finished_utc of the latest attempt
  meanAuroc: number | null;      // mean per-task AUROC, or null if none stored
}
// Per-task mastery-grid summary derived from the latest real result. ℓ/ℓ*/AUROC
// are null for a legacy (verdicts-only) result; verdict is always present.
export interface DashboardTask {
  taskK: number;
  code: string;
  label: string;
  ell: number | null;
  ellStar: number | null;
  theta: number | null;
  auroc: number | null;
  verdict: string;
}
// A real certification result blob (latest for the participant). Only the
// fields the dashboard reads are typed; the rest of the engine payload rides
// along untyped.
export interface CertResult {
  verdicts?: string[];
  roc?: Array<{ auroc?: number } | null>;
  [k: string]: unknown;
}
export interface DashboardData {
  result: CertResult | null;
  hasResult: boolean;
  tasks: DashboardTask[];
  kpis: DashboardKpis | null;   // null until a cert result exists
}

export interface TrajectoryPoint {
  taskK: number;
  phase: "eval" | "train" | "recert";
  ell: number;
  theta: number;
  sd: number;
  rt: number;
  ts: string;
}

export interface RegimenDeckEntry {
  taskK: number;
  code: string;
  label: string;
  ell: number;
  ellStar: number;
  new: number;
  learning: number;
  due: number;
}
export interface RegimenPlan {
  weeks: number;
  weekOf: number;
  deck: RegimenDeckEntry[];
}

// One completed certification attempt (real data) from /api/history.
export interface HistorySession {
  session_id: string;
  finished_utc: string | null;
  n_questions: number | null;
  stop_reason: string | null;
  result: CertResult;
}

// One question's breakdown row (GET /api/history/{id}/questions). Nullable
// fields are absent on legacy/partial trials or when ground truth is unknown.
export interface QuestionRow {
  q: number;                 // 1-based question number
  taskK: number | null;
  domain: string;            // tested-domain label (whose estimate this informed)
  answer: string | null;     // their answer: "Yes"/"No" (spike) or pattern label (IIIC)
  correct: string | null;    // correct answer: "Yes"/"No" (spike) or true pattern (IIIC)
  isCorrect: boolean | null; // answer matched the correct answer
  rt: number | null;         // reaction time, ms
  deltaR: number | null;     // per-question Δ info gain for the domain
  R: number | null;          // cumulative info gain (R)
  pi: number | null;         // pass-mass P(ℓ>ℓ*)
  ell: number | null;        // running skill ℓ
  theta: number | null;      // running bias θ
}

export function getDashboard(): Promise<DashboardData> {
  return authedFetch("/api/dashboard");
}
// Per-day activity levels for the consistency heatmap (LOCAL date → level:
// 1 = signed in, 2 = certification test, 3 = training completed). `tz` is the
// browser's getTimezoneOffset so the server reports days in the user's local time.
export function getActivity(): Promise<{ days: Record<string, number> }> {
  return authedFetch(`/api/activity?tz=${new Date().getTimezoneOffset()}`);
}
export function getRegimen(): Promise<{ regimen: RegimenPlan | null; sample: boolean }> {
  return authedFetch("/api/regimen");
}
export function getTrajectories(): Promise<{ trajectories: TrajectoryPoint[]; sample: boolean }> {
  return authedFetch("/api/trajectories");
}
export function listTrainingSessions(): Promise<{ sessions: unknown[] }> {
  return authedFetch("/api/training-sessions");
}
export function getHistory(): Promise<{ sessions: HistorySession[] }> {
  return authedFetch("/api/history");
}
export function getQuestions(sessionId: string): Promise<{ sessionId: string; nQuestions: number; questions: QuestionRow[] }> {
  return authedFetch(`/api/history/${encodeURIComponent(sessionId)}/questions`);
}
export function startTrainingSession(taskFocus?: string): Promise<{ trainingId: string }> {
  return authedFetch("/api/training-sessions", {
    method: "POST",
    body: JSON.stringify({ taskFocus: taskFocus ?? null }),
  });
}
export function finalizeTrainingSession(
  trainingId: string, nItems: number, summary?: Record<string, unknown>,
): Promise<{ ok: boolean }> {
  return authedFetch("/api/training-sessions/finalize", {
    method: "POST",
    body: JSON.stringify({ trainingId, nItems, summary: summary ?? null }),
  });
}
export function appendTrajectories(points: api_TrajectoryPointIn[]): Promise<{ ok: boolean }> {
  return authedFetch("/api/trajectories", {
    method: "POST",
    body: JSON.stringify({ points }),
  });
}
// Record the participant's consent acceptance (Phase O1). Authenticated; the
// server stamps the accept time + IP. Best-effort at the call site.
export function recordConsent(consentVersion: string, irbProtocolId?: string,
                              consentType = "research_irb"): Promise<{ ok: boolean }> {
  return authedFetch("/api/consent", {
    method: "POST",
    body: JSON.stringify({ consentType, consentVersion, irbProtocolId }),
  });
}
// The shape POSTed to /api/trajectories (a subset of TrajectoryPoint).
export interface api_TrajectoryPointIn {
  taskK: number;
  phase?: "eval" | "train" | "recert";
  ell?: number;
  theta?: number;
  sd?: number;
  rt?: number;
}

// Start a sitting: the server draws this participant's balanced, spacing-aware
// question subset (goal 3) and returns it with the session id + sample seed.
export async function startSession(
  participant: Record<string, unknown>,
): Promise<StartSessionResult> {
  return authedFetch("/api/session", {
    method: "POST",
    body: JSON.stringify({ participant }),
  });
}

// One representative IIIC segment for the in-context tutorial (no full-manifest
// fetch) — returned as a 1-segment bank the client wraps in a Bundle.
export function tutorialExample(): Promise<SessionBank> {
  return authedFetch("/api/tutorial-example");
}

// Fire-and-forget per-trial checkpoint (crash-safety). Never throws into the
// UI — a dropped checkpoint must not interrupt the test.
export function postProgress(sessionId: string, trial: TrialCheckpoint): void {
  authedFetch("/api/progress", {
    method: "POST",
    body: JSON.stringify({ sessionId, trial }),
  }).catch(() => {});
}

export async function postResults(
  sessionId: string,
  result: unknown,
  stopReason: string,
  nQuestions: number,
): Promise<void> {
  await authedFetch("/api/results", {
    method: "POST",
    body: JSON.stringify({ sessionId, result, stopReason, nQuestions }),
  });
}

// ── result delivery with local persistence + retry (PLAN §8) ──────
// The final results upload must survive a flaky network or a tab close. We
// enqueue the payload in localStorage first, then attempt delivery; anything
// undelivered is retried by flushPendingResults() on the next authed load.
const PENDING_KEY = "cortex_pending_results";

interface PendingResult {
  sessionId: string;
  result: unknown;
  stopReason: string;
  nQuestions: number;
}

function loadPending(): PendingResult[] {
  try {
    return JSON.parse(localStorage.getItem(PENDING_KEY) || "[]");
  } catch {
    return [];
  }
}
function savePending(list: PendingResult[]): void {
  localStorage.setItem(PENDING_KEY, JSON.stringify(list));
}

// Enqueue then deliver. Resolves true if delivered now, false if it was kept
// for retry (the caller still shows results — the data is safe locally).
export async function submitResults(
  sessionId: string,
  result: unknown,
  stopReason: string,
  nQuestions: number,
): Promise<boolean> {
  const list = loadPending().filter((p) => p.sessionId !== sessionId);
  list.push({ sessionId, result, stopReason, nQuestions });
  savePending(list);
  return (await flushPendingResults()).includes(sessionId);
}

// Retry every queued result; returns the session ids successfully delivered.
export async function flushPendingResults(): Promise<string[]> {
  const list = loadPending();
  if (!list.length || !getToken()) return [];
  const delivered: string[] = [];
  const remaining: PendingResult[] = [];
  for (const p of list) {
    try {
      await postResults(p.sessionId, p.result, p.stopReason, p.nQuestions);
      delivered.push(p.sessionId);
    } catch {
      remaining.push(p);
    }
  }
  savePending(remaining);
  return delivered;
}

// ── visualization videos (#8) ─────────────────────────────────────
// POST the particle-cloud trajectory (binary Float32 t/l/w) + a JSON meta to
// the backend, which renders the 4 MP4s as a background JOB: submit returns
// {jobId} immediately, we poll its status, then download the zip. (The old
// contract held one HTTP request open for the full ~10-minute render, which
// NATs/proxies routinely cut.) The Promise still resolves to the zip Blob,
// so callers are unchanged.
export interface VizPayload {
  shape: [number, number, number];
  taskCodes: string[];
  segIds: number[];
  trials: unknown[];
  certificate: unknown;
  participantName?: string;
  t: Float32Array;
  l: Float32Array;
  w: Float32Array;
}

const VIDEO_POLL_MS = 5_000;                 // status poll cadence
const VIDEO_TIMEOUT_MS = 30 * 60_000;        // give up after 30 min (renders run ~10)

export async function requestVideos(p: VizPayload): Promise<Blob> {
  const { t, l, w, ...meta } = p;
  const fd = new FormData();
  fd.append("meta", JSON.stringify(meta));
  fd.append("t", new Blob([t.buffer as ArrayBuffer]), "t.f32");
  fd.append("l", new Blob([l.buffer as ArrayBuffer]), "l.f32");
  fd.append("w", new Blob([w.buffer as ArrayBuffer]), "w.f32");
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/api/videos`, { method: "POST", body: fd, headers });
  if (!res.ok) throw new Error(`render submit failed (${res.status}): ${await res.text().catch(() => "")}`);
  const { jobId } = (await res.json()) as { jobId: string };

  const deadline = Date.now() + VIDEO_TIMEOUT_MS;
  for (;;) {
    await new Promise((r) => setTimeout(r, VIDEO_POLL_MS));
    const st = await fetch(`${API_BASE}/api/videos/${jobId}`, { headers });
    if (!st.ok) throw new Error(`render status failed (${st.status}): ${await st.text().catch(() => "")}`);
    const body = (await st.json()) as { status: string; error?: string };
    if (body.status === "done") break;
    if (body.status === "error") throw new Error(`render failed: ${body.error ?? "unknown error"}`);
    if (Date.now() > deadline) throw new Error("render timed out — please try again");
  }

  const dl = await fetch(`${API_BASE}/api/videos/${jobId}/download`, { headers });
  if (!dl.ok) throw new Error(`render download failed (${dl.status})`);
  return dl.blob();
}

// Backend API client. The server is tiny (PLAN §8): authenticate once, fetch
// the bundle URL, ingest results — no per-question round-trips. All gated
// calls carry the JWT as a Bearer token.
//
// Base URL: VITE_API_BASE if set, else same-origin "" (the SPA is served by
// the same uvicorn that serves /api in the single-process local setup). In
// `vite dev` the dev server proxies /api → :8000 (see vite.config.ts), so the
// empty base works there too.

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const TOKEN_KEY = "cortex_token";
const DISPLAY_NAME_KEY = "cortex_display_name";

export interface Manifest {
  bundleUrl: string;
  version: string;
  sessionSample: number;
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

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(t: string): void {
  localStorage.setItem(TOKEN_KEY, t);
}
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}
export function isAuthed(): boolean {
  return !!getToken();
}

// Logged-in display name, persisted alongside the token so the shell can greet
// the clinician without an extra round-trip. Set on login/register success.
export function getDisplayName(): string | null {
  return localStorage.getItem(DISPLAY_NAME_KEY);
}
export function setDisplayName(name: string): void {
  if (name) localStorage.setItem(DISPLAY_NAME_KEY, name);
}

// Sign out: drop the token AND the display name. Pending results stay queued
// (they belong to the device, not the session) and flush on the next sign-in.
export function logout(): void {
  clearToken();
  localStorage.removeItem(DISPLAY_NAME_KEY);
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
    body: JSON.stringify({ email, password }),
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

// Public open-signup. `honeypot` is the hidden form field; humans leave it
// empty. The backend accepts a non-empty value silently to avoid tipping off
// scanners that a bot trap exists. Register no longer returns a token: the
// participant must verify their email then sign in. `devCode` is only present
// in dev/CI (CORTEX_EMAIL_EXPOSE_CODE=1) and is never relied on in real UX.
export async function register(
  email: string, password: string, displayName: string, expertise: string,
  honeypot: string,
): Promise<{ needsVerification: boolean; email: string; displayName: string; devCode?: string }> {
  const res = await fetch(`${API_BASE}/api/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, displayName, expertise, honeypot }),
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

// ── dashboard / learning-protocol (Phase 2 backend) ───────────────
// Read-only surfaces backing the shell. KPIs + sample task ℓ/θ/RT are
// illustrative until the trainer is ported (flagged `sample: true`).
export interface DashboardKpis {
  streak: number;
  dueToday: { new: number; learning: number; review: number };
  nextRecertDays: number;
}
// Per-task mastery-grid summary. ℓ/ℓ*/auroc are sample; verdict/auroc are
// overridden from the real cert result when hasResult is true.
export interface DashboardTask {
  taskK: number;
  code: string;
  label: string;
  ell: number;
  ellStar: number;
  auroc: number;
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
  kpis: DashboardKpis;
  sample: boolean;
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

export function getDashboard(): Promise<DashboardData> {
  return authedFetch("/api/dashboard");
}
export function getRegimen(): Promise<{ regimen: RegimenPlan; sample: boolean }> {
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
// The shape POSTed to /api/trajectories (a subset of TrajectoryPoint).
export interface api_TrajectoryPointIn {
  taskK: number;
  phase?: "eval" | "train" | "recert";
  ell?: number;
  theta?: number;
  sd?: number;
  rt?: number;
}

export async function createSession(
  participant: Record<string, unknown>,
  sampleSeed: number,
): Promise<string> {
  const body = await authedFetch("/api/session", {
    method: "POST",
    body: JSON.stringify({ participant, sampleSeed }),
  });
  return body.sessionId;
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
// the backend, which runs the desktop renderers and returns a zip of the 4 MP4s.
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
  if (!res.ok) throw new Error(`render failed (${res.status}): ${await res.text().catch(() => "")}`);
  return res.blob();
}

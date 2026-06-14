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
export async function login(email: string, password: string): Promise<{
  email: string; displayName: string;
}> {
  const res = await fetch(`${API_BASE}/api/auth`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const body = await parse(res);
  setToken(body.token);
  return { email: body.email, displayName: body.displayName };
}

// Public open-signup. `honeypot` is the hidden form field; humans leave it
// empty. The backend accepts a non-empty value silently to avoid tipping off
// scanners that a bot trap exists.
export async function register(
  email: string, password: string, displayName: string, expertise: string,
  honeypot: string,
): Promise<{ email: string; displayName: string }> {
  const res = await fetch(`${API_BASE}/api/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, displayName, expertise, honeypot }),
  });
  const body = await parse(res);
  // honeypot case: server returned 200 but no token. Don't store anything.
  if (body?.token) setToken(body.token);
  return { email: body?.email ?? email, displayName: body?.displayName ?? displayName };
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

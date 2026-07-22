// Backend API client. The server is intentionally small: authenticate once,
// receive the per-session draw (bundle identity + question set travel inside
// POST /api/session), ingest results — no per-question round-trips. All gated
// calls carry the JWT as a Bearer token.
//
// Base URL: VITE_API_BASE if set, else same-origin "" (the SPA is served by
// the same uvicorn that serves /api in the single-process local setup). In
// `vite dev` the dev server proxies /api → :8000 (see vite.config.ts), so the
// empty base works there too.

import type { SessionBank } from "./bundle";
import type {
  RequestedComputeMode, TerminationPolicyName,
} from "../engine/types";
import type { CohortSummary } from "./api/cohorts";
import { Outbox, transportFetch } from "./transport";
import {
  API_BASE,
  ApiError,
  EmailNotVerifiedError,
  authedFetch,
  getToken,
  parseResponse as parse,
  setDisplayName,
  setToken,
} from "./api/core";

export {
  ApiError,
  EmailNotVerifiedError,
  clearToken,
  getDisplayName,
  getToken,
  isAuthed,
  logout,
  setDisplayName,
  setToken,
} from "./api/core";

export {
  acceptCohortInvite,
  cancelCohortEmailInvite,
  createCohort,
  declineCohortInvite,
  deleteCohort,
  getCohort,
  getCohortPerformance,
  inviteToCohort,
  inviteToCohortByEmail,
  leaveCohort,
  listCohorts,
  removeCohortMember,
  type CohortDetail,
  type CohortMemberInfo,
  type CohortMemberSeries,
  type CohortPerformance,
  type CohortPoint,
  type CohortSummary,
  type CohortTaskSeries,
} from "./api/cohorts";

export interface StartSessionResult {
  sessionId: string;
  sampleSeed: number;
  bank: SessionBank;        // the server-drawn per-session question subset
  terminationPolicy: TerminationPolicyName;
  computeMode: RequestedComputeMode;
}

export interface ActiveSession {
  sessionId: string;
  startedUtc: string;
  computeMode: RequestedComputeMode;
  bank: SessionBank;        // the sitting's ORIGINAL drawn pool, verbatim order
  trials: { trialIndex: number; segId: number; pick: number }[];
}

export interface TrialCheckpoint {
  trialIndex: number;
  segId?: number;
  taskK?: number;
  pick?: number;
  isCorrect?: boolean;
  reactionMs?: number;
  diag?: unknown;
  // Client wall-clock at item render / at answer (Phase L2): anchors session
  // load and between-session structure for the learning model; reactionMs
  // stays the authoritative RT delta.
  shownClientUtc?: string;
  answeredClientUtc?: string;
}

// ── public ───────────────────────────────────────────────────────
// Sign in. On success stores the JWT. A 403 {error:"email_not_verified"} is
// re-thrown as EmailNotVerifiedError so the UI can route to the verify screen
// instead of showing a credentials error.
export async function login(email: string, password: string): Promise<{
  email: string; displayName: string;
}> {
  const res = await transportFetch(`${API_BASE}/api/auth`, {
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
  const res = await transportFetch(`${API_BASE}/api/auth/google`, {
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
  const res = await transportFetch(`${API_BASE}/api/register`, {
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
  const res = await transportFetch(`${API_BASE}/api/verify/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, code }),
  });
  await parse(res);
}

// Polled by the verify screen: did the verification email we just sent
// hard-bounce (mailbox doesn't exist)? Server-side flag comes from the SES
// bounce webhook. Anti-oracle: false for unknown/verified emails.
export async function verifyStatus(email: string): Promise<{ undeliverable: boolean }> {
  const res = await transportFetch(`${API_BASE}/api/verify/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  const body = await parse(res);
  return { undeliverable: !!body?.undeliverable };
}

// Resend the email-verification code. `devCode` only present in dev/CI.
export async function resendCode(email: string): Promise<{ devCode?: string }> {
  const res = await transportFetch(`${API_BASE}/api/verify/resend`, {
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
  const res = await transportFetch(`${API_BASE}/api/forgot`, {
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
  const res = await transportFetch(`${API_BASE}/api/reset`, {
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
  const res = await transportFetch(`${API_BASE}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await parse(res);
}

// ── account / profile (Settings page) ────────────────────────────
export interface AccountProfile {
  email: string;
  displayName: string;
  expertise: string;
  authProvider: string;
  /** Unique 9-digit account id, shown in Settings (assigned server-side). */
  publicId?: string;
  profile: Record<string, string>;
  /** Training-reminder email digest on/off (defaults on server-side). */
  trainingReminders?: boolean;
}

export function getProfile(): Promise<AccountProfile> {
  return authedFetch("/api/profile", {}, { retries: 2 });
}

export function updateProfile(
  displayName: string, expertise: string, profile: Record<string, string>,
): Promise<{ ok: boolean }> {
  return authedFetch("/api/profile", {
    method: "PUT",
    body: JSON.stringify({ displayName, expertise, profile }),
  });
}

// Partial PUT: the server leaves omitted fields unchanged, so the reminders
// toggle saves instantly without resending the whole profile form.
export function setTrainingReminders(on: boolean): Promise<{ ok: boolean }> {
  return authedFetch("/api/profile", {
    method: "PUT",
    body: JSON.stringify({ trainingReminders: on }),
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
  // Extremes, not a mean: the seven domains are disjoint discrimination
  // tasks, so a cross-domain average estimates nothing; worst/best identify
  // the range (and the next training target) instead. Null until a result
  // with per-task AUROCs exists.
  worstDomain: { label: string; auroc: number } | null;
  bestDomain: { label: string; auroc: number } | null;
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
  terminationPolicy?: TerminationPolicyName;
  domainStatuses?: string[];
  determinations?: string[];
  roc?: Array<{ auroc?: number } | null>;
  [k: string]: unknown;
}
export interface DashboardData {
  result: CertResult | null;
  hasResult: boolean;
  tasks: DashboardTask[];
  kpis: DashboardKpis | null;   // null until a cert result exists
  trainingEnabled?: boolean;    // training-exposure flag (all/cohort/off); absent on legacy responses
  trainingResumable?: boolean;  // the training program has started (CTA: Resume vs Start)
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
// The learner's measured per-task cert posterior (engine coords), handed to the
// trainer to seed its belief clouds. Structurally matches the trainer's TaskPrior.
export interface RegimenTaskPrior {
  taskK: number;
  ell: number;
  theta: number;
  sd: number | null;
}
// One trial of the source cert sitting's raw test stream (learning handoff
// contract v1.1 §2a): served order, full n-way pick. A replay-capable
// trainer seeds its belief by re-observing this stream; a client seeds from
// EITHER this OR `prior`, never both (double counting).
export interface RegimenTestTrial {
  trialIndex: number;
  segId: number;
  taskK: number;
  pick: number;
}
export interface RegimenPlan {
  weeks: number;
  weekOf: number;
  deck: RegimenDeckEntry[];
  prior?: RegimenTaskPrior[];
  testStream?: RegimenTestTrial[];
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
  ell: number | null;        // running skill ℓ
  theta: number | null;      // running bias θ
}

// In-flight coalescing for the dashboard read: on first paint both the Shell
// (hasResult/trainingEnabled) and the DashboardSurface (full payload) request
// it. Sharing the pending promise collapses those into ONE round-trip. The
// entry clears the moment it settles, so a later mount always refetches fresh
// data — this is request dedup, not a stale cache.
let _dashInFlight: Promise<DashboardData> | null = null;
export function getDashboard(): Promise<DashboardData> {
  if (_dashInFlight) return _dashInFlight;
  _dashInFlight = authedFetch("/api/dashboard", {}, { retries: 2 })
    .finally(() => { _dashInFlight = null; });
  return _dashInFlight;
}
// Per-day activity levels for the consistency heatmap (LOCAL date → level:
// 1 = signed in, 2 = certification test, 3 = training completed). `tz` is the
// browser's getTimezoneOffset so the server reports days in the user's local time.
export function getActivity(): Promise<{ days: Record<string, number> }> {
  return authedFetch(`/api/activity?tz=${new Date().getTimezoneOffset()}`, {}, { retries: 2 });
}
export function getRegimen(): Promise<{ regimen: RegimenPlan | null; sample: boolean }> {
  return authedFetch("/api/regimen", {}, { retries: 2 });
}
export function getTrajectories(): Promise<{ trajectories: TrajectoryPoint[]; sample: boolean }> {
  return authedFetch("/api/trajectories", {}, { retries: 2 });
}
export function getHistory(): Promise<{ sessions: HistorySession[] }> {
  return authedFetch("/api/history", {}, { retries: 2 });
}
export function getQuestions(sessionId: string): Promise<{ sessionId: string; nQuestions: number; questions: QuestionRow[] }> {
  return authedFetch(`/api/history/${encodeURIComponent(sessionId)}/questions`, {}, { retries: 2 });
}
export function startTrainingSession(
  taskFocus?: string,
): Promise<{ trainingId: string; engineMode?: boolean }> {
  return authedFetch("/api/training-sessions", {
    method: "POST",
    body: JSON.stringify({ taskFocus: taskFocus ?? null }),
  });
}

// ── engine trainer (Phase L3): the server-side learning engine drives the
// sitting; the client renders + collects answers. Awaited (not outboxed):
// the engine's next decision depends on the response landing.
export interface EngineItem {
  task: number; segId: number; s: number; sSd: number; yStar: number;
  mode: string;
  link?: string;   // 'binary' one-vs-rest | 'nway' full identification (L4)
}
export interface EngineTaskSnapshot {
  task: number; mastered: boolean; skill: number; theta: number; sd: number;
  passMass: number; trainability: number | null;
}
export interface EngineStepResponse {
  item: EngineItem | null;
  snapshot: EngineTaskSnapshot[];
  allMastered: boolean;
  done?: boolean;
  seeded?: number;
  rebuiltSeq?: number;
  // /start only: per-task trainability report (never serving-blocking in
  // practice mode) + surviving unique ancestors of the seeding cloud.
  attainability?: Record<string, number>;
  seedUnique?: number;
}
export function engineStart(
  trainingId: string, segIds: number[], restrictTaskKs?: number[],
): Promise<EngineStepResponse> {
  return authedFetch("/api/training-engine/start", {
    method: "POST",
    body: JSON.stringify({ trainingId, segIds,
                           restrictTaskKs: restrictTaskKs ?? null }),
  });
}
export function engineRecord(
  trainingId: string, segId: number, taskK: number, pick: number,
): Promise<EngineStepResponse> {
  return authedFetch("/api/training-engine/record", {
    method: "POST",
    body: JSON.stringify({ trainingId, segId, taskK, pick }),
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
// Build (and activate) a training regimen from the latest cert result: one track
// per non-PASSED task. The server derives the weak set + ℓ*.
export function createRegimen(): Promise<{ regimenId: string; regimen: RegimenPlan }> {
  return authedFetch("/api/regimen", { method: "POST", body: "{}" });
}
export interface api_TrainingPointIn {
  taskK: number;
  segId?: number;
  ell?: number;
  theta?: number;
  sd?: number;
  rt?: number;
  seqInSession?: number;
  // Phase L2 response record (persisted into training_trials server-side).
  pick?: number;
  yStar?: number;
  isCorrect?: boolean;
  feedbackShown?: string;
  shownClientUtc?: string;
  answeredClientUtc?: string;
  // Phase L4 serving metadata (mode: skill/bias/review; link: binary/nway).
  mode?: string;
  link?: string;
}
// Fire-and-forget per-answer training checkpoint (crash-safety; the same
// contract as the exam's postProgress): unsent points queue in the outbox
// and re-flush on each later answer, and the server dedupes on
// (trainingId, seqInSession), so a retry after an ambiguous failure can
// never duplicate trajectory rows. In-memory by design: a tab crash loses
// the outbox exactly like it loses the sitting.
const trainingOutbox = new Outbox<{ trainingId: string; points: api_TrainingPointIn[] }>({
  send: (item) => authedFetch("/api/training-progress", {
    method: "POST",
    body: JSON.stringify(item),
  }, { timeoutMs: 15_000 }),
  // Permanent rejections (bad payload / unknown sitting; not an expired-token
  // 401, which heals on re-login) must not wedge the queue.
  shouldDrop: (e) => e instanceof ApiError && e.status !== 401 && e.status < 500,
});

export function postTrainingProgress(
  trainingId: string, points: api_TrainingPointIn[],
): void {
  if (!points.length) return;
  trainingOutbox.push({ trainingId, points });
  void trainingOutbox.flush();
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
// Start a sitting: the server draws this participant's balanced, spacing-aware
// question subset (goal 3) and returns it with the session id + sample seed.
export async function startSession(
  participant: Record<string, unknown>,
): Promise<StartSessionResult> {
  // Retried: a transient failure here would otherwise abort the test before
  // it begins. The endpoint creates a session row, so a retry after an
  // ambiguous failure can leave a benign orphan in_progress row (no trials,
  // never finalized) — accepted trade for a start that survives blips.
  return authedFetch("/api/session", {
    method: "POST",
    body: JSON.stringify({ participant }),
  }, { retries: 2, timeoutMs: 30_000 });
}

// Draw the candidate pool for a TRAINING sitting. A separate endpoint from
// startSession() on purpose: the training draw is ungated — no post-training
// exam washout (so "Resume training" is never blocked after a sitting) and no
// test/train exposure exclusion (so the weak-domain pools aren't starved). It
// creates no exam session row. Returns the same SessionBank shape the client
// wraps in a Bundle.
export async function startTrainingBank(): Promise<{
  sampleSeed: number; bank: SessionBank;
}> {
  return authedFetch("/api/training-bank", {
    method: "POST",
    body: "{}",
  }, { retries: 2, timeoutMs: 30_000 });
}

// One representative IIIC segment for the in-context tutorial (no full-manifest
// fetch) — returned as a 1-segment bank the client wraps in a Bundle.
export function tutorialExample(): Promise<SessionBank> {
  return authedFetch("/api/tutorial-example", {}, { retries: 2 });
}

// The most recent resumable sitting (unfinished, recent, same bundle), with
// its original drawn pool + the checkpointed trials to replay — or null.
export function activeSession(): Promise<{
  active: ActiveSession | null;
  // Set while the post-training exam washout is in force (testing.py):
  // starting an exam will 409 until reopensAtUtc.
  washout: { reopensAtUtc: string } | null;
}> {
  return authedFetch("/api/session/active", {}, { retries: 2 });
}

// Light resume/washout status: what the dashboard CTA labels need, without
// the drawn-pool payload of /api/session/active (the pre-flight click still
// fetches the full thing).
export interface SessionStatus {
  examResumable: boolean;
  washout: { reopensAtUtc: string } | null;
}

// GET /api/bootstrap — the dashboard-entry sections in one round-trip. Each
// section carries EXACTLY its standalone endpoint's payload (same server-side
// builder); a section is null when it failed server-side (failure-isolated),
// and the consumer renders its empty state exactly as if the standalone call
// had failed. Fetch through bootstrapStore.bootstrapOnce() so the surfaces
// mounted on one dashboard entry share a single request.
export interface BootstrapData {
  dashboard: DashboardData | null;
  trajectories: { trajectories: TrajectoryPoint[]; sample: boolean } | null;
  regimen: { regimen: RegimenPlan | null; sample: boolean } | null;
  activity: { days: Record<string, number> } | null;
  session: SessionStatus | null;
  cohorts: { cohorts: CohortSummary[] } | null;
  awards: { pending: PendingMilestone[] } | null;
  errors?: Record<string, string>;
}

// ── awards: domain badges + milestones (Settings page + banner) ──
export interface Award {
  awardId: string;
  kind: "badge" | "milestone";
  key: string;                 // badge: domain code; milestone: slug
  label: string;
  detail: string | null;
  awardedUtc: string;
  revokedUtc: string | null;   // a badge lost to a later underperforming test
}

// The light bootstrap `awards` section: milestones awaiting their banner.
export interface PendingMilestone {
  awardId: string;
  key: string;
  label: string;
  detail: string | null;
  awardedUtc: string;
}

export function getAwards(): Promise<{ badges: Award[]; milestones: Award[] }> {
  return authedFetch("/api/awards", {}, { retries: 2 });
}

export function ackAward(awardId: string): Promise<{ ok: boolean }> {
  return authedFetch("/api/awards/ack", {
    method: "POST",
    body: JSON.stringify({ awardId }),
  });
}
export function bootstrap(): Promise<BootstrapData> {
  return authedFetch(`/api/bootstrap?tz=${new Date().getTimezoneOffset()}`,
                     {}, { retries: 2 });
}

// Fire-and-forget per-trial checkpoint (crash-safety). Never throws into the
// UI — and no longer silently DROPS on failure: /api/progress is an
// idempotent upsert on (sessionId, trialIndex), so unsent checkpoints queue
// in the outbox and re-flush on each later trial. A network blip mid-test
// heals instead of losing rows. (The final results blob independently
// carries every trial, so a checkpoint that never lands costs only mid-test
// crash granularity, not data. In-memory by design: a tab crash loses the
// outbox exactly like it loses the session.)
const progressOutbox = new Outbox<{ sessionId: string; trial: TrialCheckpoint }>({
  send: (item) => authedFetch("/api/progress", {
    method: "POST",
    body: JSON.stringify(item),
  }, { timeoutMs: 15_000 }),
  // Permanent rejections (bad payload / unknown session — anything 4xx except
  // an expired-token 401, which heals on re-login) must not wedge the queue.
  shouldDrop: (e) => e instanceof ApiError && e.status !== 401 && e.status < 500,
});

export function postProgress(sessionId: string, trial: TrialCheckpoint): void {
  progressOutbox.push({ sessionId, trial });
  void progressOutbox.flush();
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
  }, { retries: 2, timeoutMs: 60_000 });
}

// ── result delivery with local persistence + retry ───────────────
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
function savePending(list: PendingResult[]): boolean {
  // localStorage.setItem throws on quota-exceeded / private-mode. Guarded so a
  // storage failure can't reject submitResults and strand the user (the result
  // is still delivered from memory below); returns whether the durable-retry
  // copy was written.
  try {
    localStorage.setItem(PENDING_KEY, JSON.stringify(list));
    return true;
  } catch {
    return false;
  }
}

// Enqueue then deliver. Resolves true if delivered now, false if it was kept
// for retry (the caller still shows results — the data is safe locally).
export async function submitResults(
  sessionId: string,
  result: unknown,
  stopReason: string,
  nQuestions: number,
): Promise<boolean> {
  // Give any straggler trial checkpoints one last chance to land before the
  // session is finalized (best-effort — the result blob carries them anyway).
  await progressOutbox.flush();
  // Persist FIRST (durable retry copy), then attempt delivery of THIS result
  // directly. We don't route delivery through flushPendingResults() because if
  // the storage write failed (quota / private mode) the item wouldn't be in
  // localStorage to re-read — the direct POST still delivers it from memory.
  const list = loadPending().filter((p) => p.sessionId !== sessionId);
  list.push({ sessionId, result, stopReason, nQuestions });
  savePending(list);
  try {
    await postResults(sessionId, result, stopReason, nQuestions);
    // Delivered: drop it from the durable queue (best-effort).
    savePending(loadPending().filter((p) => p.sessionId !== sessionId));
    return true;
  } catch {
    return false;   // retained for flushPendingResults() on the next authed load
  }
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

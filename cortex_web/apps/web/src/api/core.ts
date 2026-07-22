import { transportFetch, type TransportOpts } from "../transport";

export const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

const TOKEN_KEY = "cortex_token";
const DISPLAY_NAME_KEY = "cortex_display_name";

export class ApiError extends Error {
  constructor(public status: number, message: string, public body?: unknown) {
    super(message);
  }
}

export class EmailNotVerifiedError extends Error {
  constructor(public email: string) {
    super("email_not_verified");
  }
}

// Remove tokens written by the retired persistent-login implementation.
try {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(DISPLAY_NAME_KEY);
} catch { /* unavailable storage/private mode */ }

export function getToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

export function isAuthed(): boolean {
  return !!getToken();
}

export function getDisplayName(): string | null {
  return sessionStorage.getItem(DISPLAY_NAME_KEY);
}

export function setDisplayName(name: string): void {
  if (name) sessionStorage.setItem(DISPLAY_NAME_KEY, name);
}

export function logout(): void {
  clearToken();
  sessionStorage.removeItem(DISPLAY_NAME_KEY);
  try { sessionStorage.removeItem("cortex-welcome-seen"); } catch { /* private mode */ }
}

export async function parseResponse(response: Response): Promise<any> {
  const text = await response.text();
  let body: any = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { error: text };
  }
  if (!response.ok) {
    throw new ApiError(response.status, body?.error || response.statusText, body);
  }
  return body;
}

// Session expiry (a 401 on a call we sent a token with) is announced once, so
// the app shell can route to sign-in instead of every surface independently
// swallowing the error and rendering an empty state with no explanation —
// which is what a 6-hour token expiry looked like from the dashboard.
type SessionExpiredListener = () => void;
const sessionExpiredListeners = new Set<SessionExpiredListener>();

export function onSessionExpired(listener: SessionExpiredListener): () => void {
  sessionExpiredListeners.add(listener);
  return () => { sessionExpiredListeners.delete(listener); };
}

export async function authedFetch(
  path: string,
  init: RequestInit = {},
  options: TransportOpts = {},
): Promise<any> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await transportFetch(
    `${API_BASE}${path}`,
    { ...init, headers },
    options,
  );
  if (response.status === 401) {
    // Re-read rather than reusing `token`: with several requests in flight the
    // first one here still sees the token and announces the expiry; the rest
    // find it already cleared and stay quiet, so listeners fire once. Both
    // statements run in the same synchronous turn, so they cannot interleave.
    const wasAuthenticated = !!getToken();
    clearToken();
    if (wasAuthenticated) {
      for (const listener of [...sessionExpiredListeners]) {
        try { listener(); } catch { /* a listener must not break the response */ }
      }
    }
  }
  return parseResponse(response);
}

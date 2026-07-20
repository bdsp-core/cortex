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
  if (response.status === 401) clearToken();
  return parseResponse(response);
}

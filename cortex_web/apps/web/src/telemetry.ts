// Global client-error telemetry: window "error" + "unhandledrejection" (and
// explicit calls, e.g. engine-worker failures) fire-and-forget to
// POST /api/client-error, which logs them to the server journal. This is the
// SPA's counterpart to the SES bounce events: field failures become ops
// signals instead of waiting for a user report.
//
// Telemetry must never add load or noise: each distinct message is sent at
// most once per page load, total reports are capped, failures are swallowed,
// and nothing is retried. Only the pathname is sent (never query params,
// which can carry auth-deep-link codes before boot consumes them).

const MAX_PER_LOAD = 5;
let sent = new Set<string>();

/** Test hook: clear the per-page-load dedupe state. */
export function resetTelemetryForTests(): void {
  sent = new Set();
}

export function reportClientError(
  message: string, stack?: string, surface?: string,
): void {
  try {
    const key = message.slice(0, 200);
    if (!key || sent.size >= MAX_PER_LOAD || sent.has(key)) return;
    sent.add(key);
    void fetch("/api/client-error", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      keepalive: true,   // survives page unload mid-crash
      body: JSON.stringify({
        message: message.slice(0, 500),
        stack: (stack ?? "").slice(0, 4000),
        url: window.location.pathname.slice(0, 300),
        surface: surface ?? "",
        ua: navigator.userAgent.slice(0, 300),
      }),
    }).catch(() => { /* telemetry never surfaces its own failures */ });
  } catch { /* never throw from telemetry */ }
}

export function installErrorTelemetry(surface: "desktop" | "mobile"): void {
  window.addEventListener("error", (e) => {
    reportClientError(String(e.message || "window error"),
      (e.error as Error | undefined)?.stack, surface);
  });
  window.addEventListener("unhandledrejection", (e) => {
    const r: unknown = e.reason;
    reportClientError(
      r instanceof Error ? r.message : String(r),
      r instanceof Error ? r.stack : undefined, surface);
  });
}

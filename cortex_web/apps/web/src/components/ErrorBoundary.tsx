import { Component, type ErrorInfo, type ReactNode } from "react";
import { reportClientError } from "../telemetry";

// App-wide safety net. Without this, any uncaught error thrown during render
// unmounts the entire React tree and leaves a blank white screen the user
// cannot escape from. This catches it, shows the error + a way out, logs
// the full component stack to the console for diagnosis, and reports it to
// /api/client-error (boundary-caught render errors don't reach the global
// window "error" hook in production builds, so without the explicit report
// they would be console-only). Styling is
// deliberately self-contained (no theme vars / CSS classes) so the fallback is
// legible even if the theme provider is the thing that unmounted.
interface Props { children: ReactNode }
interface State { error: Error | null; stack: string | null }

export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null, stack: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[cortex] uncaught render error:", error, info.componentStack);
    reportClientError(
      String(error.message || error),
      `${error.stack ?? ""}\n${info.componentStack ?? ""}`.trim(),
      "error-boundary",
    );
    this.setState({ stack: info.componentStack ?? null });
  }

  override render(): ReactNode {
    const { error, stack } = this.state;
    if (!error) return this.props.children;
    return (
      <div style={{
        minHeight: "100vh", background: "#f7f7f8", color: "#1a1a1e",
        fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: 24, boxSizing: "border-box",
      }}>
        <div style={{ maxWidth: 680, width: "100%" }}>
          <div style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
            Something went wrong
          </div>
          <div style={{ fontSize: 14, color: "#444", marginBottom: 16 }}>
            The page hit an unexpected error and stopped rendering. Nothing you've
            completed is lost — use a button below to get back to a working state.
          </div>
          <pre style={{
            background: "#fff", border: "1px solid #d33", color: "#a11",
            padding: 12, fontSize: 12, lineHeight: 1.5, overflow: "auto",
            maxHeight: "40vh", whiteSpace: "pre-wrap", wordBreak: "break-word",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", marginBottom: 16,
          }}>
            {String(error.message || error)}
            {stack ? `\n${stack}` : ""}
          </pre>
          <div style={{ display: "flex", gap: 10 }}>
            <button
              onClick={() => { window.location.href = "/"; }}
              style={{ padding: "10px 20px", fontSize: 14, fontWeight: 600, cursor: "pointer",
                background: "#0f766e", color: "#fff", border: "none", borderRadius: 0 }}>
              Return to start
            </button>
            <button
              onClick={() => window.location.reload()}
              style={{ padding: "10px 20px", fontSize: 14, cursor: "pointer",
                background: "#fff", color: "#1a1a1e", border: "1px solid #bbb", borderRadius: 0 }}>
              Reload
            </button>
          </div>
        </div>
      </div>
    );
  }
}

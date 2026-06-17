// App-wide light/dark theme. The active theme is the `data-theme` attribute on
// <html> (set before first paint by the inline script in index.html to avoid a
// flash); this provider mirrors it into React state and persists toggles to
// localStorage["cortex-theme"], the same key the design mockups use so a theme
// chosen on one surface carries across the whole app.

import { createContext, useCallback, useContext, useEffect, useState, ReactNode } from "react";

export type Theme = "light" | "dark";
const STORAGE_KEY = "cortex-theme";

function currentDomTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

interface ThemeCtx {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggle: () => void;
}

const Ctx = createContext<ThemeCtx | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(currentDomTheme);

  const apply = useCallback((t: Theme) => {
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", t);
    }
    try {
      localStorage.setItem(STORAGE_KEY, t);
    } catch {
      /* private mode / blocked storage: keep the in-memory theme only */
    }
    setThemeState(t);
  }, []);

  // Reconcile React state with whatever the no-flash script applied.
  useEffect(() => {
    setThemeState(currentDomTheme());
  }, []);

  const toggle = useCallback(() => apply(theme === "dark" ? "light" : "dark"), [theme, apply]);

  return <Ctx.Provider value={{ theme, setTheme: apply, toggle }}>{children}</Ctx.Provider>;
}

export function useTheme(): ThemeCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useTheme must be used within ThemeProvider");
  return c;
}

// Sun/moon toggle button (inline SVG line icons, never emoji). Reused by the
// nav rail and the auth surface.
export function ThemeToggle({ style }: { style?: React.CSSProperties }) {
  const { theme, toggle } = useTheme();
  const dark = theme === "dark";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      title={dark ? "Light theme" : "Dark theme"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 32,
        height: 32,
        padding: 0,
        background: "transparent",
        border: "1px solid var(--bd)",
        borderRadius: "var(--radius-ctl)",
        color: "var(--ink-subtle)",
        cursor: "pointer",
        ...style,
      }}
    >
      {dark ? (
        // moon
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z" />
        </svg>
      ) : (
        // sun
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      )}
    </button>
  );
}

import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ThemeProvider } from "./theme/ThemeProvider";
import { LanguageProvider } from "./i18n/LanguageProvider";

// The four standalone legal/info pages are their own URLs but are visited
// rarely; lazy-load them so their markup + copy don't ride in the first-paint
// bundle the signup/test flow pays for. App (the common case) stays eager.
const PrivacyPage = lazy(() => import("./components/PrivacyPage").then((m) => ({ default: m.PrivacyPage })));
const ReportPage = lazy(() => import("./components/ReportPage").then((m) => ({ default: m.ReportPage })));
const TermsPage = lazy(() => import("./components/TermsPage").then((m) => ({ default: m.TermsPage })));
const CitationPage = lazy(() => import("./components/CitationPage").then((m) => ({ default: m.CitationPage })));

// Minimal path-based routing: the SPA is otherwise a phase state machine, but
// /privacy, /report, /terms, and /citation must resolve as their own URLs
// (Caddy serves index.html for any path).
const path = window.location.pathname.replace(/\/+$/, "");
const isPrivacy = path === "/privacy";
const isReport = path === "/report";
const isTerms = path === "/terms";
const isCitation = path === "/citation";
const standalone = isPrivacy || isReport || isTerms || isCitation;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <ThemeProvider>
        <LanguageProvider>
          {standalone ? (
            <Suspense fallback={null}>
              {isPrivacy ? <PrivacyPage />
                : isReport ? <ReportPage />
                : isTerms ? <TermsPage />
                : <CitationPage />}
            </Suspense>
          ) : <App />}
        </LanguageProvider>
      </ThemeProvider>
    </ErrorBoundary>
  </StrictMode>,
);

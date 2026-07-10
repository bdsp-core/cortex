import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ThemeProvider } from "./theme/ThemeProvider";
import { LanguageProvider } from "./i18n/LanguageProvider";
import { detectPhone } from "./device";

// The four standalone legal/info pages are their own URLs but are visited
// rarely; lazy-load them so their markup + copy don't ride in the first-paint
// bundle the signup/test flow pays for. App (the common case) stays eager.
const PrivacyPage = lazy(() => import("./components/PrivacyPage").then((m) => ({ default: m.PrivacyPage })));
const ReportPage = lazy(() => import("./components/ReportPage").then((m) => ({ default: m.ReportPage })));
const TermsPage = lazy(() => import("./components/TermsPage").then((m) => ({ default: m.TermsPage })));
const CitationPage = lazy(() => import("./components/CitationPage").then((m) => ({ default: m.CitationPage })));

// The phone companion surface (src/mobile/ — auth + dashboard + results; the
// test itself is desktop-gated). Selected ONCE at boot (src/device.ts) and
// lazy, so desktop visitors never download it. This is the ONLY module
// allowed to import from mobile/ (enforced by mobile/boundary.test.ts).
const MobileApp = lazy(() => import("./mobile/MobileApp").then((m) => ({ default: m.MobileApp })));

// Minimal path-based routing: the SPA is otherwise a phase state machine, but
// /privacy, /report, /terms, and /citation must resolve as their own URLs
// (Caddy serves index.html for any path).
const path = window.location.pathname.replace(/\/+$/, "");
const isPrivacy = path === "/privacy";
const isReport = path === "/report";
const isTerms = path === "/terms";
const isCitation = path === "/citation";
const standalone = isPrivacy || isReport || isTerms || isCitation;
const phone = detectPhone();

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
          ) : phone ? (
            <Suspense fallback={null}><MobileApp /></Suspense>
          ) : <App />}
        </LanguageProvider>
      </ThemeProvider>
    </ErrorBoundary>
  </StrictMode>,
);

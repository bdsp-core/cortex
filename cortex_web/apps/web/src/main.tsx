import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { PrivacyPage } from "./components/PrivacyPage";
import { ReportPage } from "./components/ReportPage";
import { TermsPage } from "./components/TermsPage";
import { ThemeProvider } from "./theme/ThemeProvider";
import { LanguageProvider } from "./i18n/LanguageProvider";

// Minimal path-based routing: the SPA is otherwise a phase state machine, but
// /privacy, /report, and /terms must resolve as their own URLs (Caddy serves
// index.html for any path).
const path = window.location.pathname.replace(/\/+$/, "");
const isPrivacy = path === "/privacy";
const isReport = path === "/report";
const isTerms = path === "/terms";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <LanguageProvider>
        {isPrivacy ? <PrivacyPage /> : isReport ? <ReportPage /> : isTerms ? <TermsPage /> : <App />}
      </LanguageProvider>
    </ThemeProvider>
  </StrictMode>,
);

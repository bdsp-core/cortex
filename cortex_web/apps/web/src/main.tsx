import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { PrivacyPage } from "./components/PrivacyPage";
import { ReportPage } from "./components/ReportPage";
import { ThemeProvider } from "./theme/ThemeProvider";
import { LanguageProvider } from "./i18n/LanguageProvider";

// Minimal path-based routing: the SPA is otherwise a phase state machine, but
// /privacy and /report must resolve as their own URLs (Caddy serves index.html
// for any path).
const path = window.location.pathname.replace(/\/+$/, "");
const isPrivacy = path === "/privacy";
const isReport = path === "/report";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <LanguageProvider>
        {isPrivacy ? <PrivacyPage /> : isReport ? <ReportPage /> : <App />}
      </LanguageProvider>
    </ThemeProvider>
  </StrictMode>,
);

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { PrivacyPage } from "./components/PrivacyPage";
import { ThemeProvider } from "./theme/ThemeProvider";
import { LanguageProvider } from "./i18n/LanguageProvider";

// Minimal path-based routing: the SPA is otherwise a phase state machine, but
// /privacy must resolve as its own URL (Caddy serves index.html for any path).
const isPrivacy = window.location.pathname.replace(/\/+$/, "") === "/privacy";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <LanguageProvider>
        {isPrivacy ? <PrivacyPage /> : <App />}
      </LanguageProvider>
    </ThemeProvider>
  </StrictMode>,
);

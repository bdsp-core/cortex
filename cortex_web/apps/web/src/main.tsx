import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { PrivacyPage } from "./components/PrivacyPage";
import { ThemeProvider } from "./theme/ThemeProvider";

// Minimal path-based routing: the SPA is otherwise a phase state machine, but
// /privacy must resolve as its own URL (Caddy serves index.html for any path).
const isPrivacy = window.location.pathname.replace(/\/+$/, "") === "/privacy";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      {isPrivacy ? <PrivacyPage /> : <App />}
    </ThemeProvider>
  </StrictMode>,
);

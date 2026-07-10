// Phone companion surface (v1: auth + dashboard + results) — a deliberately
// SEPARATE module tree from the desktop app (../App.tsx), selected once at
// boot by main.tsx via src/device.ts. Desktop layout never has to
// accommodate phone constraints and vice versa; the import boundary is
// enforced by boundary.test.ts (mobile may only reach the shared allowlist:
// api client, deep links, theme, i18n, and the shared AuthFlow).
//
// The certification test + training are desktop-gated on purpose: the
// EEG/spectrogram reading task was calibrated (ℓ*/σ*) at desktop scale, and
// phone-scale viewing would change task difficulty under the certification's
// claims. This surface covers everything else a participant needs on the
// go — most importantly the email verify/reset deep links, which are usually
// opened on a phone.

import { useState } from "react";
import * as api from "../api";
import { AuthFlow } from "../components/AuthFlow";
import { consumeAuthDeepLink, consumeCohortDeepLink } from "../deepLink";
import { MobileHome } from "./MobileHome";
import { MobileSettings } from "./MobileSettings";

export function MobileApp() {
  // Same one-shot email deep-link consumption as the desktop App: parse +
  // strip the URL params, honored only for signed-out visitors.
  const [authDeepLink] = useState(() => {
    const link = consumeAuthDeepLink();
    return api.isAuthed() ? null : link;
  });
  // Cohort-invite email deep link (/?cohort=...): survives the sign-in so
  // the home banner can pulse the matching invitation.
  const [cohortDeepLink] = useState(() => consumeCohortDeepLink());
  const [authed, setAuthed] = useState(api.isAuthed());
  const [screen, setScreen] = useState<"home" | "settings">("home");

  if (!authed) {
    return <AuthFlow onAuthed={() => { setScreen("home"); setAuthed(true); }} deepLink={authDeepLink} />;
  }
  if (screen === "settings") {
    return <MobileSettings onBack={() => setScreen("home")} />;
  }
  return (
    <MobileHome
      onSettings={() => setScreen("settings")}
      onSignOut={() => { api.logout(); setAuthed(false); }}
      inviteHighlightId={cohortDeepLink}
    />
  );
}

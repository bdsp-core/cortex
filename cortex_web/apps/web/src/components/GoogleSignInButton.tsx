// "Continue with Google" — a custom button built to Google's branding
// guidelines (NEUTRAL variant: #f2f2f2 fill, no stroke, #1f1f1f text, official
// multicolor G logo; .gsi-material-button CSS lives in index.html). Because a
// custom button can't use the GIS ID-token "rendered button" flow, sign-in goes
// through Google's OAuth token flow (google.accounts.oauth2.initTokenClient):
// the access token is sent to /api/auth/google, which verifies the token's
// audience + fetches the profile. Self-hides if VITE_GOOGLE_CLIENT_ID is unset.

import { useEffect, useRef, useState } from "react";
import { loginWithGoogle } from "../api";
import { COLORS, FONTS } from "../../ui/theme";
import { useI18n } from "../i18n/LanguageProvider";

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined;
const GIS_SRC = "https://accounts.google.com/gsi/client";

declare global {
  interface Window { google?: any } // eslint-disable-line @typescript-eslint/no-explicit-any
}

let gisPromise: Promise<void> | null = null;
function loadGis(): Promise<void> {
  if (gisPromise) return gisPromise;
  gisPromise = new Promise<void>((resolve, reject) => {
    if (window.google?.accounts?.oauth2) return resolve();
    const s = document.createElement("script");
    s.src = GIS_SRC;
    s.async = true;
    s.defer = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Could not load Google sign-in."));
    document.head.appendChild(s);
  });
  return gisPromise;
}

// Official Google "G" logo (Google branding guidelines).
function GoogleG() {
  return (
    <svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"
      width="20" height="20" style={{ display: "block" }} aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z" />
      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z" />
      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z" />
      <path fill="none" d="M0 0h48v48H0z" />
    </svg>
  );
}

export function GoogleSignInButton({ onAuthed, onError }: {
  onAuthed: () => void;
  onError: (msg: string) => void;
}) {
  const { t } = useI18n();
  const tokenClient = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!CLIENT_ID) return;
    let cancelled = false;
    loadGis()
      .then(() => {
        if (cancelled) return;
        tokenClient.current = window.google.accounts.oauth2.initTokenClient({
          client_id: CLIENT_ID,
          scope: "openid email profile",
          callback: async (resp: { access_token?: string; error?: string }) => {
            if (resp.error || !resp.access_token) { onError("Google sign-in was cancelled."); return; }
            try {
              await loginWithGoogle(resp.access_token);
              onAuthed();
            } catch (e) {
              onError(e instanceof Error ? e.message : "Google sign-in failed.");
            }
          },
        });
        setReady(true);
      })
      .catch((e) => onError(e instanceof Error ? e.message : "Could not load Google sign-in."));
    return () => { cancelled = true; };
  }, [onAuthed, onError]);

  if (!CLIENT_ID) return null;

  const handleClick = () => {
    if (!tokenClient.current) { onError("Google sign-in is still loading. Please try again."); return; }
    tokenClient.current.requestAccessToken();
  };

  return (
    <>
      {/* neutral "or" divider between Create-account and the Google button */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "16px 0 20px" }}>
        <span style={{ flex: 1, height: 1, background: COLORS.borderInactive }} />
        <span style={{ fontSize: 12, color: COLORS.textFaint, fontFamily: FONTS.sans,
          textTransform: "uppercase", letterSpacing: "0.06em" }}>{t("common.or")}</span>
        <span style={{ flex: 1, height: 1, background: COLORS.borderInactive }} />
      </div>
      <button type="button" className="gsi-material-button" onClick={handleClick}
        disabled={!ready} style={{ width: "100%" }}>
        <div className="gsi-material-button-state" />
        <div className="gsi-material-button-content-wrapper">
          <div className="gsi-material-button-icon"><GoogleG /></div>
          <span className="gsi-material-button-contents">Continue with Google</span>
        </div>
      </button>
    </>
  );
}

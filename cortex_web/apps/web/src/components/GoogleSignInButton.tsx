// "Continue with Google" — renders Google's official Sign-in-with-Google button
// (Google Identity Services). Self-hides if VITE_GOOGLE_CLIENT_ID is unset, so
// the app is safe to ship before the OAuth client is configured. On success it
// posts the Google ID token to /api/auth/google (via loginWithGoogle) and calls
// onAuthed(); errors are surfaced through onError.

import { useEffect, useRef, useState } from "react";
import { loginWithGoogle } from "../api";

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined;
const GIS_SRC = "https://accounts.google.com/gsi/client";

declare global {
  // GIS injects window.google; typed loosely (no official types bundled).
  interface Window { google?: any } // eslint-disable-line @typescript-eslint/no-explicit-any
}

// Load the GIS script once, shared across mounts.
let gisPromise: Promise<void> | null = null;
function loadGis(): Promise<void> {
  if (gisPromise) return gisPromise;
  gisPromise = new Promise<void>((resolve, reject) => {
    if (window.google?.accounts?.id) return resolve();
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

export function GoogleSignInButton({ onAuthed, onError }: {
  onAuthed: () => void;
  onError: (msg: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    if (!CLIENT_ID || !ref.current) return;
    let cancelled = false;
    loadGis()
      .then(() => {
        if (cancelled || !ref.current) return;
        const id = window.google.accounts.id;
        id.initialize({
          client_id: CLIENT_ID,
          callback: async (resp: { credential?: string }) => {
            if (!resp.credential) { onError("Google sign-in was cancelled."); return; }
            try {
              await loginWithGoogle(resp.credential);
              onAuthed();
            } catch (e) {
              onError(e instanceof Error ? e.message : "Google sign-in failed.");
            }
          },
        });
        id.renderButton(ref.current, {
          type: "standard", theme: "outline", size: "large",
          text: "continue_with", shape: "rectangular", width: 376,
        });
        setShown(true);
      })
      .catch((e) => onError(e instanceof Error ? e.message : "Could not load Google sign-in."));
    return () => { cancelled = true; };
  }, [onAuthed, onError]);

  if (!CLIENT_ID) return null;
  return (
    <div style={{ marginTop: 12, display: "flex", justifyContent: "center",
      minHeight: shown ? undefined : 0 }}>
      <div ref={ref} />
    </div>
  );
}

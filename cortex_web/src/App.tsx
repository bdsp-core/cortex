// Dev-slice app shell. Boots straight into the Viewer on the local bundle,
// owns the engine Web Worker client + session state, and shows a Results
// panel when the session completes.
//
// The full screen flow (Landing / Consent / Registration / Tutorial /
// Computing) is PLAN §10 phase 4 — this slice exercises the engine end-to-end
// in the browser, which is the thing to verify first.

import { useEffect, useRef, useState } from "react";
import { Bundle } from "./bundle";
import { EngineClient } from "./engineClient";
import { Viewer, Item } from "./components/Viewer";
import { estimateFinishProbability, Progress } from "./progress";
import { MAX_QUESTIONS } from "../engine/session";
import { TrialDiag } from "../engine/types";
import { COLORS, FONTS, IIIC_OPTIONS, VERDICT_STYLE } from "../ui/theme";

const BUNDLE_URL = "/bundle/v1.1-local";

type Phase = "loading" | "running" | "done" | "error";

export function App() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [item, setItem] = useState<Item | null>(null);
  const [progress, setProgress] = useState<Progress>({ answered: 0, maxQ: 0, finishProb: null });
  const [verdicts, setVerdicts] = useState<string[]>([]);
  const [nQ, setNQ] = useState(0);
  const [msg, setMsg] = useState("");
  const clientRef = useRef<EngineClient | null>(null);

  useEffect(() => {
    let disposed = false;
    Bundle.load(BUNDLE_URL)
      .then((b) => {
        if (disposed) return;
        setBundle(b);
        const maxQ = Math.min(MAX_QUESTIONS, b.inputs.segments.length);
        setProgress({ answered: 0, maxQ, finishProb: null });
        const client = new EngineClient({
          onItem: (it) => setItem(it),
          onTrial: (diag: TrialDiag) => {
            const answered = diag.nPerTask.reduce((a, n) => a + n, 0);
            setProgress({ answered, maxQ, finishProb: estimateFinishProbability(diag, maxQ) });
          },
          onDone: (r) => {
            setVerdicts(r.verdicts);
            setNQ(r.nQuestions);
            setPhase("done");
          },
          onError: (m) => { setMsg(m); setPhase("error"); },
        });
        clientRef.current = client;
        client.start(b.inputs, `web-${Date.now()}`);
        setPhase("running");
      })
      .catch((e) => { setMsg(String(e)); setPhase("error"); });
    return () => {
      disposed = true;
      clientRef.current?.dispose();
    };
  }, []);

  if (phase === "loading")
    return <Centered>Loading test bank…</Centered>;
  if (phase === "error")
    return <Centered>Error: {msg}</Centered>;

  if (phase === "done")
    return (
      <Centered>
        <div style={{ width: 700 }}>
          <h1 style={{ fontFamily: FONTS.sans, color: COLORS.textPrimary, letterSpacing: 1 }}>
            Assessment Complete
          </h1>
          <div style={{ color: COLORS.textBody, marginBottom: 16 }}>
            {nQ} recordings reviewed.
          </div>
          {IIIC_OPTIONS.map((o, k) => {
            const v = verdicts[k] ?? "PENDING";
            const st = VERDICT_STYLE[v] ?? VERDICT_STYLE.PENDING;
            return (
              <div key={o.code} style={{ display: "flex", justifyContent: "space-between",
                                          padding: "8px 0", borderBottom: `1px solid ${COLORS.borderInactive}` }}>
                <span style={{ color: COLORS.textSecondary, fontWeight: 600 }}>{o.label}</span>
                <span style={{ color: st.color, fontWeight: 700 }}>{st.label}</span>
              </div>
            );
          })}
        </div>
      </Centered>
    );

  // running
  return (
    <Viewer
      bundle={bundle!}
      item={item}
      progress={progress}
      onAnswer={(pick) => clientRef.current?.answer(pick)}
    />
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
                  height: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      {children}
    </div>
  );
}

// App shell — the full participant flow state machine:
//
//   landing → login → consent → registration → tutorial
//           → loading → running (Viewer) → computing → done (Results)
//
// The engine runs in a Web Worker (off the UI thread). The backend is touched
// exactly twice per sitting — create-session at start, post-results at end —
// plus fire-and-forget per-trial checkpoints for crash-safety (PLAN §8).

import { useCallback, useEffect, useRef, useState } from "react";
import { Bundle } from "./bundle";
import { EngineClient } from "./engineClient";
import { Viewer, Item } from "./components/Viewer";
import { resolutionConfidence, Progress } from "./progress";
import { sampleSession } from "./sampleSession";
import { MAX_QUESTIONS } from "../engine/session";
import { TrialDiag } from "../engine/types";
import * as api from "./api";
import { Landing } from "./components/Landing";
import { Login } from "./components/Login";
import { Consent } from "./components/Consent";
import { Registration, Participant } from "./components/Registration";
import { Tutorial } from "./components/Tutorial";
import { Computing } from "./components/Computing";
import { Results, ResultSummary } from "./components/Results";
import { Stage, Card, Heading, Button } from "./components/ui";
import { COLORS } from "../ui/theme";

type Phase =
  | "landing"
  | "login"
  | "consent"
  | "registration"
  | "tutorial"
  | "loading"
  | "running"
  | "computing"
  | "done"
  | "error";

export function App() {
  const [phase, setPhase] = useState<Phase>("landing");
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [item, setItem] = useState<Item | null>(null);
  const [progress, setProgress] = useState<Progress>({ answered: 0, maxQ: 0, resolveConf: null });
  const [summary, setSummary] = useState<ResultSummary | null>(null);
  const [msg, setMsg] = useState("");

  const clientRef = useRef<EngineClient | null>(null);
  const participantRef = useRef<Participant | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const shownAtRef = useRef<number>(0);
  const lastPickRef = useRef<number | null>(null);
  const lastRtRef = useRef<number | null>(null);
  const lastDiagRef = useRef<TrialDiag | null>(null);

  // Retry any results that failed to upload in a previous sitting, as soon as
  // we have a token (crash-safety reconnect, PLAN §8).
  useEffect(() => {
    if (api.isAuthed()) void api.flushPendingResults();
  }, [phase]);

  // ── flow transitions ──────────────────────────────────────────
  const begin = () => setPhase(api.isAuthed() ? "consent" : "login");

  // Launch the assessment: manifest → bundle → fresh sample → session → engine.
  const startTest = useCallback(async () => {
    setPhase("loading");
    try {
      const manifest = await api.getManifest();
      const b = await Bundle.load(manifest.bundleUrl);
      setBundle(b);
      const { inputs, info } = sampleSession(b.inputs, manifest.sessionSample);
      console.info(
        `[cortex] session sample: ${info.nSampled} of ${info.nPool} pool ` +
        `(seed ${info.seed}); per-class`, info.perClass,
      );
      const sessionId = await api.createSession(
        { ...(participantRef.current ?? {}) }, info.seed,
      );
      sessionIdRef.current = sessionId;

      const maxQ = Math.min(MAX_QUESTIONS, inputs.segments.length);
      setProgress({ answered: 0, maxQ, resolveConf: null });

      const client = new EngineClient({
        onItem: (it) => {
          shownAtRef.current = performance.now();
          setItem(it);
        },
        onTrial: (diag: TrialDiag) => {
          lastDiagRef.current = diag;
          const answered = diag.nPerTask.reduce((a, n) => a + n, 0);
          setProgress({ answered, maxQ, resolveConf: resolutionConfidence(diag) });
          // crash-safe per-trial checkpoint (fire-and-forget)
          api.postProgress(sessionId, {
            trialIndex: diag.trialIndex,
            segId: diag.segId,
            taskK: diag.taskK,
            pick: lastPickRef.current ?? undefined,
            isCorrect: diag.y === 1,
            reactionMs: lastRtRef.current ?? undefined,
            diag,
          });
        },
        onDone: async (r) => {
          setPhase("computing");
          const d = lastDiagRef.current;
          const sum: ResultSummary = {
            nQuestions: r.nQuestions,
            stopReason: r.stopReason,
            verdicts: r.verdicts,
            pi: d?.pi,
            R: d?.R,
            nPerTask: d?.nPerTask,
            // Carry the bundle's task list so the Results table reads the
            // right labels at any K and indexes verdicts by real task idx.
            taskCodes: inputs.taskCodes,
            taskLabels: inputs.taskLabels,
            taskClasses: inputs.taskClasses,
          };
          setSummary(sum);
          // Persist-then-deliver: the payload is saved locally before the
          // POST, so a failed upload is retried on the next authed load
          // rather than lost (PLAN §8).
          const delivered = await api.submitResults(sessionId, {
            verdicts: r.verdicts,
            servedSegIds: r.servedSegIds,
            trials: r.trials,
            participant: participantRef.current,
            sampleSeed: info.seed,
          }, r.stopReason, r.nQuestions);
          if (!delivered) console.warn("[cortex] results upload deferred; retained locally");
          setPhase("done");
        },
        onError: (m) => { setMsg(m); setPhase("error"); },
      });
      clientRef.current = client;
      client.start(inputs, `web-${info.seed}`);
      setPhase("running");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }, []);

  const onAnswer = useCallback((pick: number) => {
    lastPickRef.current = pick;
    lastRtRef.current = Math.round(performance.now() - shownAtRef.current);
    clientRef.current?.answer(pick);
  }, []);

  // ── render ────────────────────────────────────────────────────
  switch (phase) {
    case "landing":
      return <Landing onBegin={begin} />;
    case "login":
      return <Login onAuthed={() => setPhase("consent")} onBack={() => setPhase("landing")} />;
    case "consent":
      return (
        <Consent
          onAccept={() => setPhase("registration")}
          onDecline={() => setPhase("landing")}
        />
      );
    case "registration":
      return (
        <Registration
          onBack={() => setPhase("consent")}
          onComplete={(p) => { participantRef.current = p; setPhase("tutorial"); }}
        />
      );
    case "tutorial":
      return <Tutorial onStart={startTest} onBack={() => setPhase("registration")} />;
    case "loading":
      return <Computing note="Loading the test bank…" />;
    case "computing":
      return <Computing note="Computing your results…" />;
    case "running":
      return (
        <Viewer bundle={bundle!} item={item} progress={progress} onAnswer={onAnswer} />
      );
    case "done":
      return summary ? <Results summary={summary} /> : <Computing />;
    case "error":
      return (
        <Stage maxW={520}>
          <Card>
            <Heading>Something went wrong</Heading>
            <div style={{ color: COLORS.fail, fontSize: 14, marginBottom: 20 }}>{msg}</div>
            <Button onClick={() => setPhase("landing")}>Return to start</Button>
          </Card>
        </Stage>
      );
  }
}

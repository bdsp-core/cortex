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
import { SpikeViewer } from "./components/SpikeViewer";
import { resolutionConfidence, Progress } from "./progress";
import { empiricalPoint, onCurvePoint } from "./roc";
import { sampleSession } from "./sampleSession";
import { MAX_QUESTIONS } from "../engine/session";
import { TrialDiag } from "../engine/types";
import * as api from "./api";
import { Landing } from "./components/Landing";
import { AuthFlow } from "./components/AuthFlow";
import { Consent } from "./components/Consent";
import { Registration, Participant } from "./components/Registration";
import { Computing } from "./components/Computing";
import { Results, ResultSummary } from "./components/Results";
import { Stage, Card, Heading, Button } from "./components/ui";
import { COLORS } from "../ui/theme";

type Phase =
  | "landing"
  | "auth"
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
  const [tutorialItem, setTutorialItem] = useState<Item | null>(null);
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
  const vizRef = useRef<api.VizPayload | null>(null); // trajectory for the video zip (#8)

  // Retry any results that failed to upload in a previous sitting, as soon as
  // we have a token (crash-safety reconnect, PLAN §8).
  useEffect(() => {
    if (api.isAuthed()) void api.flushPendingResults();
  }, [phase]);

  const bundleRef = useRef<Bundle | null>(null);

  // ── flow transitions ──────────────────────────────────────────
  const begin = () => setPhase(api.isAuthed() ? "consent" : "auth");

  // Load the bundle and show the in-context tutorial over an IIIC example seg
  // (the tutorial walks the IIIC UI — spectrogram, red box).
  const enterTutorial = useCallback(async () => {
    setPhase("loading");
    try {
      const manifest = await api.getManifest();
      const b = await Bundle.load(manifest.bundleUrl);
      bundleRef.current = b;
      setBundle(b);
      const example = b.inputs.segments.find((s) => (s as { testClass?: string }).testClass !== "spike")
        ?? b.inputs.segments[0];
      setTutorialItem({ trialIndex: 0, taskK: 1, segId: example.segId });
      setPhase("tutorial");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }, []);

  // Launch the assessment: manifest → bundle → fresh sample → session → engine.
  const startTest = useCallback(async () => {
    setPhase("loading");
    try {
      const manifest = await api.getManifest();
      const b = bundleRef.current ?? await Bundle.load(manifest.bundleUrl);
      bundleRef.current = b;
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
          // Per-task ROC: posterior-mean AUROC + the examinee's empirical
          // operating point projected onto the binormal curve. Spike truth =
          // sign(s_mean); IIIC truth = segment pattern class.
          const truth = new Map(inputs.segments.map((s) => [s.segId, s.patternClass]));
          const words = inputs.taskPatternWords;
          const roc = (r.finalAuroc ?? []).map((auroc, k) => {
            const spike = inputs.taskClasses?.[k] === "spike";
            const emp = empiricalPoint(r.trials, k, words[k], (id) => truth.get(id), spike);
            const op = emp ? onCurvePoint(auroc, emp[0]) : null;
            return { auroc, hw: r.finalAurocHw?.[k] ?? 0, opFar: op?.[0] ?? null, opHr: op?.[1] ?? null };
          });
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
            roc,
          };
          setSummary(sum);
          // Build the visualization-video payload (trajectory + the trial fields
          // the desktop renderers read) for the optional download (#8).
          vizRef.current = {
            shape: r.traj.shape,
            taskCodes: inputs.taskCodes,
            segIds: r.servedSegIds,
            trials: r.trials.map((dd) => ({
              trial_index: dd.trialIndex, task_k: dd.taskK, task_code: inputs.taskCodes[dd.taskK],
              response_y: dd.y, s_mean: dd.s, is_correct: null,
              auroc_hw: dd.aurocHw, policy_diag: { pi: dd.pi, mcse: dd.mcse }, verdicts: dd.verdicts,
            })),
            certificate: { stop_reason: r.stopReason, per_task: r.verdicts.map((v) => ({ verdict: v })) },
            participantName: (participantRef.current as { name?: string } | null)?.name ?? "Anonymous",
            t: r.traj.t, l: r.traj.l, w: r.traj.w,
          };
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

  // Render + download the visualization MP4 zip (#8). Throws on failure so the
  // Results button can surface it.
  const downloadVideos = useCallback(async () => {
    if (!vizRef.current) throw new Error("no session data");
    const blob = await api.requestVideos(vizRef.current);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cortex_visualizations.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, []);

  // ── render ────────────────────────────────────────────────────
  switch (phase) {
    case "landing":
      return <Landing onBegin={begin} />;
    case "auth":
      return <AuthFlow onAuthed={() => setPhase("consent")} />;
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
          onComplete={(p) => { participantRef.current = p; void enterTutorial(); }}
        />
      );
    case "tutorial":
      // In-context tutorial: the real IIIC Viewer on an example segment, with the
      // coach-marks overlay walking the UI; the overlay's "Begin" → startTest.
      return bundle && tutorialItem ? (
        <Viewer bundle={bundle} item={tutorialItem} onAnswer={() => {}}
          tutorial={{ onFinish: startTest }} />
      ) : (
        <Computing note="Loading the tutorial…" />
      );
    case "loading":
      return <Computing note="Loading the test bank…" />;
    case "computing":
      return <Computing note="Computing your results…" />;
    case "running": {
      // Route by the current item's task class — spike → SpikeViewer (binary
      // Yes/No, no spectrogram, 10-s clip), IIIC → the 6-button viewer.
      const tc = bundle?.inputs.taskClasses;
      const k = item?.taskK;
      const isSpike = tc != null && k != null && tc[k] === "spike";
      if (isSpike) {
        return (
          <SpikeViewer bundle={bundle!} item={item} progress={progress}
            onAnswer={onAnswer} spikeTaskIdx={k!}
            totalTasks={bundle!.inputs.taskCodes.length} />
        );
      }
      return (
        <Viewer bundle={bundle!} item={item} onAnswer={onAnswer} />
      );
    }
    case "done":
      return summary ? <Results summary={summary} onDownloadVideos={downloadVideos} /> : <Computing />;
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

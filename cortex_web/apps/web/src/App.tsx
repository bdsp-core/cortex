// App shell — the full participant flow state machine:
//
//   login → consent → registration → tutorial
//         → loading → running (Viewer) → computing → done (Results)
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
import { TrialDiag } from "../engine/types";
import * as api from "./api";
import { AuthFlow } from "./components/AuthFlow";
import { Consent, CONSENT_VERSION, IRB_PROTOCOL_ID } from "./components/Consent";
import { Participant, participantFromProfile } from "./profileFields";
import { Computing } from "./components/Computing";
import { Results, ResultSummary } from "./components/Results";
import { Shell } from "./components/Shell";
import { Stage, Card, Heading, Button } from "./components/ui";
import { COLORS } from "../ui/theme";
import { TrainingRunner } from "./components/TrainingRunner";
import { buildTrainerBank, cutScores, seedClouds, seedCloudsFromPrior } from "./trainerBank";
import { ArrayBank, TrainerSession, buildFilters } from "../trainer/session";
import type { FilterParams } from "../trainer/filter";

type Phase =
  | "auth"
  | "dashboard"
  | "consent"
  | "tutorial"
  | "loading"
  | "running"
  | "computing"
  | "training"
  | "done"
  | "error";

// The trainer's assumed learner dynamics (anchored EXTSET rates; matches the
// Python production defaults). sigmaInf is per-task, set in startTraining.
const TRAINER_PARAMS: FilterParams = {
  alphaT: 0.097, alphaSigma: 0.047, sigmaInf: 0.4,
  qT: 0.05, qSigma: 0.02, rho: 0.5, rule: "soft",
};

export function App() {
  const [phase, setPhase] = useState<Phase>(api.isAuthed() ? "dashboard" : "auth");
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [item, setItem] = useState<Item | null>(null);
  const [tutorialItem, setTutorialItem] = useState<Item | null>(null);
  const [progress, setProgress] = useState<Progress>({ answered: 0, maxQ: 0, resolveConf: null });
  const [summary, setSummary] = useState<ResultSummary | null>(null);
  const [msg, setMsg] = useState("");
  const [trainState, setTrainState] = useState<
    { session: TrainerSession; trainingId: string; labels: string[]; bundle: Bundle } | null
  >(null);

  const clientRef = useRef<EngineClient | null>(null);
  // Terminate + forget the engine worker. Idempotent; called before a new
  // sitting, on done/error, and on sign-out so workers never accumulate or
  // outlive the session (each startTest spins up a fresh Worker).
  const disposeClient = useCallback(() => {
    clientRef.current?.dispose();
    clientRef.current = null;
  }, []);
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

  // Auto sign-out after 30 minutes of inactivity. Any interaction resets the
  // timer; a visibility change re-checks immediately (covers a tab left in the
  // background past the limit). Only runs while signed in.
  useEffect(() => {
    if (phase === "auth") return;
    const IDLE_MS = 30 * 60 * 1000;
    let last = Date.now();
    const bump = () => { last = Date.now(); };
    const events = ["mousemove", "mousedown", "keydown", "scroll", "touchstart", "click"];
    events.forEach((e) => window.addEventListener(e, bump, { passive: true }));
    const check = () => {
      if (Date.now() - last >= IDLE_MS) { disposeClient(); api.logout(); setPhase("auth"); }
    };
    const id = window.setInterval(check, 30_000);
    const onVisible = () => { if (document.visibilityState === "visible") check(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      events.forEach((e) => window.removeEventListener(e, bump));
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [phase, disposeClient]);

  const bundleRef = useRef<Bundle | null>(null);

  // ── flow transitions ──────────────────────────────────────────
  // Load the bundle and show the in-context tutorial over an IIIC example seg
  // (the tutorial walks the IIIC UI — spectrogram, red box).
  const enterTutorial = useCallback(async () => {
    setPhase("loading");
    try {
      const ex = await api.tutorialExample();
      const b = Bundle.fromSessionBank(ex);
      bundleRef.current = b;
      setBundle(b);
      // Load the account profile (collected at signup) into the session record,
      // so demographics are captured without a pre-tutorial wizard. Best-effort.
      try {
        const acct = await api.getProfile();
        participantRef.current = participantFromProfile(
          acct.displayName, { ...acct.profile, expertise: acct.expertise });
      } catch { /* keep going; session records whatever is available */ }
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
      // Server-side draw (goal 3): the backend picks this sitting's balanced,
      // spacing-aware question subset from the full 35k bank and returns it —
      // the browser never downloads the full manifest.
      const { sessionId, sampleSeed, bank } = await api.startSession(
        { ...(participantRef.current ?? {}) },
      );
      const b = Bundle.fromSessionBank(bank);
      bundleRef.current = b;
      setBundle(b);
      const inputs = b.inputs;
      console.info(
        `[cortex] session bank: ${inputs.segments.length} of ${bank.nPool} pool ` +
        `(seed ${sampleSeed})`,
      );
      sessionIdRef.current = sessionId;

      // Adaptive test: the engine ends on per-task resolution (AD6 + per-domain
      // cap), bounded by the drawn pool — use the pool as the progress max.
      const maxQ = inputs.segments.length;
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
         try {
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
          // Per-task posterior SD of ℓ (σ) from the FINAL particle-cloud step —
          // the ± band on the ℓ evolution chart. l is [T,N,K] row-major, w is
          // [T,N]; weighted std over the N particles at t = T-1.
          const [Tn, Np, Kp] = r.traj.shape;
          const lCloud = r.traj.l, wCloud = r.traj.w, ti = Tn - 1;
          const sdPerTask = (inputs.taskCodes ?? []).map((_c, k) => {
            let wsum = 0, mean = 0;
            for (let i = 0; i < Np; i++) {
              const w = wCloud[ti * Np + i];
              wsum += w; mean += w * lCloud[(ti * Np + i) * Kp + k];
            }
            if (wsum <= 0) return null;
            mean /= wsum;
            let varAcc = 0;
            for (let i = 0; i < Np; i++) {
              const w = wCloud[ti * Np + i];
              const dl = lCloud[(ti * Np + i) * Kp + k] - mean;
              varAcc += w * dl * dl;
            }
            return Math.sqrt(varAcc / wsum);
          });
          // Real per-task certification values, persisted so the dashboard reads
          // genuine numbers (not sample data) for ℓ/θ/σ/AUROC. ℓ/θ are the final
          // posterior means from the last engine diagnostic (lMean/tMean); σ is
          // the cloud SD above; ℓ* is the bundle's Youden cut-score; AUROC is the
          // per-task posterior mean. The backend turns these into the first
          // "eval" trajectory point per domain.
          const perTask = (inputs.taskCodes ?? []).map((code, k) => ({
            taskK: k,
            code,
            label: inputs.taskLabels?.[k] ?? code,
            ell: d?.lMean?.[k] ?? null,
            theta: d?.tMean?.[k] ?? null,
            sd: sdPerTask[k] ?? null,
            ellStar: inputs.ellStar?.[k] ?? null,
            auroc: roc[k]?.auroc ?? null,
            aurocHw: roc[k]?.hw ?? null,
            verdict: r.verdicts?.[k] ?? "PENDING",
          }));
          // Persist-then-deliver: the payload is saved locally before the
          // POST, so a failed upload is retried on the next authed load
          // rather than lost (PLAN §8).
          const delivered = await api.submitResults(sessionId, {
            verdicts: r.verdicts,
            perTask,
            roc,
            servedSegIds: r.servedSegIds,
            trials: r.trials,
            participant: participantRef.current,
            sampleSeed,
          }, r.stopReason, r.nQuestions);
          if (!delivered) console.warn("[cortex] results upload deferred; retained locally");
          disposeClient();     // sitting complete → terminate the worker
          setPhase("done");
         } catch (err) {
          // A throw while building/delivering the result must not strand the
          // user on the Computing screen (onDone is an un-awaited worker
          // callback, so its rejection would otherwise be unhandled).
          disposeClient();
          setMsg(err instanceof Error ? err.message : String(err));
          setPhase("error");
         }
        },
        onError: (m) => { disposeClient(); setMsg(m); setPhase("error"); },
      });
      disposeClient();         // never leak a prior sitting's worker
      clientRef.current = client;
      client.start(inputs, `web-${sampleSeed}`);
      setPhase("running");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }, [disposeClient]);

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
  const startTraining = useCallback(async () => {
    setPhase("loading");
    try {
      // createRegimen and startSession are independent server calls; run them
      // in parallel (was a 3-call sequential waterfall). startTrainingSession is
      // sequenced after the regimen since it links the sitting to it.
      const [{ regimen: plan }, { bank }] = await Promise.all([
        api.createRegimen(),                                  // weak-set + measured prior
        // Candidate pool (spacing-aware + media) — reuse the balanced draw; the
        // trainer picks adaptively from it and each seg is renderable.
        api.startSession({ ...(participantRef.current ?? {}) }),
      ]);
      const { trainingId } = await api.startTrainingSession();
      const b = Bundle.fromSessionBank(bank);
      const inputs = b.inputs;
      const ellStar = inputs.ellStar ?? inputs.taskCodes.map(() => 0.3);
      const { ellStars, sigmaStars, sigmaInf } = cutScores(ellStar);
      // Real-skill handoff: seed each task's belief from the learner's measured
      // cert posterior (variance-inflated); fall back to a generic prior only if
      // the regimen carries no posterior (legacy result).
      const clouds = plan?.prior && plan.prior.length
        ? seedCloudsFromPrior(plan.prior, ellStars.length, 200, 1)
        : seedClouds(ellStars.map(() => 0.0), 200, 1);
      const filters = buildFilters(clouds, TRAINER_PARAMS, sigmaInf, ellStars, { seed: 7, useMixture: false });
      const session = new TrainerSession(
        filters, ellStars, sigmaStars,
        new ArrayBank(buildTrainerBank({ segments: inputs.segments })), { seed: 7 });
      setTrainState({ session, trainingId, labels: inputs.taskLabels, bundle: b });
      setPhase("training");
    } catch (e) {
      setMsg(String((e as Error)?.message ?? e));
      setPhase("error");
    }
  }, []);

  switch (phase) {
    case "auth":
      return <AuthFlow onAuthed={() => setPhase("dashboard")} />;
    case "dashboard":
      return (
        <Shell
          onStartTest={() => setPhase("consent")}
          onStartTraining={startTraining}
          onSignOut={() => { disposeClient(); api.logout(); setPhase("auth"); }}
        />
      );
    case "consent":
      return (
        <Consent
          onAccept={() => {
            // Record consent acceptance (best-effort; the per-session blob
            // still carries version/IRB as a durable backup). Phase O1.
            void api.recordConsent(CONSENT_VERSION, IRB_PROTOCOL_ID).catch(() => {});
            // Demographics are now collected at signup (not here); go straight
            // to the in-context tutorial, which loads the profile into the
            // session record.
            void enterTutorial();
          }}
          onDecline={() => setPhase("dashboard")}
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
    case "training":
      return trainState ? (
        <TrainingRunner
          bundle={trainState.bundle}
          session={trainState.session}
          trainingId={trainState.trainingId}
          labels={trainState.labels}
          onExit={() => { setTrainState(null); setPhase("dashboard"); }}
        />
      ) : <Computing note="Preparing your training session…" />;
    case "done":
      return summary
        ? <Results summary={summary} onDownloadVideos={downloadVideos}
            onReturn={() => setPhase("dashboard")} />
        : <Computing />;
    case "error":
      return (
        <Stage maxW={520}>
          <Card>
            <Heading>Something went wrong</Heading>
            <div style={{ color: COLORS.fail, fontSize: 14, marginBottom: 20 }}>{msg}</div>
            <Button onClick={() => setPhase(api.isAuthed() ? "dashboard" : "auth")}>Return to start</Button>
          </Card>
        </Stage>
      );
  }
}

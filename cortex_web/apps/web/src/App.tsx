// App shell — the full participant flow state machine:
//
//   login → consent → registration → tutorial
//         → loading → running (Viewer) → computing → done (Results)
//
// The engine runs in a Web Worker (off the UI thread). The backend is touched
// exactly twice per sitting — create-session at start, post-results at end —
// plus fire-and-forget per-trial checkpoints for crash-safety.

import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { Bundle, SessionBank } from "./bundle";
import { EngineClient } from "./engineClient";
import { Viewer, Item } from "./components/Viewer";
import { SpikeViewer } from "./components/SpikeViewer";
import { resolutionConfidence, Progress } from "./progress";
import { TrialDiag } from "../engine";
import * as api from "./api";
import { bootstrapOnce, invalidateBootstrap } from "./bootstrapStore";
import { AuthFlow } from "./components/AuthFlow";
import {
  consumeAuthDeepLink,
  consumeAuthStartScreen,
  consumeCohortDeepLink,
} from "./deepLink";
import { ReplayDriver, ReplayTrial } from "./resume";
import { reportClientError } from "./telemetry";
import { Consent, CONSENT_VERSION, IRB_PROTOCOL_ID } from "./components/Consent";
import { Participant, participantFromProfile } from "./profileFields";
import { Computing } from "./components/Computing";
import { Results, ResultSummary } from "./components/Results";
import { Shell } from "./components/Shell";
import { Stage, Card, Heading, Button } from "./components/ui";
import { WashoutBanner } from "./components/exam/WashoutBanner";
import {
  buildPerTaskRecords, cloudSdPerTask, perTaskRoc,
} from "./features/exam/results";
import { COLORS } from "../ui/theme";
import type { TrainingState } from "./trainingSetup";
import { EnginePerformanceCollector } from "./performanceSummary";
import { answerIsCorrect } from "./tasks";
import {
  preloadPercentile, scoreOrUnavailable,
} from "./percentile/runtime";
import type { PercentileProfile } from "./percentile/types";

// The training surface is trainer-only weight (belief-filter engine, candidate
// bank, feedback runner) that most visitors — signup, exam — never reach, so
// it loads on demand: the runner via lazy() (same pattern as Cohorts /
// MobileApp), the session construction via the dynamic import in
// startTraining. Only type imports from trainer modules may appear above
// (values would pull the engine back into the main chunk).
const TrainingRunner = lazy(() =>
  import("./components/TrainingRunner").then((m) => ({ default: m.TrainingRunner })));

type Phase =
  | "auth"
  | "dashboard"
  | "consent"
  | "resume"
  | "tutorial"
  | "loading"
  | "running"
  | "saveExit"
  | "computing"
  | "training"
  | "done"
  | "error";

export function App() {
  // One-click email link (verify/reset): parse + strip the URL params once at
  // boot. Honored only for signed-out visitors — the pending-signup user is
  // never authed, and an authed session shouldn't be yanked to auth screens.
  const [authDeepLink] = useState(() => {
    const link = consumeAuthDeepLink();
    return api.isAuthed() ? null : link;
  });
  const [authStartScreen] = useState(() => {
    const screen = consumeAuthStartScreen();
    return api.isAuthed() ? null : screen;
  });
  // Cohort-invite email deep link (/?cohort=...): kept through the auth flow
  // so the dashboard banner can pulse the matching invitation after sign-in.
  const [cohortDeepLink] = useState(() => consumeCohortDeepLink());
  const [phase, setPhase] = useState<Phase>(api.isAuthed() ? "dashboard" : "auth");
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [item, setItem] = useState<Item | null>(null);
  const [tutorialItem, setTutorialItem] = useState<Item | null>(null);
  const [progress, setProgress] = useState<Progress>({ answered: 0, maxQ: 0, resolveConf: null });
  const [summary, setSummary] = useState<ResultSummary | null>(null);
  const [msg, setMsg] = useState("");
  const [trainState, setTrainState] = useState<TrainingState | null>(null);

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
  const shownIsoRef = useRef<string>("");      // wall clock at item render (L2)
  const answeredIsoRef = useRef<string>("");   // wall clock at answer (L2)
  const lastRtRef = useRef<number | null>(null);
  const lastDiagRef = useRef<TrialDiag | null>(null);

  // Retry any results that failed to upload in a previous sitting, as soon as
  // we have a token (crash-safety reconnect).
  useEffect(() => {
    if (api.isAuthed()) void api.flushPendingResults();
  }, [phase]);

  // Deliver queued trial checkpoints when the tab hides or closes (one
  // keepalive batch the browser finishes after the page dies).
  useEffect(() => api.installProgressLifecycleFlush(), []);

  // Session expiry (the 6 h token running out). Without this every surface
  // swallowed its own 401 and rendered an empty state, so an expired session
  // looked like a dashboard that had simply lost its data.
  //
  // A sitting in progress is deliberately NOT interrupted. Answers are
  // checkpointed server-side, and submitResults persists the result locally
  // before attempting delivery, so finishing the sitting loses nothing — it
  // uploads via flushPendingResults after the next sign-in. Ejecting the
  // participant mid-exam is the only way to actually lose their work.
  const phaseRef = useRef(phase);
  phaseRef.current = phase;
  useEffect(() => api.onSessionExpired(() => {
    const inSitting = phaseRef.current === "running"
      || phaseRef.current === "computing"
      || phaseRef.current === "tutorial"
      || phaseRef.current === "training";
    if (inSitting || phaseRef.current === "auth") return;
    disposeClient();
    api.logout();
    // Routed to sign-in without an inline notice: AuthFlow's copy is entirely
    // i18n-driven, so a raw English string here would be the one untranslated
    // line on that screen. A localized "your session expired" notice is the
    // natural follow-up.
    setPhase("auth");
  }), [disposeClient]);

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
  // Mid-test resume: the unfinished sitting offered on "Start test", and the
  // fresh-start fallback runSession uses when a replay diverges.
  const [pendingResume, setPendingResume] = useState<api.ActiveSession | null>(null);
  const startFreshRef = useRef<(() => Promise<void>) | null>(null);
  // Post-training exam washout (server-enforced): when set, the dashboard
  // greys the test button and a top-sheet modal demands acknowledgment.
  const [washoutUntil, setWashoutUntil] = useState<string | null>(null);
  const [washoutAck, setWashoutAck] = useState(false);
  const prevWashoutRef = useRef<string | null>(null);
  // A NEW washout window (different reopen time) needs a fresh acknowledgment.
  useEffect(() => {
    if (washoutUntil !== prevWashoutRef.current) {
      prevWashoutRef.current = washoutUntil;
      setWashoutAck(false);
    }
  }, [washoutUntil]);
  // A resumable exam sitting exists (drives the CTA label: Resume vs Start).
  const [examResumable, setExamResumable] = useState(false);
  // Refresh on every dashboard (re)entry: finishing a training session lands
  // back here, and the test button must grey out immediately. One shared
  // /api/bootstrap round-trip feeds this, the DashboardSurface, and the
  // invite banner; the cleanup invalidates the cache when the phase leaves
  // the dashboard so the next entry re-fetches fresh state.
  useEffect(() => {
    if (phase !== "dashboard") return;
    let gone = false;
    bootstrapOnce()
      .then((b) => {
        if (gone || !b.session) return;
        setWashoutUntil(b.session.washout?.reopensAtUtc ?? null);
        setExamResumable(b.session.examResumable);
      })
      .catch(() => { /* pre-flight + the server 409 still guard the path */ });
    return () => { gone = true; invalidateBootstrap(); };
  }, [phase]);

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

  // Run one engine sitting over a drawn bank — shared by a fresh start and a
  // resume. On resume, `replayTrials` (the server's checkpointed prefix) is
  // fed back through the engine first: the engine seed derives from
  // `web-${sampleSeed}` and the pool replays verbatim (original draw order),
  // so the recorded picks deterministically reconstruct the pre-crash state
  // before the user sees their next live question.
  const runSession = useCallback(async (
    sessionId: string, sampleSeed: number, bank: SessionBank,
    replayTrials: ReplayTrial[], computeMode: api.StartSessionResult["computeMode"],
    percentileProfile: PercentileProfile | null,
  ) => {
      const b = Bundle.fromSessionBank(bank);
      bundleRef.current = b;
      setBundle(b);
      const inputs = b.computeInputs;
      const segmentById = new Map(b.manifest.segments.map((segment) => [segment.segId, segment]));
      sessionIdRef.current = sessionId;
      preloadPercentile(percentileProfile);

      // Precision selects from the full 35k candidate bank but can serve only
      // 60 per domain (<=420). Do not present the candidate-pool size as the
      // participant's question ceiling. Preserve the legacy AD6 display.
      const maxQ = inputs.terminationPolicy === "precision_v1"
        ? inputs.taskCodes.length * (inputs.perDomainCap ?? 60)
        : inputs.segments.length;
      setProgress({ answered: replayTrials.length, maxQ, resolveConf: null });

      const replay = new ReplayDriver(replayTrials);
      const performanceCollector = new EnginePerformanceCollector();
      let checkpointsToSkip = replayTrials.length;   // already server-logged

      const client = new EngineClient({
        onPrefetch: ({ segId }) => {
          if (checkpointsToSkip === 0) b.prefetch([segId]);
        },
        onItem: (it) => {
          const step = replay.next(it.segId);
          if (step.kind === "answer") { client.answer(step.pick); return; }
          if (step.kind === "mismatch") {
            // The replay no longer reproduces the recorded sequence (bank or
            // engine drift since the sitting started) — abandon the resume;
            // a fresh sitting is always safe.
            console.warn("[cortex] resume replay diverged; starting a fresh session");
            disposeClient();
            void startFreshRef.current?.();
            return;
          }
          shownAtRef.current = performance.now();
          shownIsoRef.current = new Date().toISOString();
          setItem(it);
        },
        onTrial: (diag: TrialDiag) => {
          lastDiagRef.current = diag;
          const answered = diag.nPerTask.reduce((a, n) => a + n, 0);
          setProgress({ answered, maxQ, resolveConf: resolutionConfidence(diag) });
          if (checkpointsToSkip > 0) {
            // Replayed trial: its checkpoint (with the real pick/RT) is already
            // on the server — re-posting would overwrite it with stale refs.
            checkpointsToSkip -= 1;
            return;
          }
          // crash-safe per-trial checkpoint (fire-and-forget)
          api.postProgress(sessionId, {
            trialIndex: diag.trialIndex,
            segId: diag.segId,
            taskK: diag.taskK,
            pick: diag.pick,
            isCorrect: answerIsCorrect(
              b.inputs, segmentById.get(diag.segId), diag.taskK, diag.pick,
            ),
            reactionMs: lastRtRef.current ?? undefined,
            shownClientUtc: shownIsoRef.current || undefined,
            answeredClientUtc: answeredIsoRef.current || undefined,
            diag,
          });
        },
        onPerformance: (event) => performanceCollector.record(event),
        onDone: async (r) => {
         try {
          setPhase("computing");
          const d = lastDiagRef.current;
          // Per-task ROC: posterior-mean AUROC + the examinee's empirical
          // operating point projected onto the binormal curve. Spike truth =
          // sign(s_mean); IIIC truth = segment pattern class.
          const truth = new Map(b.manifest.segments.map((s) => [s.segId, s.patternClass]));
          const roc = perTaskRoc(r.finalAuroc, r.finalAurocHw, r.trials,
                                 inputs, (id) => truth.get(id));
          const percentile = await scoreOrUnavailable(
            percentileProfile, r.traj, inputs.taskCodes,
          );
          const sum: ResultSummary = {
            nQuestions: r.nQuestions,
            stopReason: r.stopReason,
            verdicts: r.verdicts,
            terminationPolicy: r.terminationPolicy,
            determinations: r.determinations,
            // Carry the bundle's task list so the Results table reads the
            // right labels at any K and indexes verdicts by real task idx.
            taskCodes: inputs.taskCodes,
            taskLabels: inputs.taskLabels,
            taskClasses: inputs.taskClasses,
            biasFlags: r.biasFlags,
            roc,
            percentile,
          };
          setSummary(sum);
          const sdPerTask = cloudSdPerTask(r.traj, (inputs.taskCodes ?? []).length);
          const perTask = buildPerTaskRecords({
            inputs, lastDiag: d, sdPerTask, roc,
            verdicts: r.verdicts,
            determinations: r.determinations,
            skillIntervals: r.skillIntervals,
            biasIntervals: r.biasIntervals,
            biasFlags: r.biasFlags,
            percentiles: percentile?.status === "available"
              ? percentile.domains : undefined,
          });
          // Persist-then-deliver: the payload is saved locally before the
          // POST, so a failed upload is retried on the next authed load
          // rather than lost.
          const delivered = await api.submitResults(sessionId, {
            verdicts: r.verdicts,
            terminationPolicy: r.terminationPolicy,
            nwayProfile: r.nwayProfile,
            domainStatuses: r.domainStatuses,
            determinations: r.determinations,
            terminalReasons: r.terminalReasons,
            skillIntervals: r.skillIntervals,
            biasIntervals: r.biasIntervals,
            biasFlags: r.biasFlags,
            perTask,
            roc,
            servedSegIds: r.servedSegIds,
            trials: r.trials,
            percentile,
            // Stored for operations/performance review and stripped from every
            // participant dashboard/history response by the API.
            _enginePerformance: performanceCollector.summary(replayTrials.length),
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
        onError: (m) => {
          // Worker failures bypass window.onerror; report explicitly so a
          // field crash mid-test reaches the journal, then surface it.
          reportClientError(`engine: ${m}`, undefined, "desktop");
          disposeClient(); setMsg(m); setPhase("error");
        },
      });
      disposeClient();         // never leak a prior sitting's worker
      clientRef.current = client;
      client.start(inputs, `web-${sampleSeed}`, {
        requestedComputeMode: computeMode,
      });
      setPhase("running");
  }, [disposeClient]);

  // Launch a fresh assessment: server draw → session → engine.
  const startTest = useCallback(async () => {
    setPhase("loading");
    try {
      // Server-authoritative candidate profile: AD6 receives its balanced
      // sample; the allowlisted Precision pilot receives the complete
      // exposure-eligible served bank required by its frozen profile.
      const {
        sessionId, sampleSeed, bank, computeMode, percentileProfile,
      } = await api.startSession(
        { ...(participantRef.current ?? {}) },
      );
      console.info(
        `[cortex] session bank: ${bank.segments.length} of ${bank.nPool} pool ` +
        `(seed ${sampleSeed})`,
      );
      await runSession(
        sessionId, sampleSeed, bank, [], computeMode, percentileProfile,
      );
    } catch (e) {
      if (e instanceof api.ApiError && e.message === "training_washout") {
        const body = e.body as { reopensAtUtc?: string } | undefined;
        setWashoutUntil(body?.reopensAtUtc ?? new Date().toISOString());
        setWashoutAck(false);
        setPhase("dashboard");
        return;
      }
      setMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }, [runSession]);

  // The replay-mismatch fallback inside runSession needs startTest before it
  // is defined — bridge with a ref.
  useEffect(() => { startFreshRef.current = startTest; }, [startTest]);

  // Resume the checkpointed sitting the server offered: same engine path,
  // with the logged trials replayed first.
  const resumeTest = useCallback(async (active: api.ActiveSession) => {
    setPendingResume(null);
    setPhase("loading");
    try {
      console.info(
        `[cortex] resuming session ${active.sessionId} at trial ${active.trials.length}`);
      await runSession(active.sessionId, active.bank.sampleSeed ?? 0,
        active.bank, active.trials, active.computeMode,
        active.percentileProfile);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }, [runSession]);

  // "Start test" from the dashboard: offer resume when an unfinished recent
  // sitting exists (its consent + tutorial were already done); otherwise the
  // normal consent → tutorial → test flow. Resume detection is best-effort —
  // any failure falls through to the normal flow.
  const onStartTestClick = useCallback(async () => {
    // No phase change during the pre-flight: switching to "loading" here
    // flashed the test-pipeline spinner for a round-trip that usually ends
    // back on the dashboard (washout) or on the consent screen (no data
    // needed). The dashboard stays up until we know where we're going.
    let active: api.ActiveSession | null = null;
    try {
      const r = await api.activeSession();
      if (r.washout) {
        // Blocked before consent/tutorial: the server would 409 anyway.
        setWashoutUntil(r.washout.reopensAtUtc);
        setWashoutAck(false);
        return;
      }
      active = r.active;
    } catch { /* best-effort */ }
    if (active && active.trials.length > 0) {
      setPendingResume(active);
      setPhase("resume");
    } else {
      setPhase("consent");
    }
  }, []);

  const onAnswer = useCallback((pick: number) => {
    lastRtRef.current = Math.round(performance.now() - shownAtRef.current);
    answeredIsoRef.current = new Date().toISOString();
    clientRef.current?.answer(pick);
  }, []);

  const onMediaReady = useCallback((trialIndex: number) => {
    clientRef.current?.mediaReady(trialIndex);
  }, []);


  // ── render ────────────────────────────────────────────────────
  const startTraining = useCallback(async () => {
    setPhase("loading");
    try {
      // createRegimen and the training-bank draw are independent server calls;
      // run them in parallel (was a 3-call sequential waterfall), and pull the
      // lazy trainer-engine chunk down alongside them. startTrainingSession is
      // sequenced after the regimen since it links the sitting to it.
      const [{ regimen: plan }, { bank }, setup] = await Promise.all([
        api.createRegimen(),                                  // weak-set + measured prior
        // Candidate pool for training — an UNGATED draw (no exam washout, no
        // test/train exposure exclusion), so resuming training is never blocked
        // and the weak-domain pools aren't starved. Every seg is renderable.
        api.startTrainingBank(),
        import("./trainingSetup"),
      ]);
      const { trainingId, percentileProfile } = await api.startTrainingSession();
      // The server-side learning engine is the SOLE trainer (2026-07-17:
      // the incumbent client trainer was removed, no fallback). If the
      // engine is disabled server-side, this surfaces as an error state.
      setTrainState(await setup.buildServerTrainingState(
        plan, bank, trainingId, percentileProfile,
      ));
      setPhase("training");
    } catch (e) {
      setMsg(String((e as Error)?.message ?? e));
      setPhase("error");
    }
  }, []);

  switch (phase) {
    case "auth":
      return (
        <AuthFlow
          onAuthed={() => setPhase("dashboard")}
          deepLink={authDeepLink}
          initialScreen={authStartScreen}
        />
      );
    case "dashboard":
      return (
        <>
          <Shell
            onStartTest={() => { void onStartTestClick(); }}
            onStartTraining={startTraining}
            onSignOut={() => { disposeClient(); api.logout(); setPhase("auth"); }}
            inviteHighlightId={cohortDeepLink}
            testDisabledUntil={washoutUntil}
            examResumable={examResumable}
          />
          {washoutUntil && !washoutAck && (
            <WashoutBanner reopensAtUtc={washoutUntil}
              onAccept={() => setWashoutAck(true)} />
          )}
        </>
      );
    case "resume":
      return pendingResume ? (
        <Stage maxW={520}>
          <Card>
            <Heading>Resume your test?</Heading>
            <div style={{ fontSize: 14, marginBottom: 20 }}>
              You have an unfinished test from{" "}
              {new Date(pendingResume.startedUtc).toLocaleString()} with{" "}
              {pendingResume.trials.length} answer
              {pendingResume.trials.length === 1 ? "" : "s"} saved. You can pick
              up where you left off, or start a new test from scratch.
            </div>
            <div style={{ display: "flex", gap: 12 }}>
              <Button onClick={() => { void resumeTest(pendingResume); }}>
                Resume test
              </Button>
              <Button kind="ghost"
                onClick={() => { setPendingResume(null); setPhase("consent"); }}>
                Start over
              </Button>
            </div>
          </Card>
        </Stage>
      ) : <Computing />;
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
            onAnswer={onAnswer} onMediaReady={onMediaReady} spikeTaskIdx={k!}
            totalTasks={bundle!.inputs.taskCodes.length}
            onExit={() => setPhase("saveExit")} />
        );
      }
      return (
        <Viewer bundle={bundle!} item={item} onAnswer={onAnswer}
          onMediaReady={onMediaReady}
          onExit={() => setPhase("saveExit")} />
      );
    }
    case "saveExit":
      // The engine worker stays alive behind this card: "Keep testing"
      // returns to the same question; "Save & exit" leans on the per-trial
      // server checkpoints + the resume flow (answers already saved).
      return (
        <Stage maxW={520}>
          <Card>
            <Heading>Save and finish later?</Heading>
            <div style={{ fontSize: 14, marginBottom: 20 }}>
              Your answers save automatically as you go. You can pick up right
              where you left off from Start test within 24 hours; after that,
              a new test starts fresh.
            </div>
            <div style={{ display: "flex", gap: 12 }}>
              <Button onClick={() => setPhase("running")}>Keep testing</Button>
              <Button kind="ghost"
                onClick={() => { disposeClient(); setPhase("dashboard"); }}>
                Save &amp; exit
              </Button>
            </div>
          </Card>
        </Stage>
      );
    case "training":
      // The Suspense fallback should never paint in practice — startTraining
      // pulls the chunk down in parallel with the server calls — but covers a
      // cold cache racing a fast API.
      return trainState ? (
        <Suspense fallback={<Computing note="Preparing your training session…" />}>
          <TrainingRunner
            bundle={trainState.bundle}
            session={trainState.session}
            trainingId={trainState.trainingId}
            labels={trainState.labels}
            attainability={trainState.attainability}
            percentileProfile={trainState.percentileProfile}
            onExit={() => { setTrainState(null); setPhase("dashboard"); }}
          />
        </Suspense>
      ) : <Computing note="Preparing your training session…" />;
    case "done":
      return summary
        ? <Results summary={summary} onReturn={() => setPhase("dashboard")} />
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

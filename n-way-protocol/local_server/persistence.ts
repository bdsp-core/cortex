// Production-layout data capture for local test sessions.
//
// Persistence path (chosen at implementation time, see README): this node is
// v20 (no node:sqlite, which needs >= 22) and better-sqlite3 is not vendored
// under ../cortex_web/node_modules, so the server persists newline-JSON rows
// in the exact two-table production shape while a session runs
// (data/ndjson/<id>.session.json rewritten per state change,
// data/ndjson/<id>.trials.jsonl appended per answer) and shells out to
// local_server/persist.py (python3 stdlib sqlite3) at session stop to
// materialize/refresh the real SQLite file data/local_test.db against
// local_server/schema.sql.
//
// Timestamps use the production canonical ISO-Z second format
// (cortex_web/services/api/timeutil.ISO_FMT, "%Y-%m-%dT%H:%M:%SZ") so
// lexicographic ordering over the TEXT columns stays chronological, exactly
// as prod relies on. Client-side stamps arrive from the browser as-is.

import { execFileSync } from "node:child_process";
import {
  appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync,
} from "node:fs";
import path from "node:path";

import type { LocalSession, PersistenceHooks, TrialCapture } from "./engine_session";
import type { ConditionalF1ArtifactEnsemble, EngineProfile } from "../src/types";

/** One `sessions` row, keys = exact production column names (schema.sql). */
export interface SessionRow {
  session_id: string;
  code: string | null;
  participant: string;
  sample_seed: number;
  started_utc: string;
  finished_utc: string | null;
  status: string;
  stop_reason: string | null;
  n_questions: number | null;
  bundle_version: string;
  drawn_seg_ids: number[];
  termination_policy: string;
  candidate_exclusion: string | null;
  candidate_bank_sha256: string;
  compute_mode: string;
  nway_profile: EngineProfile;
  norm_id: string | null;
  norm_sha256: string | null;
  score_schema_version: string | null;
  norm_profile: string | null;
  engine_profile_id: string;
  response_model: string;
  response_artifact_id: string;
  response_artifact_sha256: string;
  selector_version: string;
  engine_algorithm_version: string;
}

export interface SessionListEntry {
  session_id: string;
  participant: string;
  status: string;
  started_utc: string;
  finished_utc: string | null;
  stop_reason: string | null;
  /** Finalized count (NULL while the session is still open, as in prod). */
  n_questions: number | null;
  /** Live `trials` row count (equals n_questions once finalized). */
  n_trials: number;
  sample_seed: number;
  persisted_to_sqlite: boolean;
}

export interface PersistenceOptions {
  dataDir: string;
  /** local_server/persist.py (invoked via python3 at session stop). */
  persistScript: string;
  bundleVersion: string;
  profile: EngineProfile;
  artifact: ConditionalF1ArtifactEnsemble;
  candidateBankSha256: string;
}

/** Production timeutil.utc_now(): fixed-width ISO-Z, second resolution. */
export function utcNow(): string {
  return `${new Date().toISOString().slice(0, 19)}Z`;
}

export class LocalPersistence implements PersistenceHooks {
  readonly dbPath: string;
  private readonly ndjsonDir: string;
  private readonly persisted = new Set<string>();

  constructor(private readonly options: PersistenceOptions) {
    this.ndjsonDir = path.join(options.dataDir, "ndjson");
    this.dbPath = path.join(options.dataDir, "local_test.db");
    mkdirSync(this.ndjsonDir, { recursive: true });
  }

  private sessionPath(sessionId: string): string {
    return path.join(this.ndjsonDir, `${sessionId}.session.json`);
  }

  private trialsPath(sessionId: string): string {
    return path.join(this.ndjsonDir, `${sessionId}.trials.jsonl`);
  }

  private writeSessionRow(row: SessionRow): void {
    writeFileSync(this.sessionPath(row.session_id), `${JSON.stringify(row)}\n`);
  }

  private readSessionRow(sessionId: string): SessionRow {
    return JSON.parse(
      readFileSync(this.sessionPath(sessionId), "utf8"),
    ) as SessionRow;
  }

  onCreate(session: LocalSession): void {
    const { options } = this;
    this.writeSessionRow({
      session_id: session.id,
      code: null,
      participant: session.participantLabel,
      sample_seed: session.config.seed,
      started_utc: utcNow(),
      finished_utc: null,
      // Production open-state for a session carrying an nway_profile.
      status: "in_progress_nway",
      stop_reason: null,
      n_questions: null,
      bundle_version: options.bundleVersion,
      drawn_seg_ids: session.drawnSegIds,
      termination_policy: "precision_frozen",
      candidate_exclusion: null,
      candidate_bank_sha256: options.candidateBankSha256,
      compute_mode: "local_isolated_draw_latent_rd",
      nway_profile: options.profile,
      norm_id: null,
      norm_sha256: null,
      score_schema_version: null,
      norm_profile: null,
      engine_profile_id: options.profile.engineProfileId,
      response_model: options.profile.responseModel,
      response_artifact_id: options.artifact.artifactId,
      response_artifact_sha256: options.artifact.sha256,
      selector_version: options.profile.selectorVersion,
      engine_algorithm_version: options.profile.engineAlgorithmVersion,
    });
  }

  onTrial(session: LocalSession, capture: TrialCapture): void {
    // Keys = exact production trials column names; diag stays an object here
    // and is serialized to the TEXT column by persist.py.
    const row = {
      session_id: session.id,
      trial_index: capture.trialIndex,
      seg_id: capture.segId,
      task_k: capture.taskK,
      pick: capture.pick,
      is_correct: capture.isCorrect,
      reaction_ms: capture.reactionMs,
      diag: capture.diag,
      received_utc: utcNow(),
      shown_client_utc: capture.shownClientUtc,
      answered_client_utc: capture.answeredClientUtc,
    };
    appendFileSync(this.trialsPath(session.id), `${JSON.stringify(row)}\n`);
  }

  onStop(session: LocalSession): void {
    const row = this.readSessionRow(session.id);
    row.finished_utc = utcNow();
    row.status = "complete";
    row.stop_reason = session.stoppedReason;
    row.n_questions = session.questionsAsked;
    this.writeSessionRow(row);
    // Materialize the real SQLite file. Synchronous on purpose: when the
    // stopping answer's HTTP response returns, the .db row is on disk.
    execFileSync("python3", [
      this.options.persistScript,
      "--db", this.dbPath,
      "--data-dir", this.options.dataDir,
      "--session", session.id,
    ], { stdio: ["ignore", "inherit", "inherit"] });
    this.persisted.add(session.id);
  }

  private countTrials(sessionId: string): number {
    const file = this.trialsPath(sessionId);
    if (!existsSync(file)) return 0;
    return readFileSync(file, "utf8").split("\n").filter((line) => line.trim().length > 0).length;
  }

  listSessions(): SessionListEntry[] {
    const entries: SessionListEntry[] = [];
    for (const name of readdirSync(this.ndjsonDir)) {
      if (!name.endsWith(".session.json")) continue;
      const row = JSON.parse(
        readFileSync(path.join(this.ndjsonDir, name), "utf8"),
      ) as SessionRow;
      entries.push({
        session_id: row.session_id,
        participant: row.participant,
        status: row.status,
        started_utc: row.started_utc,
        finished_utc: row.finished_utc,
        stop_reason: row.stop_reason,
        n_questions: row.n_questions,
        n_trials: this.countTrials(row.session_id),
        sample_seed: row.sample_seed,
        persisted_to_sqlite: this.persisted.has(row.session_id)
          || (row.status === "complete" && existsSync(this.dbPath)),
      });
    }
    entries.sort((a, b) => a.started_utc.localeCompare(b.started_utc));
    return entries;
  }
}

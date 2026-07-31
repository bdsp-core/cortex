// Fully local HTTP server for sitting draw-latent n-way sessions.
//
// Serves a static console page plus a JSON API:
//   GET  /                     the session console (local_server/index.html)
//   GET  /healthz              boot/readiness probe
//   POST /session              create a session (mode manual|simulated)
//   GET  /session/:id          full session status
//   POST /session/:id/answer   {pick: 1..6} manual answer
//   POST /session/:id/autostep {n, reader?} simulated answers
//
// No external network: binds 127.0.0.1 only, reads only repo-local files,
// and uses node's built-in http module.

import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { fileURLToPath } from "node:url";

import {
  buildLocalProfile, HttpError, loadAtoms17Artifact, loadServedBank,
  LocalSession, type SessionOptions,
} from "./engine_session";

const BANK_PATH = fileURLToPath(
  new URL("../.artifacts/categorical_bank_axes.csv", import.meta.url),
);
const ARTIFACT_PATH = fileURLToPath(
  new URL("../artifacts/iiic_conditional_f1_engine_frame_atoms17_rd.json", import.meta.url),
);
const PAGE_PATH = fileURLToPath(new URL("../local_server/index.html", import.meta.url));

const bank = loadServedBank(BANK_PATH);
const artifact = loadAtoms17Artifact(ARTIFACT_PATH);
const profile = buildLocalProfile(artifact, bank.sha256);

const sessions = new Map<string, LocalSession>();

function json(response: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  response.end(text);
}

async function readBody(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += (chunk as Buffer).length;
    if (size > 1_000_000) throw new HttpError(413, "request body too large");
    chunks.push(chunk as Buffer);
  }
  if (chunks.length === 0) return {};
  try {
    const parsed = JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("body must be a JSON object");
    }
    return parsed as Record<string, unknown>;
  } catch (error) {
    throw new HttpError(400, `invalid JSON body: ${(error as Error).message}`);
  }
}

function getSession(id: string): LocalSession {
  const session = sessions.get(id);
  if (!session) throw new HttpError(404, `unknown session ${id}`);
  return session;
}

function requireIdle(session: LocalSession): void {
  if (session.busy) {
    throw new HttpError(409, "session is busy executing an autostep batch");
  }
}

const yieldToEventLoop = () => new Promise<void>((resolve) => setImmediate(resolve));

async function autostep(
  session: LocalSession,
  body: Record<string, unknown>,
): Promise<{ stepsExecuted: number }> {
  const n = body.n ?? 1;
  if (typeof n !== "number" || !Number.isInteger(n) || n < 1 || n > 1000) {
    throw new HttpError(400, "n must be an integer in [1, 1000]");
  }
  if (!session.hasReader) {
    if (body.reader === undefined) {
      throw new HttpError(
        409, "manual session has no simulated reader; pass a reader to attach one",
      );
    }
    session.attachReader(body.reader as SessionOptions["reader"]);
  }
  session.busy = true;
  let stepsExecuted = 0;
  try {
    for (let step = 0; step < n && !session.isStopped; step += 1) {
      session.answer(session.simulatedPick());
      stepsExecuted += 1;
      // Keep status polls responsive during long batches.
      await yieldToEventLoop();
    }
  } finally {
    session.busy = false;
  }
  return { stepsExecuted };
}

async function route(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const url = new URL(request.url ?? "/", "http://127.0.0.1");
  const method = request.method ?? "GET";

  if (method === "GET" && url.pathname === "/") {
    response.writeHead(200, {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
    });
    response.end(readFileSync(PAGE_PATH));
    return;
  }
  if (method === "GET" && url.pathname === "/favicon.ico") {
    response.writeHead(204);
    response.end();
    return;
  }
  if (method === "GET" && url.pathname === "/healthz") {
    json(response, 200, {
      ok: true,
      bankRows: bank.rows.length,
      bankSha256: bank.sha256,
      artifactId: artifact.artifactId,
      atomCount: artifact.draws.length,
      engineProfileId: profile.engineProfileId,
      responseAggregation: profile.responseAggregation,
      sessions: sessions.size,
    });
    return;
  }
  if (method === "POST" && url.pathname === "/session") {
    const body = await readBody(request);
    const session = new LocalSession(
      bank, artifact, profile, body as unknown as SessionOptions,
    );
    sessions.set(session.id, session);
    json(response, 201, session.status());
    return;
  }
  const sessionMatch = url.pathname.match(/^\/session\/([^/]+)(?:\/(answer|autostep))?$/);
  if (sessionMatch) {
    const session = getSession(sessionMatch[1]);
    const action = sessionMatch[2];
    if (method === "GET" && action === undefined) {
      json(response, 200, session.status());
      return;
    }
    if (method === "POST" && action === "answer") {
      requireIdle(session);
      const body = await readBody(request);
      if (typeof body.pick !== "number") {
        throw new HttpError(400, "body must carry a numeric pick");
      }
      session.answer(body.pick);
      json(response, 200, session.status());
      return;
    }
    if (method === "POST" && action === "autostep") {
      requireIdle(session);
      const body = await readBody(request);
      const { stepsExecuted } = await autostep(session, body);
      json(response, 200, { stepsExecuted, ...session.status() });
      return;
    }
  }
  throw new HttpError(404, `no route for ${method} ${url.pathname}`);
}

const port = Number(process.env.PORT ?? 8734);
const server = createServer((request, response) => {
  route(request, response).catch((error: unknown) => {
    const status = error instanceof HttpError ? error.status : 500;
    const message = error instanceof Error ? error.message : String(error);
    if (!response.headersSent) json(response, status, { ok: false, error: message });
    else response.end();
    if (status === 500) console.error(error);
  });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`draw-latent local session console: http://127.0.0.1:${port}/`);
  console.log(`  bank: ${bank.rows.length} segments (${BANK_PATH})`);
  console.log(`  artifact: ${artifact.artifactId} (${artifact.draws.length} atoms, sha256 ${artifact.sha256.slice(0, 12)}...)`);
  console.log(`  profile: ${profile.engineProfileId} responseAggregation=${profile.responseAggregation}`);
  console.log("  stopping: precision_v1 (per-domain cap 60, nMin 20, full-bank tercile band edges)");
});

// Transport core for the API client: per-request timeouts, bounded retry for
// transient transport failures, and a small in-memory outbox for idempotent
// fire-and-forget posts. Pure — no storage/DOM access and injectable
// fetch/sleep — so it unit-tests in the node vitest environment.

export interface TransportOpts {
  /** Time-to-response-HEADERS guard (a hung connection must not stall the
   *  caller forever). Body streaming is deliberately NOT bounded, so large
   *  downloads on slow links still complete. */
  timeoutMs?: number;
  /** Extra attempts after the first, for TRANSIENT failures only: network
   *  errors, timeouts, and 502/503/504 (the gateway during a deploy or a
   *  service restart). Any other status returns to the caller — 4xx
   *  (including 401/429) never retries. Only use on idempotent requests. */
  retries?: number;
  fetchImpl?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
}

export const RETRY_STATUS = new Set([502, 503, 504]);
const DEFAULT_TIMEOUT_MS = 20_000;

const realSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// 500ms, 1s, 2s… with up-to-2× jitter so concurrent clients don't stampede.
export function backoffMs(attempt: number): number {
  return 500 * 2 ** (attempt - 1) * (1 + Math.random());
}

export async function transportFetch(
  url: string, init: RequestInit = {}, opts: TransportOpts = {},
): Promise<Response> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, retries = 0, sleep = realSleep } = opts;
  const fetchImpl = opts.fetchImpl ?? fetch;
  let lastErr: unknown = new Error("transportFetch: no attempt made");
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) await sleep(backoffMs(attempt));
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetchImpl(url, { ...init, signal: ctl.signal });
      if (RETRY_STATUS.has(res.status) && attempt < retries) {
        lastErr = new Error(`transient HTTP ${res.status}`);
        continue;
      }
      return res;
    } catch (e) {
      lastErr = e;   // network error or timeout abort — retry if any left
    } finally {
      clearTimeout(timer);   // fires at headers; body read stays unbounded
    }
  }
  throw lastErr;
}

// ── outbox ─────────────────────────────────────────────────────────
// FIFO queue + serialized flush for idempotent posts (e.g. the per-trial
// /api/progress upsert). A transient send failure keeps the item queued for
// the next flush; `shouldDrop` lets the owner discard poison pills (permanent
// 4xx rejections) so one bad item can't wedge the queue forever.

export interface OutboxOpts<T> {
  send: (item: T) => Promise<void>;
  /** Return true to DISCARD the failed item instead of retrying it. */
  shouldDrop?: (err: unknown) => boolean;
  /** Oldest items are evicted beyond this size (default 500). */
  cap?: number;
}

export class Outbox<T> {
  private queue: T[] = [];
  private flushing = false;

  constructor(private opts: OutboxOpts<T>) {}

  get size(): number {
    return this.queue.length;
  }

  push(item: T): void {
    this.queue.push(item);
    const cap = this.opts.cap ?? 500;
    while (this.queue.length > cap) this.queue.shift();
  }

  /** Drain the queue in order. Stops (keeping the remainder) on the first
   *  transient failure; never throws. Concurrent calls coalesce. */
  async flush(): Promise<void> {
    if (this.flushing) return;
    this.flushing = true;
    try {
      while (this.queue.length) {
        try {
          await this.opts.send(this.queue[0]);
          this.queue.shift();
        } catch (e) {
          if (this.opts.shouldDrop?.(e)) {
            this.queue.shift();
            continue;
          }
          return;   // transient — retry on a later flush
        }
      }
    } finally {
      this.flushing = false;
    }
  }
}

// Transport-core tests: timeout, bounded retry classification, and outbox
// drain/keep/drop semantics. All network + time is injected — no real I/O.

import { describe, expect, it } from "vitest";
import { Outbox, transportFetch } from "./transport";

const noSleep = () => Promise.resolve();

function res(status: number): Response {
  return new Response("{}", { status });
}

describe("transportFetch", () => {
  it("returns a success response without retrying", async () => {
    let calls = 0;
    const r = await transportFetch("/x", {}, {
      retries: 2, sleep: noSleep,
      fetchImpl: async () => { calls++; return res(200); },
    });
    expect(r.status).toBe(200);
    expect(calls).toBe(1);
  });

  it("retries transient 503 then succeeds", async () => {
    let calls = 0;
    const r = await transportFetch("/x", {}, {
      retries: 2, sleep: noSleep,
      fetchImpl: async () => { calls++; return res(calls < 3 ? 503 : 200); },
    });
    expect(r.status).toBe(200);
    expect(calls).toBe(3);
  });

  it("returns the last 503 once retries are exhausted (caller sees the status)", async () => {
    let calls = 0;
    const r = await transportFetch("/x", {}, {
      retries: 1, sleep: noSleep,
      fetchImpl: async () => { calls++; return res(503); },
    });
    expect(r.status).toBe(503);
    expect(calls).toBe(2);
  });

  it("never retries 4xx (401/429 are the caller's problem)", async () => {
    let calls = 0;
    const r = await transportFetch("/x", {}, {
      retries: 3, sleep: noSleep,
      fetchImpl: async () => { calls++; return res(429); },
    });
    expect(r.status).toBe(429);
    expect(calls).toBe(1);
  });

  it("retries network errors and rethrows the last one", async () => {
    let calls = 0;
    await expect(transportFetch("/x", {}, {
      retries: 2, sleep: noSleep,
      fetchImpl: async () => { calls++; throw new TypeError("network down"); },
    })).rejects.toThrow("network down");
    expect(calls).toBe(3);
  });

  it("aborts a hung request at the timeout", async () => {
    const hung: typeof fetch = (_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
      });
    await expect(transportFetch("/x", {}, {
      timeoutMs: 30, retries: 0, fetchImpl: hung,
    })).rejects.toThrow("aborted");
  });
});

describe("Outbox", () => {
  it("drains in order and empties on success", async () => {
    const sent: number[] = [];
    const box = new Outbox<number>({ send: async (n) => { sent.push(n); } });
    box.push(1); box.push(2); box.push(3);
    await box.flush();
    expect(sent).toEqual([1, 2, 3]);
    expect(box.size).toBe(0);
  });

  it("keeps items across a transient failure and resumes on the next flush", async () => {
    const sent: number[] = [];
    let failFirst = true;
    const box = new Outbox<number>({
      send: async (n) => {
        if (failFirst) { failFirst = false; throw new Error("blip"); }
        sent.push(n);
      },
    });
    box.push(1); box.push(2);
    await box.flush();                    // fails on item 1 → both retained
    expect(box.size).toBe(2);
    await box.flush();                    // heals: drains in order
    expect(sent).toEqual([1, 2]);
    expect(box.size).toBe(0);
  });

  it("drops poison pills when shouldDrop says so, and continues", async () => {
    const sent: number[] = [];
    const box = new Outbox<number>({
      send: async (n) => {
        if (n === 1) throw new Error("permanent rejection");
        sent.push(n);
      },
      shouldDrop: (e) => String(e).includes("permanent"),
    });
    box.push(1); box.push(2);
    await box.flush();
    expect(sent).toEqual([2]);
    expect(box.size).toBe(0);
  });

  it("evicts oldest items beyond the cap", () => {
    const box = new Outbox<number>({ send: async () => {}, cap: 2 });
    box.push(1); box.push(2); box.push(3);
    expect(box.size).toBe(2);
  });
});

describe("Outbox batched drain", () => {
  type Item = { session: string; n: number };
  const opts = (batches: Item[][]) => ({
    send: async () => { throw new Error("single send must not be used"); },
    sendBatch: async (items: Item[]) => { batches.push(items.slice()); },
    maxBatch: 3,
    groupKey: (item: Item) => item.session,
  });

  it("drains as maxBatch-capped chunks in order", async () => {
    const batches: Item[][] = [];
    const box = new Outbox<Item>(opts(batches));
    for (let n = 1; n <= 7; n++) box.push({ session: "a", n });
    await box.flush();
    expect(batches.map((b) => b.map((i) => i.n))).toEqual([[1, 2, 3], [4, 5, 6], [7]]);
    expect(box.size).toBe(0);
  });

  it("never mixes group keys inside one batch", async () => {
    const batches: Item[][] = [];
    const box = new Outbox<Item>(opts(batches));
    box.push({ session: "a", n: 1 });
    box.push({ session: "a", n: 2 });
    box.push({ session: "b", n: 3 });
    await box.flush();
    expect(batches.map((b) => b.map((i) => `${i.session}${i.n}`)))
      .toEqual([["a1", "a2"], ["b3"]]);
  });

  it("keeps the whole chunk on a transient failure and resumes", async () => {
    const batches: Item[][] = [];
    let failFirst = true;
    const box = new Outbox<Item>({
      ...opts(batches),
      sendBatch: async (items: Item[]) => {
        if (failFirst) { failFirst = false; throw new Error("blip"); }
        batches.push(items.slice());
      },
    });
    box.push({ session: "a", n: 1 });
    box.push({ session: "a", n: 2 });
    await box.flush();
    expect(box.size).toBe(2);
    await box.flush();
    expect(batches[0].map((i) => i.n)).toEqual([1, 2]);
    expect(box.size).toBe(0);
  });

  it("drops a poison chunk when shouldDrop says so, and continues", async () => {
    const batches: Item[][] = [];
    let first = true;
    const box = new Outbox<Item>({
      ...opts(batches),
      sendBatch: async (items: Item[]) => {
        if (first) { first = false; throw new Error("permanent rejection"); }
        batches.push(items.slice());
      },
      shouldDrop: (e) => String(e).includes("permanent"),
    });
    for (let n = 1; n <= 4; n++) box.push({ session: "a", n });
    await box.flush();
    // First chunk (1,2,3) dropped as poison; remainder delivered.
    expect(batches.map((b) => b.map((i) => i.n))).toEqual([[4]]);
    expect(box.size).toBe(0);
  });

  it("peekBatch exposes the batchable prefix without dequeuing", () => {
    const box = new Outbox<Item>(opts([]));
    box.push({ session: "a", n: 1 });
    box.push({ session: "a", n: 2 });
    box.push({ session: "b", n: 3 });
    expect(box.peekBatch(8).map((i) => i.n)).toEqual([1, 2]);
    expect(box.size).toBe(3);
  });
});

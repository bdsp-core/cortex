import { afterEach, describe, expect, it, vi } from "vitest";

import type { EngineWorkerResponse } from "../engine/worker_protocol";
import { EngineClient } from "./engineClient";

class FakeWorker {
  static latest: FakeWorker | null = null;
  onmessage: ((event: MessageEvent<EngineWorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  readonly postMessage = vi.fn();
  readonly terminate = vi.fn();

  constructor() {
    FakeWorker.latest = this;
  }

  emit(message: EngineWorkerResponse): void {
    this.onmessage?.({ data: message } as MessageEvent<EngineWorkerResponse>);
  }
}

describe("engine client render-boundary telemetry", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    FakeWorker.latest = null;
  });

  it("routes one prefetch hint and measures answer through media readiness", () => {
    vi.stubGlobal("Worker", FakeWorker);
    const now = vi.spyOn(performance, "now");
    now.mockReturnValueOnce(100).mockReturnValueOnce(160).mockReturnValueOnce(220);
    const performanceEvents: unknown[] = [];
    const onPrefetch = vi.fn();
    const client = new EngineClient({
      onItem: vi.fn(), onPrefetch,
      onPerformance: (event) => performanceEvents.push(event),
      onDone: vi.fn(),
    });
    const worker = FakeWorker.latest!;

    client.answer(2);
    worker.emit({ type: "prefetch", trialIndex: 1, segId: 41 });
    worker.emit({ type: "item", trialIndex: 1, taskK: 2, segId: 41 });
    client.mediaReady(0);
    client.mediaReady(1);
    client.mediaReady(1);

    expect(onPrefetch).toHaveBeenCalledOnce();
    expect(onPrefetch).toHaveBeenCalledWith({ trialIndex: 1, segId: 41 });
    expect(performanceEvents).toEqual([
      { kind: "answer_to_item", trialIndex: 1, durationMs: 60 },
      { kind: "answer_to_media_ready", trialIndex: 1, durationMs: 120 },
    ]);
    client.dispose();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
});

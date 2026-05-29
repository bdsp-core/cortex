// Main-thread client for the engine Web Worker (engine/worker.ts). Keeps the
// particle filter off the UI thread; the client just relays init/answer/abort
// and surfaces item/trial/done/error events.
export class EngineClient {
    handlers;
    worker;
    constructor(handlers) {
        this.handlers = handlers;
        this.worker = new Worker(new URL("../engine/worker.ts", import.meta.url), {
            type: "module",
        });
        this.worker.onmessage = (ev) => {
            const m = ev.data;
            switch (m.type) {
                case "item":
                    this.handlers.onItem({ trialIndex: m.trialIndex, taskK: m.taskK, segId: m.segId });
                    break;
                case "trial":
                    this.handlers.onTrial?.(m.diag);
                    break;
                case "done":
                    this.handlers.onDone(m.result);
                    break;
                case "error":
                    this.handlers.onError?.(m.message);
                    break;
            }
        };
    }
    start(inputs, sessionId, seed) {
        this.worker.postMessage({ type: "init", inputs, sessionId, seed });
    }
    answer(pick) {
        this.worker.postMessage({ type: "answer", pick });
    }
    abort() {
        this.worker.postMessage({ type: "abort" });
    }
    dispose() {
        this.worker.terminate();
    }
}

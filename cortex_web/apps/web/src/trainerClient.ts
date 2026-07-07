// Main-thread client for the trainer Web Worker (trainer/worker.ts) — the
// trainer analogue of engineClient.ts. Keeps the filter + policy off the UI
// thread; the client relays init/next/submit and surfaces item/state/done/error.
import type { FilterParams } from "../trainer/filter";
import type { Choice } from "../trainer/policy";
import type { CandidateArrays, Cloud, TaskSnapshot } from "../trainer/session";

export interface TrainerInit {
  clouds: Cloud[];
  params: FilterParams;
  sigmaInf: number[];
  ellStars: number[];
  sigmaStars: number[];
  bank: CandidateArrays[];
  seed?: number;
  useMixture?: boolean;
  exactKernel?: boolean;
  minMargin?: number;
  maxConsec?: number;
}

export interface TrainerClientHandlers {
  onItem: (choice: Choice) => void;      // the next training item to present
  onDone: () => void;                    // all trained tasks graduated / budget hit
  onState?: (snapshot: TaskSnapshot[]) => void;  // per-task belief after a submit
  onError?: (message: string) => void;
}

export class TrainerClient {
  private worker: Worker;

  constructor(private handlers: TrainerClientHandlers) {
    this.worker = new Worker(new URL("../trainer/worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (ev: MessageEvent) => {
      const m = ev.data;
      switch (m.type) {
        case "item":
          this.handlers.onItem(m.choice);
          break;
        case "done":
          this.handlers.onDone();
          break;
        case "state":
          this.handlers.onState?.(m.snapshot);
          break;
        case "error":
          this.handlers.onError?.(m.message);
          break;
      }
    };
  }

  start(init: TrainerInit): void {
    this.worker.postMessage({ type: "init", ...init });
  }

  next(): void {
    this.worker.postMessage({ type: "next" });
  }

  submit(choice: Choice, y: number): void {
    this.worker.postMessage({ type: "submit", choice, y });
  }

  dispose(): void {
    this.worker.terminate();
  }
}

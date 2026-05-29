import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// SPA build. The engine Web Worker (engine/worker.ts) is imported with the
// `new Worker(new URL("./engine/worker.ts", import.meta.url), {type:"module"})`
// pattern, which Vite bundles automatically.
//
// If we ever shard particles across workers via SharedArrayBuffer, add the
// COOP/COEP headers here (server.headers) and on the CloudFront distribution.
export default defineConfig({
  plugins: [react()],
  worker: { format: "es" },
  build: {
    target: "es2022",
    sourcemap: true,
  },
});

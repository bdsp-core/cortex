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
  // In `vite dev` (port 5173) proxy /api to the FastAPI backend (port 8000) so
  // the SPA's same-origin /api calls work without CORS gymnastics. The EEG
  // bundle is served from public/ by Vite directly, so only /api is proxied.
  server: {
    proxy: {
      "/api": {
        target: process.env.CORTEX_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});

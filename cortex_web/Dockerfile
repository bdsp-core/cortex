# Single-image deploy: build the SPA, then serve SPA + EEG bundle + API from one
# uvicorn process (CORTEX_SERVE_STATIC=1). NOTE: the cortexeeg.org production
# deploy does NOT use this — it uses Caddy (static) + systemd (uvicorn pure API),
# see deploy/. This image is kept for container/portability use.
#
#   docker build -t cortex-web cortex_web/
#   docker run -p 8000:8000 -e CORTEX_ADMIN_TOKEN=... -e CORTEX_JWT_SECRET=... \
#     -v $PWD/bundle:/app/apps/web/public/bundle cortex-web
#
# The EEG bundle (apps/web/public/bundle/<version>/) is NOT baked in (large +
# PHI-adjacent) — mount it, or set CORTEX_BUNDLE_URL to an S3/CloudFront origin.

# ── stage 1: build the SPA (npm workspace) ──────────────────────
FROM node:20-slim AS spa
WORKDIR /build
COPY package.json package-lock.json* ./
COPY apps/web/package.json apps/web/package.json
RUN npm ci || npm install
COPY . .
RUN npm run build

# ── stage 2: runtime (pure-API service + optional static) ───────
FROM python:3.11-slim
WORKDIR /app
COPY services/api/requirements.txt ./services/api/requirements.txt
RUN pip install --no-cache-dir -r services/api/requirements.txt
COPY services ./services
COPY --from=spa /build/apps/web/dist ./apps/web/dist
# apps/web/public/bundle is mounted at runtime (see header).
EXPOSE 8000
ENV CORTEX_HOST=0.0.0.0 CORTEX_PORT=8000 CORTEX_SERVE_STATIC=1
WORKDIR /app/services
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

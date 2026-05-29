# Single-image deploy: build the SPA, then serve SPA + EEG bundle + API from
# one uvicorn process. Mirrors scripts/run_local.sh but containerized.
#
#   docker build -t cortex-web cortex_web/
#   docker run -p 8000:8000 \
#     -e CORTEX_ADMIN_TOKEN=... -e CORTEX_JWT_SECRET=... \
#     -v $PWD/data:/app/data cortex-web
#
# The EEG bundle (public/bundle/<version>/) is NOT baked into the image (it is
# large + PHI-adjacent). Mount it at runtime, or set CORTEX_BUNDLE_URL to an
# S3/CloudFront origin and serve only the API + SPA from here.

# ── stage 1: build the SPA ──────────────────────────────────────
FROM node:20-slim AS spa
WORKDIR /build
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
RUN npm run build

# ── stage 2: runtime ────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY server ./server
COPY --from=spa /build/dist ./dist
# public/bundle is mounted at runtime (see header).
EXPOSE 8000
ENV CORTEX_HOST=0.0.0.0 CORTEX_PORT=8000
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

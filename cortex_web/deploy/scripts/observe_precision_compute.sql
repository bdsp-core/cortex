-- Privacy-safe post-rollout observation for Precision browser computation.
--
-- Usage on production:
--   sudo -u postgres psql -d cortex \
--     -v since_utc=2026-07-20T05:05:00Z \
--     -f /opt/cortex/cortex_web/deploy/scripts/observe_precision_compute.sql
--
-- This is a read-only aggregate. It emits no participant code, email,
-- session id, responses, question ids, or result content outside the bounded
-- internal `_enginePerformance` summary.

\set ON_ERROR_STOP on
\if :{?since_utc}
\else
  \set since_utc '2026-07-20T05:05:00Z'
  \echo 'since_utc not supplied; using the public-rollout timestamp'
\endif

BEGIN TRANSACTION READ ONLY;

\echo 'Observation start:' :since_utc
\echo 'Session stamps and lifecycle'
WITH scoped AS (
  SELECT s.status, s.termination_policy, s.compute_mode,
         r.result::jsonb -> '_enginePerformance' AS perf
  FROM sessions s
  LEFT JOIN results r ON r.session_id = s.session_id
  WHERE s.started_utc >= :'since_utc'
)
SELECT termination_policy, compute_mode, status,
       count(*) AS sessions,
       count(*) FILTER (WHERE perf IS NOT NULL) AS with_performance
FROM scoped
GROUP BY termination_policy, compute_mode, status
ORDER BY termination_policy, compute_mode, status;

\echo 'Precision telemetry completeness and fallback totals'
WITH scoped AS (
  SELECT s.status, s.compute_mode,
         r.result::jsonb -> '_enginePerformance' AS perf
  FROM sessions s
  LEFT JOIN results r ON r.session_id = s.session_id
  WHERE s.started_utc >= :'since_utc'
    AND s.termination_policy = 'precision_v1'
)
SELECT count(*) AS precision_sessions,
       count(*) FILTER (WHERE status = 'complete') AS completed_sessions,
       count(*) FILTER (WHERE status = 'complete' AND perf IS NOT NULL)
         AS completed_with_telemetry,
       count(*) FILTER (WHERE status = 'complete' AND perf IS NULL)
         AS completed_missing_telemetry,
       coalesce(sum((perf ->> 'serialFallbackCount')::integer), 0)
         AS serial_fallbacks,
       coalesce(sum((perf #>> '{engineTotal,count}')::integer), 0)
         AS measured_steps
FROM scoped;

\echo 'Observed browser mode, device stratum, and per-session timing summaries'
WITH completed AS (
  SELECT r.result::jsonb -> '_enginePerformance' AS perf
  FROM sessions s
  JOIN results r ON r.session_id = s.session_id
  WHERE s.started_utc >= :'since_utc'
    AND s.status = 'complete'
    AND s.termination_policy = 'precision_v1'
    AND r.result::jsonb -> '_enginePerformance' IS NOT NULL
), metrics AS (
  SELECT coalesce(perf #>> '{profile,requested}', 'missing') AS requested_mode,
         coalesce(perf #>> '{profile,executionMode}', 'missing') AS actual_mode,
         CASE
           WHEN perf #>> '{profile,hardwareConcurrency}' IS NULL THEN 'unknown'
           WHEN (perf #>> '{profile,hardwareConcurrency}')::integer < 4 THEN '<4'
           WHEN (perf #>> '{profile,hardwareConcurrency}')::integer < 8 THEN '4-7'
           ELSE '8+'
         END AS hardware_bucket,
         (perf #>> '{answerToItem,p50Ms}')::double precision AS answer_p50_ms,
         (perf #>> '{answerToItem,p95Ms}')::double precision AS answer_p95_ms,
         (perf #>> '{answerToItem,maxMs}')::double precision AS answer_max_ms,
         coalesce((perf ->> 'serialFallbackCount')::integer, 0) AS fallbacks,
         coalesce((perf #>> '{engineTotal,count}')::integer, 0) AS measured_steps
  FROM completed
)
SELECT requested_mode, actual_mode, hardware_bucket,
       count(*) AS sessions,
       sum(measured_steps) AS measured_steps,
       sum(fallbacks) AS serial_fallbacks,
       round(avg(answer_p50_ms)::numeric, 1) AS mean_session_p50_ms,
       round(avg(answer_p95_ms)::numeric, 1) AS mean_session_p95_ms,
       round(max(answer_max_ms)::numeric, 1) AS worst_session_max_ms
FROM metrics
GROUP BY requested_mode, actual_mode, hardware_bucket
ORDER BY requested_mode, actual_mode, hardware_bucket;

COMMIT;

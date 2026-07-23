-- Privacy-safe aggregate observation for the provisional percentile preview.
-- Run in psql against the production CORTEX database. No participant identity,
-- free text, session id, or individual score is selected.

SELECT
  COALESCE(norm_id, '(unstamped)') AS norm_id,
  COALESCE(score_schema_version, '(none)') AS score_schema_version,
  status,
  COUNT(*) AS sessions
FROM sessions
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

SELECT
  COALESCE((r.result::jsonb #>> '{percentile,status}'), '(legacy/unstamped)')
    AS percentile_status,
  COUNT(*) AS results
FROM results r
GROUP BY 1
ORDER BY 1;

WITH scored AS (
  SELECT
    domain.key AS domain,
    (domain.value ->> 'estimate')::double precision AS estimate,
    (domain.value ->> 'lower')::double precision AS lower,
    (domain.value ->> 'upper')::double precision AS upper
  FROM results r
  CROSS JOIN LATERAL jsonb_each(
    COALESCE(r.result::jsonb #> '{percentile,domains}', '{}'::jsonb)
  ) AS domain
  WHERE r.result::jsonb #>> '{percentile,status}' = 'available'
)
SELECT
  domain,
  COUNT(*) AS n,
  ROUND(AVG(estimate)::numeric, 1) AS mean_percentile,
  ROUND(AVG(upper - lower)::numeric, 1) AS mean_interval_width,
  COUNT(*) FILTER (WHERE estimate < 5) AS below_5,
  COUNT(*) FILTER (WHERE estimate > 95) AS above_95
FROM scored
GROUP BY domain
ORDER BY domain;

SELECT
  COALESCE((summary::jsonb #>> '{percentile,status}'), '(unstamped)') AS status,
  COUNT(*) AS training_sessions
FROM training_sessions
WHERE status = 'complete'
GROUP BY 1
ORDER BY 1;

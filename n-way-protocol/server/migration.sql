-- Additive protocol migration. This file is an isolated specification and is
-- not executed against cortex_web until explicit production approval.
ALTER TABLE sessions ADD COLUMN engine_profile_id TEXT NOT NULL DEFAULT 'precision_binary_ovr_v1';
ALTER TABLE sessions ADD COLUMN response_model TEXT NOT NULL DEFAULT 'binary_ovr_v1';
ALTER TABLE sessions ADD COLUMN response_artifact_id TEXT;
ALTER TABLE sessions ADD COLUMN response_artifact_sha256 TEXT;
ALTER TABLE sessions ADD COLUMN selector_version TEXT NOT NULL DEFAULT 'binary_totalvar_v1';
ALTER TABLE sessions ADD COLUMN engine_algorithm_version TEXT NOT NULL DEFAULT 'cortex_web_binary_current';

-- Raw trials.pick is already authoritative and intentionally requires no
-- migration.  is_correct must be interpreted as gold correctness, never as
-- the derived asked-class marginal.


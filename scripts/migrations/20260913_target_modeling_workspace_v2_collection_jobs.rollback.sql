-- Operational P3 rollback. Stop workers and disable the V2 feature flag.
-- Preserve job history and dataset snapshots for safe recovery.
BEGIN;
UPDATE public.ti_admin_settings SET target_modeling_workspace_v2 = FALSE WHERE setting_key = 'global';
COMMIT;


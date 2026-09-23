-- Operational P2 rollback: disable V2 and preserve all plan versions and
-- review metadata. Column removal is intentionally not performed during the
-- rollback window; cleanup requires a separately approved destructive change.
BEGIN;
UPDATE public.ti_admin_settings SET target_modeling_workspace_v2 = FALSE WHERE setting_key = 'global';
COMMIT;

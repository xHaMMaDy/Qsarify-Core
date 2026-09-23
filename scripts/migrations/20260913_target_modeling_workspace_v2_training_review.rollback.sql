-- Operational P4 rollback: disable V2 and preserve training/candidate history.
BEGIN;
UPDATE public.ti_admin_settings SET target_modeling_workspace_v2 = FALSE WHERE setting_key = 'global';
COMMIT;


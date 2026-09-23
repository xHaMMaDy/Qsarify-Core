-- Operational rollback for P1. Preserve Study and membership data; disable V2.
BEGIN;
UPDATE public.ti_admin_settings
SET target_modeling_workspace_v2 = FALSE
WHERE setting_key = 'global';
COMMIT;


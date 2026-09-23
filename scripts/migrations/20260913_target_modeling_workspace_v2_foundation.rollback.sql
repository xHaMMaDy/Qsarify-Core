-- Safe operational rollback for Target Modeling Workspace V2 P0.
-- Historical additive data is retained. Destructive cleanup requires a
-- separate backup and must wait for the documented retention period.

BEGIN;

UPDATE public.ti_admin_settings
SET target_modeling_workspace_v2 = FALSE
WHERE setting_key = 'global';

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime'
      AND schemaname = 'public'
      AND tablename = 'ti_job_events'
  ) THEN
    EXECUTE 'ALTER PUBLICATION supabase_realtime DROP TABLE public.ti_job_events';
  END IF;
END;
$$;

COMMIT;

-- Operational rollback: restore the historical fresh-install default without
-- changing any existing administrator-configured quota value.
BEGIN;
ALTER TABLE public.ti_admin_settings
  ALTER COLUMN public_daily_analyses SET DEFAULT 3;
COMMIT;

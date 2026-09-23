-- Target Modeling Workspace V2 P6: align fresh-install public search default
-- with the current product limit. Existing administrator values are preserved.
BEGIN;
ALTER TABLE public.ti_admin_settings
  ALTER COLUMN public_daily_analyses SET DEFAULT 10;
COMMENT ON COLUMN public.ti_admin_settings.public_daily_analyses IS
  'Admin-configurable daily public search quota; fresh installs default to 10.';
COMMIT;

-- Target Modeling Workspace V2 P6: durable report-chat quotas.
-- Chat consumes the existing atomic quota RPC under event_type=chat_request.
-- Existing analysis quotas and administrator values are preserved.

ALTER TABLE public.ti_admin_settings
  ADD COLUMN IF NOT EXISTS public_daily_chat_requests INTEGER NOT NULL DEFAULT 30
    CHECK (public_daily_chat_requests >= 0),
  ADD COLUMN IF NOT EXISTS authenticated_daily_chat_requests INTEGER NOT NULL DEFAULT 100
    CHECK (authenticated_daily_chat_requests >= 0);

COMMENT ON COLUMN public.ti_admin_settings.public_daily_chat_requests IS
  'Durable per-anonymous-identity report-chat quota; provider calls remain server-side.';
COMMENT ON COLUMN public.ti_admin_settings.authenticated_daily_chat_requests IS
  'Durable per-user report-chat quota; provider calls remain server-side.';

-- Recreate the public capability function so non-secret chat limits are visible
-- to the UI and operational checks without exposing provider settings or keys.
DROP FUNCTION IF EXISTS public.get_ti_capabilities();

CREATE OR REPLACE FUNCTION public.get_ti_capabilities()
RETURNS TABLE (
  enabled BOOLEAN,
  access_mode TEXT,
  chat_layout TEXT,
  target_modeling_workspace_v2 BOOLEAN,
  public_daily_analyses INTEGER,
  authenticated_daily_analyses INTEGER,
  public_daily_chat_requests INTEGER,
  authenticated_daily_chat_requests INTEGER,
  max_uploads_per_analysis INTEGER,
  max_upload_bytes BIGINT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    s.enabled,
    s.access_mode,
    s.chat_layout,
    s.target_modeling_workspace_v2,
    s.public_daily_analyses,
    s.authenticated_daily_analyses,
    s.public_daily_chat_requests,
    s.authenticated_daily_chat_requests,
    s.max_uploads_per_analysis,
    s.max_upload_bytes
  FROM public.ti_admin_settings AS s
  WHERE s.setting_key = 'global'
  LIMIT 1;
$$;

REVOKE ALL ON FUNCTION public.get_ti_capabilities() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_ti_capabilities() TO anon, authenticated;

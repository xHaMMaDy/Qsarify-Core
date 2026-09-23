-- Operational rollback for durable report-chat quotas.
-- Preserve columns and history; disable chat quota consumption by setting the
-- limits to zero, while retaining the V2 feature flag as an independent gate.

UPDATE public.ti_admin_settings
SET public_daily_chat_requests = 0,
    authenticated_daily_chat_requests = 0,
    updated_at = NOW()
WHERE setting_key = 'global';

COMMENT ON TABLE public.ti_usage_events IS
  'Usage history is preserved during operational rollback; report-chat quotas are disabled.';
